import os
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from typing import Tuple, Dict
from tqdm import tqdm
from peft import get_peft_model, LoraConfig

# ==========================================
# 导入基础组件
# ==========================================
# from transformers import AutoModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from REVE.modeling_reve import Reve, TransformerBackbone, FourierEmb4D, GEGLU
except ImportError:
    print("Warning: REVE internal components not found. Ensure REVE module is in src/REVE.")

# ==========================================
# 1. 核心架构定义 (掩码生成器与包装器)
# ==========================================
class FastSpatioTemporalMaskGenerator(nn.Module):
    def __init__(self, mask_ratio: float = 0.55, spatial_radius: float = 0.03, 
                 temp_radius: int = 2, drop_ratio: float = 0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.spatial_radius = spatial_radius
        self.temp_radius = temp_radius
        self.drop_ratio = drop_ratio

    def forward(self, pos: torch.Tensor, num_patches: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, _ = pos.shape
        T = num_patches
        device = pos.device

        noise = torch.rand(B, C, T, device=device)
        noise = F.max_pool1d(
            noise, 
            kernel_size=2 * self.temp_radius + 1, 
            stride=1, 
            padding=self.temp_radius
        )

        dist = torch.cdist(pos, pos)
        adj = (dist <= self.spatial_radius).float()
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        noise = torch.bmm(adj, noise)

        drop_mask = (torch.rand(B, C, 1, device=device) < self.drop_ratio).float()
        noise = noise + drop_mask * 100.0

        noise_flat = noise.view(B, C * T)
        ids_shuffle = torch.argsort(noise_flat, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        len_keep = int((C * T) * (1 - self.mask_ratio))
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:]

        return ids_keep, ids_masked, ids_restore


class ReveForPreTraining(nn.Module):
    def __init__(self, encoder, decoder_depth: int = 2, mask_ratio: float = 0.55):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size
        
        self.mask_generator = FastSpatioTemporalMaskGenerator(
            mask_ratio=mask_ratio, spatial_radius=0.03, temp_radius=12, drop_ratio=0.1
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        config = self.encoder.config
        self.decoder = TransformerBackbone(
            dim=self.embed_dim, depth=decoder_depth, heads=config.heads, 
            head_dim=config.head_dim, mlp_dim=int(self.embed_dim * config.mlp_dim_ratio), geglu=config.use_geglu
        )
        
        self.primary_head = nn.Linear(self.embed_dim, self.patch_size)
        
        hidden_dim = self.embed_dim * 2
        self.secondary_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim * 2),
            GEGLU() if config.use_geglu else nn.GELU(),
            nn.Linear(hidden_dim, self.patch_size)
        )
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, eeg: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, L = eeg.shape
        overlap_size = self.encoder.overlap_size
        patch_size = self.encoder.patch_size
        patches = eeg.unfold(dimension=2, size=patch_size, step=patch_size - overlap_size)
        _, _, T, _p = patches.shape
        
        target = rearrange(patches, "b c t p -> b (c t) p")
        
        pos_4d = FourierEmb4D.add_time_patch(pos, T)
        pos_embed = self.encoder.ln(self.encoder.fourier4d(pos_4d) + self.encoder.mlp4d(pos_4d))
        
        x = rearrange(self.encoder.to_patch_embedding(patches), "b c t e -> b (c t) e", c=C, t=T, e=self.embed_dim)
        x = x + pos_embed
        
        ids_keep, ids_masked, ids_restore = self.mask_generator(pos, T)
        
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        
        encoded_visible = self.encoder.transformer(x_visible)
        
        query_out = self.encoder.cls_query_token.expand(B, -1, -1)
        attn_scores = torch.matmul(query_out, encoded_visible.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        global_token = torch.matmul(attn_weights, encoded_visible).squeeze(1)  
        
        len_keep = ids_keep.shape[1]
        len_masked = (C * T) - len_keep
        mask_tokens = self.mask_token.expand(B, len_masked, -1)
        
        x_full = torch.cat([encoded_visible, mask_tokens], dim=1)  
        x_restored = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        x_restored = x_restored + pos_embed
        
        decoded = self.decoder(x_restored)
        pred_primary = self.primary_head(decoded)  
        pred_secondary = self.secondary_head(global_token)  
        
        pred_primary_masked = torch.gather(pred_primary, dim=1, index=ids_masked.unsqueeze(-1).expand(-1, -1, self.patch_size))
        target_masked = torch.gather(target, dim=1, index=ids_masked.unsqueeze(-1).expand(-1, -1, self.patch_size))
        primary_loss = F.l1_loss(pred_primary_masked, target_masked)
        
        target_global = target
        pred_secondary_expanded = pred_secondary.unsqueeze(1).expand_as(target_global)
        secondary_loss = F.l1_loss(pred_secondary_expanded, target_global)
        
        total_loss = primary_loss + 0.1 * secondary_loss
        
        return {
            "total_loss": total_loss,
            "primary_loss": primary_loss,
            "secondary_loss": secondary_loss
        }

# ==========================================
# 2. 数据加载器定义
# ==========================================
class RunsDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        run = self.data[idx]
        return run['eeg'], run['pos']

def collate_fn(batch):
    return batch[0][0], batch[0][1]

class PTDataLoaderFactory:
    @staticmethod
    def create_loader(pt_path: str):
        print(f"Loading cached tensors from {pt_path} ... (Requires huge RAM)")
        payload = torch.load(pt_path, weights_only=False)
        dataset = RunsDataset(payload)
        return DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True, collate_fn=collate_fn)

# ==========================================
# 3. LoRA 训练引擎
# ==========================================
class MAELoRATrainer:
    def __init__(self, model: nn.Module, lr: float = 1e-4, epochs: int = 5, start_epoch: int = 0, device: str = "cuda"):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.device = device
        
        # 只取requires_grad=True的参数，包括LoRA部分、Decoder、Heads、Mask Token
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        
    def _memory_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_task(self, dataloader: DataLoader, task_name: str, save_dir: str):
        print(f"--- 启动 MAE LoRA 训练: {task_name} ---")
        self.model.train()
        total_steps = 0
        
        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss = 0.0
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for step, batch in pbar:
                eeg_data, coords = batch
                eeg_data = eeg_data.to(self.device, dtype=torch.float16)
                coords = coords.to(self.device, dtype=torch.float16)
                
                # 微批次极致防 OOM 
                # 【急救】如果 4090 依然爆显存，我已经帮你把 4 降到了 2。如果还不行，就可以降为 1。
                mb_size = 4
                for i in range(0, eeg_data.shape[0], mb_size):
                    mb_eeg = eeg_data[i:i+mb_size]
                    mb_coords = coords.unsqueeze(0).expand(mb_eeg.shape[0], -1, -1)
                    
                    if hasattr(self.model, "enable_input_require_grads"):
                        self.model.enable_input_require_grads()
                    elif hasattr(mb_eeg, "requires_grad_"):
                        mb_eeg.requires_grad_(True)
                        
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        outputs = self.model(eeg=mb_eeg, pos=mb_coords)
                        loss = outputs["total_loss"]
                        prim_loss = outputs["primary_loss"]
                        sec_loss = outputs["secondary_loss"]
                        
                    if torch.isnan(loss):
                        print(f"警告: 侦测到 NaN loss. 单批次跳过保护.")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    epoch_loss += loss.item()
                    total_steps += 1
                    
                    if total_steps % 10 == 0:
                        pbar.set_postfix({
                            "L1_Tot": f"{loss.item():.4f}",
                            "L1_Prim": f"{prim_loss.item():.4f}",
                            "L1_Sec": f"{sec_loss.item():.4f}"
                        })
                        
            print(f"Epoch {epoch+1} / {task_name} - Avg Loss: {epoch_loss/max(1, total_steps):.4f}")
            self._memory_cleanup()
            
            # 每一个 Epoch 结束后立刻保存一份独立的快照
            epoch_save_path = os.path.join(save_dir, f"mae_lora_{task_name}_epoch_{epoch+1}")
            os.makedirs(epoch_save_path, exist_ok=True)
            self.model.save_pretrained(epoch_save_path, safe_serialization=True)
            print(f"[{task_name}] 第 {epoch+1} 轮的 LoRA 与解码器权重已安全保存至: {epoch_save_path}")
            
        print(f"--- {task_name} 所有 {self.epochs} 轮训练已圆满完成！ ---")

# ==========================================
# 4. 主执行入口
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reve_dir = os.path.join(current_dir, "REVE")
    
    print("1. 加载冻结的 REVE 基础模型 ...")
    base_encoder = Reve.from_pretrained(
        reve_dir, 
        use_safetensors=True, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    for param in base_encoder.parameters():
        param.requires_grad = False
    
    # [选填] 极限显存模式
    try:
        if hasattr(base_encoder, "gradient_checkpointing_enable"):
            base_encoder.gradient_checkpointing_enable()
            if hasattr(base_encoder.config, "use_cache"):
                base_encoder.config.use_cache = False
    except ValueError:
        print("注意: REVE 原生架构未提供梯度检查点接口，将不使用它（显存依然由 mb_size=4 安全兜底）。")
            
    print("2. 组装非对称 MAE 包装网络 ...")
    mae_model = ReveForPreTraining(encoder=base_encoder, decoder_depth=2, mask_ratio=0.55).to(device)
    
    # ==========================================
    # [新增] 断点续训配置
    # 如果你想从第 2 轮继续训练，只需取消下面的注释并填入正确的路径
    resume_from_checkpoint = os.path.join(current_dir, "models_out", "mae_lora_read_finetune_epoch_2")
    # ==========================================
    # 刚才你改了上一行，但忘了把下面这行 None 注释掉，导致路径被强行覆盖成 None 啦！我已经帮你注释掉了。
    # resume_from_checkpoint = None 
    start_epoch = 0
    
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"3. 正在从断点恢复权重: {resume_from_checkpoint} ...")
        from peft import PeftModel
        # 必须带 is_trainable=True，这样载入的权重才会挂上梯度，继续受优化器训练
        peft_model = PeftModel.from_pretrained(mae_model, resume_from_checkpoint, is_trainable=True)
        
        # 自动尝试从文件夹名字中提取开始的 Epoch (比如 ..._epoch_2 -> start_epoch=2)
        try:
            start_epoch = int(resume_from_checkpoint.split("_epoch_")[-1])
            print(f"成功解析到之前的进度，将从 第 {start_epoch + 1} 轮开始继续训练。")
        except:
            pass
    else:
        print("3. 全新初始化 PEFT LoRA (仅打击 Encoder，保存 Decoder) ...")
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["to_qkv", "to_out"], # 靶向打击 Attention 中的线性映射
            # 神仙用法：告诉 peft 把我们自定义包装器里的这些非 LoRA 残差块一并加入保存豪华套餐！
            modules_to_save=["mask_token", "decoder", "primary_head", "secondary_head"], 
            bias="none"
        )
        peft_model = get_peft_model(mae_model, config)
        
    peft_model.print_trainable_parameters()
    
    print("4. 启动训练循环 ...")
    trainer = MAELoRATrainer(peft_model, lr=1e-4, epochs=5, start_epoch=start_epoch, device=device)
    out_dir = os.path.join(current_dir, "models_out")
    
    # 真实测试数据：可以解注下方的真实数据逻辑
    dummy_pt_path = r'd:\python\Project\Multi-Paradigm pretrained large speech EEG model\neural_task_arithmetic\data\ds_kul_listen.pt'
    if os.path.exists(dummy_pt_path):
        loader = PTDataLoaderFactory.create_loader(dummy_pt_path)
        trainer.train_task(loader, task_name="kul_listen_finetune", save_dir=out_dir)
    else:
        print(f"Warning: Data file not found at {dummy_pt_path}. 脚本已备好，请确保已经跑过了 extract_kul.py！")
