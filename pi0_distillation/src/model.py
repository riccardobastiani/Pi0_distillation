import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from transformers import CLIPTokenizer, CLIPTextModel


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self._pos_x: torch.Tensor | None = None
        self._pos_y: torch.Tensor | None = None

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature_map.shape
        if self._pos_x is None or self._pos_x.shape[0] != h * w:
            pos_x, pos_y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            self._pos_x = torch.from_numpy(pos_x.flatten()).float().to(feature_map.device)
            self._pos_y = torch.from_numpy(pos_y.flatten()).float().to(feature_map.device)

        flat = feature_map.view(b, c, -1) / self.temperature
        softmax_attention = F.softmax(flat, dim=2)
        expected_x = torch.sum(self._pos_x * softmax_attention, dim=2)
        expected_y = torch.sum(self._pos_y * softmax_attention, dim=2)
        return torch.cat([expected_x, expected_y], dim=1)


class SpatialResNetEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 512,
        unfreeze_last_n: int = 2,
        dropout: float = 0.1,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        all_layers = list(resnet.children())[:-2]

        if input_channels != 3:
            original_conv = all_layers[0]
            new_conv = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = original_conv.weight
                for idx in range(3, input_channels):
                    new_conv.weight[:, idx] = original_conv.weight[:, idx % 3]
            all_layers[0] = new_conv

        for layer in all_layers:
            for param in layer.parameters():
                param.requires_grad = False

        if unfreeze_last_n > 0:
            for layer in all_layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        if input_channels != 3:
            for param in all_layers[0].parameters():
                param.requires_grad = True

        self.backbone = nn.Sequential(*all_layers)
        self.spatial_softmax = SpatialSoftmax()
        self.projector = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = self.spatial_softmax(feats)
        return self.projector(pooled)


class PromptEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.projector = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=77,
        ).to(device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return self.projector(outputs.pooler_output)


class RecursiveBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        combined = h + x_cond
        combined = combined.unsqueeze(1)
        attn_out, _ = self.attention(combined, combined, combined)
        combined = self.norm1(combined + self.dropout1(attn_out))
        mlp_out = self.mlp(combined)
        return self.norm2(combined + mlp_out).squeeze(1)


class TRMPolicyFiLM(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 128,
        visual_dim: int = 128,
        num_heads: int = 4,
        num_recursions: int = 6,
        dropout: float = 0.05,
        proprio_dim: int = 7,
        unfreeze_last_n: int = 2,
        use_frame_stacking: bool = True,
        num_stacked_frames: int = 4,
    ) -> None:
        super().__init__()
        self.num_recursions = num_recursions
        input_channels = 3 * num_stacked_frames if use_frame_stacking else 3

        self.encoder = SpatialResNetEncoder(
            output_dim=visual_dim,
            unfreeze_last_n=unfreeze_last_n,
            dropout=dropout,
            input_channels=input_channels,
        )
        self.text_encoder = PromptEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.proprio_processor = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.conditioning_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.gamma_proj = nn.Linear(hidden_dim, visual_dim)
        self.beta_proj = nn.Linear(hidden_dim, visual_dim)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.bias)

        self.fusion_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.recursive_block = RecursiveBlock(hidden_dim, num_heads, dropout)
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        proprio: torch.Tensor,
        prompts: list[str],
    ) -> torch.Tensor:
        x_vis = self.encoder(obs)
        x_prop = self.proprio_processor(proprio)
        x_text = self.text_encoder(prompts, device=obs.device)
        cond = self.conditioning_proj(torch.cat([x_text, x_prop], dim=-1))

        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)
        x_fused = gamma * x_vis + beta

        x_cond = self.fusion_proj(x_fused)
        h = x_cond.clone()
        for _ in range(self.num_recursions):
            h = self.recursive_block(h, x_cond)
        return self.action_head(h)
