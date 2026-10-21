import torch
import torch.nn as nn
import math

"""================================================================================
                                Activation Functions
================================================================================"""


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU)
    - 원래 BERT에서 사용된 버전
    - OpenAI GPT의 GELU와는 약간 다름
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def leakyrelu(x):
    """LeakyReLU (slope=0.2)"""
    return nn.functional.leaky_relu_(x, 0.2)


class CustomAct(nn.Module):
    """
    Custom Activation Layer
    - 문자열로 활성화 함수 선택 가능 ("gelu", "leakyrelu")
    """

    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu

    def forward(self, x):
        return self.act_layer(x)


"""================================================================================
                                MLP Block
================================================================================"""


class Mlp(nn.Module):
    """
    MLP Block
    - Linear → Activation → Dropout → Linear → Dropout
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=gelu,
        drop=0.2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


"""================================================================================
                                Transformer Encoder
================================================================================"""


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Block (Pre-LayerNorm)
    - LayerNorm → Multi-Head Attention → Residual
    - LayerNorm → FeedForward (MLP) → Residual
    """

    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Attention + Residual
        x = (
            x
            + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        )
        # MLP + Residual
        x = x + self.mlp(self.norm2(x))
        return x


"""================================================================================
                                TCGAN1D Generator
================================================================================"""


class TCGAN1D_Generator(nn.Module):
    """
    Generator (1D Signal 생성)
    - 입력: [batch, noise_dim]
    - 출력: [batch, output_length]
    """

    def __init__(self, noise_dim=256, embed_dim=512, output_length=200):
        super().__init__()

        self.output_length = output_length
        self.embed_dim = embed_dim
        self.initial_length = 50  # 초기 시퀀스 길이

        # Step 1: Noise → Embedding
        # [batch, 256] → [batch, 50*512]
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, self.initial_length * embed_dim),
            nn.Dropout(0.2),
        )

        # Stage 1: Transformer Encoders (5 layers)
        self.stage1_transformers = nn.ModuleList(
            [TransformerEncoder(dim=embed_dim) for _ in range(5)]
        )

        # Upsample: 50 → 100
        self.upsample1 = nn.Upsample(scale_factor=2, mode="linear")
        self.unflatten1 = nn.Linear(embed_dim, embed_dim)

        # Stage 2: Transformer Encoders (4 layers)
        self.stage2_transformers = nn.ModuleList(
            [TransformerEncoder(dim=embed_dim) for _ in range(4)]
        )

        # Upsample: 100 → 200
        self.upsample2 = nn.Upsample(scale_factor=2, mode="linear")
        self.unflatten2 = nn.Linear(embed_dim, embed_dim)

        # Final projection: [batch, 200, 512] → [batch, 200, 1]
        self.to_signal = nn.Linear(embed_dim, 1)

    def forward(self, z):
        # Noise → Embedding
        x = self.mlp(z)  # [batch, 50*512]
        x = x.view(-1, self.initial_length, self.embed_dim)  # [batch, 50, 512]

        # Stage 1 Transformers
        for transformer in self.stage1_transformers:
            x = transformer(x)

        # Upsample: 50 → 100
        x = x.transpose(1, 2)  # [batch, 512, 50]
        x = self.upsample1(x)  # [batch, 512, 100]
        x = x.transpose(1, 2)  # [batch, 100, 512]

        # Stage 2 Transformers
        for transformer in self.stage2_transformers:
            x = transformer(x)

        # Upsample: 100 → 200
        x = x.transpose(1, 2)  # [batch, 512, 100]
        x = self.upsample2(x)  # [batch, 512, 200]
        x = x.transpose(1, 2)  # [batch, 200, 512]

        # Linear Unflatten
        x = self.unflatten2(x)

        # Final projection → 1D signal
        x = self.to_signal(x).squeeze(-1)  # [batch, 200]

        return x


"""================================================================================
                                TCGAN1D Discriminator
================================================================================"""


class TCGAN1D_Discriminator(nn.Module):
    """
    Discriminator (1D Signal 판별)
    - 입력: [batch, signal_length]
    - 출력: [batch, 1] (진짜/가짜 판별)
    """

    def __init__(self, signal_length=200):
        super().__init__()

        self.conv_layers = nn.ModuleList()

        # Conv1: [batch, 1, 200] → [batch, 64, 100]
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1
            )
        )

        # Conv2~4: 채널 확장 + BatchNorm + LeakyReLU
        channels = [64, 128, 256, 512]
        for i in range(len(channels) - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Final Conv: [batch, 512, L] → [batch, 1, L]
        self.final_conv = nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1)

        # Sigmoid (Binary Classification)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 입력: [batch, 200] → [batch, 1, 200]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv1
        x = self.conv_layers[0](x)

        # Conv2~4
        for i in range(1, len(self.conv_layers)):
            x = self.conv_layers[i](x)

        # Final Conv
        x = self.final_conv(x)

        # Global Average Pooling → [batch, 1]
        x = torch.mean(x, dim=2)

        return x
