import torch
import torch.nn as nn
# import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        return x, (H, W)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2)
        x = x + self.dropout(x2)
        x2 = self.norm2(x)
        x2 = self.ffn(x2)
        x = x + self.dropout(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ImageTransformer(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, img_size=(1024, 1024),
                 patch_size=16, embed_dim=768, depth=12, num_heads=12, ff_dim=3072, dropout=0.1):
        super(ImageTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads, ff_dim, dropout)
        self.linear_proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_channels = out_channels
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embedding(x)  # (B, N, D)
        x = self.transformer(x)  # (B, N, D)
        x = self.linear_proj(x)  # (B, N, patch_size*patch_size*out_channels)
        x = x.view(B, Hp, Wp, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, Hp * self.patch_size, Wp * self.patch_size)
        x = self.sigmoid(x)  # Apply Sigmoid activation
        return x


if __name__ == '__main__':
    # Example usage:
    img_size = (1024, 1024)
    model = ImageTransformer(in_channels=3, out_channels=2, img_size=img_size, patch_size=16, embed_dim=768, depth=12,
                             num_heads=12, ff_dim=3072)
    input_image = torch.randn(1, 3, *img_size)
    output_image = model(input_image)
    print(output_image.shape)  # Should output (1, 2, 1024, 1024)
