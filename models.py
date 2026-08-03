#!/usr/bin/env python3
"""Dual-branch multimodal fusion model with hierarchical gated co-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2),
        )
        self.image_encoder_patch_dense_1 = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )
        self.image_encoder_patch_dense_2 = nn.Linear(embed_dim, embed_dim)
        self.image_encoder_patch_norm_1 = nn.LayerNorm(embed_dim)

    def forward(self, x, training=True, verbose=False):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)

        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)
        patches = x.permute(0, 2, 3, 1).reshape(batch_size, h * w, -1)
        patches = self.image_encoder_patch_dense_1(patches)
        patches = self.image_encoder_patch_dense_2(patches)
        patches = self.image_encoder_patch_norm_1(patches)
        return patches


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_length=None, embed_dim=256):
        super().__init__()
        if max_length is None:
            max_length = config.get_max_token_length()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.lstm1 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=embed_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.text_encoder_token_dense_1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.text_encoder_token_dense_2 = nn.Linear(embed_dim, embed_dim)
        self.text_encoder_token_norm_1 = nn.LayerNorm(embed_dim)

    def forward(self, x, training=True, verbose=False):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.text_encoder_token_dense_1(x)
        x = self.text_encoder_token_dense_2(x)
        x = self.text_encoder_token_norm_1(x)
        return x


class HierarchicalCoAttention(nn.Module):
    """Local and global gated cross-attention with global-to-local feedback."""

    def __init__(self, embed_dim, num_heads, instance_name="coattn"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.instance_name = instance_name

        self.local_image_gate_weights = nn.Parameter(torch.randn(embed_dim))
        self.local_text_gate_weights = nn.Parameter(torch.randn(embed_dim))
        self.global_image_gate_weights = nn.Parameter(torch.randn(embed_dim))
        self.global_text_gate_weights = nn.Parameter(torch.randn(embed_dim))

        self.cross_attention1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.global_cross_attention1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.global_cross_attention2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.global_norm1 = nn.LayerNorm(embed_dim)
        self.global_norm2 = nn.LayerNorm(embed_dim)
        self.global_norm3 = nn.LayerNorm(embed_dim)
        self.global_norm4 = nn.LayerNorm(embed_dim)

        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.global_ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.global_ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, image_tokens, text_tokens):
        attended_image, _ = self.cross_attention1(
            query=image_tokens, key=text_tokens, value=text_tokens
        )
        local_image_gate = torch.sigmoid(self.local_image_gate_weights).view(1, 1, self.embed_dim)
        gated_image = local_image_gate * attended_image + (1 - local_image_gate) * image_tokens
        image_tokens = self.norm1(image_tokens + gated_image)
        image_tokens = self.norm2(image_tokens + self.ffn1(image_tokens))

        attended_text, _ = self.cross_attention2(
            query=text_tokens, key=image_tokens, value=image_tokens
        )
        local_text_gate = torch.sigmoid(self.local_text_gate_weights).view(1, 1, self.embed_dim)
        gated_text = local_text_gate * attended_text + (1 - local_text_gate) * text_tokens
        text_tokens = self.norm3(text_tokens + gated_text)
        text_tokens = self.norm4(text_tokens + self.ffn2(text_tokens))

        global_image_token = torch.mean(image_tokens, dim=1, keepdim=True)
        global_text_token = torch.mean(text_tokens, dim=1, keepdim=True)

        attended_global_image, _ = self.global_cross_attention1(
            query=global_image_token, key=text_tokens, value=text_tokens
        )
        global_image_gate = torch.sigmoid(self.global_image_gate_weights).view(1, 1, self.embed_dim)
        gated_global_image = (
            global_image_gate * attended_global_image
            + (1 - global_image_gate) * global_image_token
        )
        global_image_token = self.global_norm1(global_image_token + gated_global_image)
        global_image_token = self.global_norm2(
            global_image_token + self.global_ffn1(global_image_token)
        )

        attended_global_text, _ = self.global_cross_attention2(
            query=global_text_token, key=image_tokens, value=image_tokens
        )
        global_text_gate = torch.sigmoid(self.global_text_gate_weights).view(1, 1, self.embed_dim)
        gated_global_text = (
            global_text_gate * attended_global_text
            + (1 - global_text_gate) * global_text_token
        )
        global_text_token = self.global_norm3(global_text_token + gated_global_text)
        global_text_token = self.global_norm4(
            global_text_token + self.global_ffn2(global_text_token)
        )

        image_tokens = image_tokens + global_image_token.expand(-1, image_tokens.size(1), -1)
        text_tokens = text_tokens + global_text_token.expand(-1, text_tokens.size(1), -1)
        return image_tokens, text_tokens


class BranchEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, name="branch"):
        super().__init__()
        self.name = name
        self.co_attn_layers = nn.ModuleList(
            [
                HierarchicalCoAttention(
                    embed_dim, num_heads, instance_name=f"{name}_coattn_{i + 1}"
                )
                for i in range(num_layers)
            ]
        )
        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, image_tokens, text_tokens):
        for layer in self.co_attn_layers:
            image_tokens, text_tokens = layer(image_tokens, text_tokens)
        image_emb = torch.mean(image_tokens, dim=1)
        text_emb = torch.mean(text_tokens, dim=1)
        image_emb = self.image_proj(image_emb)
        text_emb = self.text_proj(text_emb)
        return image_emb, text_emb


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, _, embeddings):
        image_embeddings, text_embeddings = embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
        similarity_matrix = similarity_matrix / self.temperature
        labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        return (loss_i2t + loss_t2i) / 2


class SynergyLoss(nn.Module):
    """Maximize similarity for matching image-text pairs."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, _, embeddings):
        image_embeddings, text_embeddings = embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
        similarity_matrix = similarity_matrix / self.temperature
        labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        return (loss_i2t + loss_t2i) / 2


class DifferenceLoss(nn.Module):
    """Encourage low similarity on matching pairs via anti-diagonal targets."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, _, embeddings):
        image_embeddings, text_embeddings = embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
        similarity_matrix = similarity_matrix / self.temperature
        batch_size = image_embeddings.size(0)
        anti_labels = torch.arange(1, batch_size + 1, device=image_embeddings.device) % batch_size
        loss_i2t = F.cross_entropy(similarity_matrix, anti_labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), anti_labels)
        return (loss_i2t + loss_t2i) / 2


class MultimodalFusion(nn.Module):
    def __init__(self, vocab_size, embed_dim=None, num_heads=None, num_layers=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = config.get_embed_dim()
        if num_heads is None:
            num_heads = config.get_current_config()["num_heads"]
        if num_layers is None:
            num_layers = config.get_current_config()["num_layers"]

        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim=embed_dim)
        self.synergy_branch = BranchEncoder(embed_dim, num_heads, num_layers, name="synergy")
        self.difference_branch = BranchEncoder(embed_dim, num_heads, num_layers, name="difference")

    def forward(self, inputs, training=False, verbose=False, return_branch_embeddings=False):
        images, texts = inputs
        image_tokens = self.image_encoder(images, training=training, verbose=verbose)
        text_tokens = self.text_encoder(texts, training=training, verbose=verbose)

        synergy_img_emb, synergy_txt_emb = self.synergy_branch(image_tokens, text_tokens)
        diff_img_emb, diff_txt_emb = self.difference_branch(image_tokens, text_tokens)

        final_image_emb = F.normalize((synergy_img_emb + diff_img_emb) / 2, p=2, dim=-1)
        final_text_emb = F.normalize((synergy_txt_emb + diff_txt_emb) / 2, p=2, dim=-1)

        if return_branch_embeddings:
            return (
                final_image_emb,
                final_text_emb,
                synergy_img_emb,
                synergy_txt_emb,
                diff_img_emb,
                diff_txt_emb,
            )
        return final_image_emb, final_text_emb
