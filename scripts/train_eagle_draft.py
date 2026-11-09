#!/usr/bin/env python3
"""
EAGLE Speculative Draft Head Distillation Trainer for KriyaRT.

Trains the lightweight 1-layer EAGLE draft head (FC Fuse + Decoder Layer)
against a frozen target LLM using feature regression and classification distillation.

References:
    - EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (Li et al., 2024)
    - KriyaRT Native Binary Format (KRYAEGL1)

Usage:
    python3 scripts/train_eagle_draft.py \
        --model_path models/smollm2-135m-instruct \
        --output_path models/smollm2-135m-instruct-kriya-q4/kriya_eagle_draft_v1.bin \
        --epochs 3 \
        --lr 1e-3 \
        --batch_size 16
"""

import argparse
import hashlib
import json
import os
import struct
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class EagleDraftHead(nn.Module):
    """EAGLE Autoregression Draft Head:
    1. FC Fuse layer reducing [tok_emb; prev_feature] (2 * H) -> H.
    2. Single decoder projection layer (H -> H) with residual connection.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc_fuse = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.decoder_layer = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize with model-aligned identity skip for prev_feature
        with torch.no_grad():
            self.fc_fuse.weight.zero_()
            self.fc_fuse.weight[:, hidden_size:].copy_(torch.eye(hidden_size))
            self.decoder_layer.weight.zero_()

    def forward(
        self,
        tok_emb: torch.Tensor,
        prev_feature: torch.Tensor,
        noise_bound: float = 0.1,
    ) -> torch.Tensor:
        # In training, inject uniform noise U(-noise_bound, noise_bound) into prev_feature
        if self.training and noise_bound > 0.0:
            noise = (torch.rand_like(prev_feature) * 2.0 - 1.0) * noise_bound
            noisy_prev = prev_feature + noise
        else:
            noisy_prev = prev_feature

        fused = torch.cat([tok_emb, noisy_prev], dim=-1)
        reduced = self.fc_fuse(fused)
        delta = self.decoder_layer(reduced)
        next_feature = reduced + delta
        return next_feature


class FeatureTransitionDataset(Dataset):
    """Dataset of (prev_feature, advanced_token_id, target_next_feature, target_next_token_id)."""

    def __init__(
        self,
        prev_features: torch.Tensor,
        advanced_token_ids: torch.Tensor,
        target_next_features: torch.Tensor,
        target_next_token_ids: torch.Tensor,
    ):
        self.prev_features = prev_features
        self.advanced_token_ids = advanced_token_ids
        self.target_next_features = target_next_features
        self.target_next_token_ids = target_next_token_ids

    def __len__(self) -> int:
        return self.prev_features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.prev_features[idx],
            self.advanced_token_ids[idx],
            self.target_next_features[idx],
            self.target_next_token_ids[idx],
        )


def extract_features_from_prompts(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_len: int = 128,
) -> FeatureTransitionDataset:
    """Passes text through the frozen model to collect ground-truth hidden states."""
    model.eval()
    all_prev_f = []
    all_adv_t = []
    all_next_f = []
    all_next_t = []

    print(f"Extracting features from {len(prompts)} prompts...")
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            input_ids = enc.input_ids[0]
            if len(input_ids) < 3:
                continue

            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            # Last hidden state before LM head
            hidden_states = outputs.hidden_states[-1][0].float()  # [seq_len, hidden_size]

            for i in range(len(input_ids) - 2):
                all_prev_f.append(hidden_states[i].cpu())
                all_adv_t.append(input_ids[i].cpu())
                all_next_f.append(hidden_states[i + 1].cpu())
                all_next_t.append(input_ids[i + 1].cpu())

    prev_f = torch.stack(all_prev_f)
    adv_t = torch.stack(all_adv_t)
    next_f = torch.stack(all_next_f)
    next_t = torch.stack(all_next_t)
    print(f"Collected {prev_f.shape[0]} transition samples.")
    return FeatureTransitionDataset(prev_f, adv_t, next_f, next_t)


def save_kriya_binary_weights(
    head: EagleDraftHead,
    hidden_size: int,
    vocab_size: int,
    output_path: str,
):
    """Serializes weights in KriyaRT's KRYAEGL1 binary format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    magic = b"KRYAEGL1"
    version = 1
    reserved_flags = 0

    header = (
        magic
        + struct.pack("<I", version)
        + struct.pack("<Q", hidden_size)
        + struct.pack("<Q", vocab_size)
        + struct.pack("<I", reserved_flags)
    )

    fc_w = head.fc_fuse.weight.detach().cpu().to(torch.float32).numpy().tobytes()
    dec_w = head.decoder_layer.weight.detach().cpu().to(torch.float32).numpy().tobytes()

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(fc_w)
        f.write(dec_w)

    total_bytes = os.path.getsize(output_path)
    sha256 = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
    print(f"Saved EAGLE draft head artifact to {output_path} ({total_bytes} bytes, SHA-256: {sha256})")

    # Update manifest if present
    manifest_path = os.path.join(os.path.dirname(output_path), "kriya-artifact-manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r") as mf:
                manifest_data = json.load(mf)
            manifest_data["speculative_draft_file"] = {
                "file_name": os.path.basename(output_path),
                "byte_len": total_bytes,
                "sha256": sha256,
            }
            manifest_data["speculative_draft_head_bundled"] = True
            with open(manifest_path, "w") as mf:
                json.dump(manifest_data, mf, indent=2)
            print(f"Updated {manifest_path} with new artifact checksum.")
        except Exception as e:
            print(f"Warning: could not update manifest: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train EAGLE draft head for KriyaRT")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model or HF repo")
    parser.add_argument("--output_path", type=str, required=True, help="Output kriya_eagle_draft_v1.bin path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cls_weight", type=float, default=0.5, help="Classification distillation weight")
    parser.add_argument("--noise_bound", type=float, default=0.1, help="Feature perturbation noise bound")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on device: {device}")

    print(f"Loading base model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"Model: hidden_size={hidden_size}, vocab_size={vocab_size}")

    training_prompts = [
        "In computer science and artificial intelligence, inference engines optimize tensor operations to achieve low latency.",
        "On Apple Silicon systems, the GPU and Neural Engine share unified LPDDR memory with high-performance CPU cores.",
        "Over the past few years, the development of large language models has accelerated at an unprecedented pace.",
        "The quick brown fox jumps over the lazy dog. Natural language processing relies on attention mechanisms.",
        "Modern inference runtime systems leverage FlashAttention, Block-Sparse Row paged key-value caches, and 4-bit block quantization.",
        "Speculative decoding algorithms leverage smaller draft heads or speculative trees to propose multiple candidate tokens per step.",
        "Rust guarantees memory safety and thread safety using compile-time ownership semantics without a runtime garbage collector.",
        "Continuous batching allows new requests to join the inference engine without waiting for running sequences to finish.",
    ]

    dataset = extract_features_from_prompts(model, tokenizer, training_prompts, device)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    head = EagleDraftHead(hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    embedding_weight = model.get_input_embeddings().weight.detach()
    lm_head_weight = model.get_output_embeddings().weight.detach()

    print("Beginning training...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        total_reg = 0.0
        for prev_f, adv_t, next_f, next_t in train_loader:
            prev_f = prev_f.to(device)
            adv_t = adv_t.to(device)
            next_f = next_f.to(device)
            next_t = next_t.to(device)

            tok_emb = F.embedding(adv_t, embedding_weight)
            pred_f = head(tok_emb, prev_f, noise_bound=args.noise_bound)

            loss_reg = F.smooth_l1_loss(pred_f, next_f)

            # Distillation on logits
            pred_logits = F.linear(pred_f, lm_head_weight)
            target_logits = F.linear(next_f, lm_head_weight)

            loss_cls = F.kl_div(
                F.log_softmax(pred_logits, dim=-1),
                F.softmax(target_logits, dim=-1),
                reduction="batchmean",
            )

            loss = loss_reg + args.cls_weight * loss_cls

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_reg += loss_reg.item()

        # Validation
        head.eval()
        val_matches = 0
        val_total = 0
        with torch.no_grad():
            for prev_f, adv_t, next_f, next_t in val_loader:
                prev_f = prev_f.to(device)
                adv_t = adv_t.to(device)
                next_f = next_f.to(device)
                next_t = next_t.to(device)

                tok_emb = F.embedding(adv_t, embedding_weight)
                pred_f = head(tok_emb, prev_f, noise_bound=0.0)
                pred_logits = F.linear(pred_f, lm_head_weight)
                preds = pred_logits.argmax(dim=-1)

                val_matches += (preds == next_t).sum().item()
                val_total += len(next_t)

        val_acc = (val_matches / max(1, val_total)) * 100.0
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {total_loss / len(train_loader):.4f} | Reg: {total_reg / len(train_loader):.4f} | Val Draft Match: {val_acc:.1f}%")

    print(f"Training completed in {time.time() - start_time:.2f}s")
    save_kriya_binary_weights(head, hidden_size, vocab_size, args.output_path)


if __name__ == "__main__":
    main()
