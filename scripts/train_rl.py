"""RL fine-tuning for sarcasm style transfer using REINFORCE + KL penalty.

Takes an SFT-trained BART (or T5) checkpoint and refines it using a sarcasm
classifier as the reward signal. The classifier scores how well the generated
output achieves the target style (non-sarcastic for de-sarcasm).

Pipeline:
    1. Load SFT checkpoint (BART/T5) as both policy and frozen reference
    2. For each batch: generate outputs via sampling
    3. Score outputs with sarcasm classifier → reward
    4. Compute REINFORCE loss + KL divergence penalty
    5. Update policy; reference stays frozen

Usage:
    python scripts/train_rl.py \
        --sft_checkpoint outputs/bart-base/sar-to-non/final \
        --classifier_model SeeYangZhi/sarcasm-classifier \
        --direction sar-to-non \
        --epochs 3 --kl_coeff 0.1

References:
    - ViSP (arxiv 2507.09482): PPO + sarcasm classifier reward for sarcasm generation
    - Williams (1992): REINFORCE algorithm
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_data_paths(direction: str) -> dict[str, Path]:
    splits_dir = PROJECT_ROOT / "data" / "splits"
    if direction == "sar-to-non":
        base = splits_dir / "sar_to_non"
    else:
        base = splits_dir
    return {
        "train": base / "train.jsonl",
        "val": base / "val.jsonl",
        "test": base / "test.jsonl",
    }


def prepare_examples(records: list[dict], direction: str, model_name: str) -> list[dict]:
    """Map raw JSONL records to input_text / target_text pairs."""
    examples = []
    for r in records:
        if direction == "sar-to-non":
            source = r["original_headline"]
            target = r["generated_headline"]
            if "t5" in model_name.lower() or "flan" in model_name.lower():
                input_text = f"desarcasm: {source}"
            else:
                input_text = source
        else:
            strategy = r["strategy"]
            source = r.get("non_sarcastic_source", r.get("original_headline", ""))
            target = r["generated_headline"]
            input_text = f"<{strategy}> {source}"

        examples.append({"input_text": input_text, "target_text": target})
    return examples


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------
class RewardModel:
    """Wraps a sarcasm classifier to produce reward scores.

    For de-sarcasm: reward = 1 - P(sarcastic)
    High reward means the output is classified as non-sarcastic.
    """

    def __init__(self, model_name: str, device: str, direction: str = "sar-to-non"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.direction = direction

        # Detect which label index is "sarcastic"
        # Common label mappings: {0: "not_sarcastic", 1: "sarcastic"} or vice versa
        id2label = self.model.config.id2label
        self.sarcastic_idx = None
        for idx, label in id2label.items():
            if "sarc" in label.lower() or label == "1":
                self.sarcastic_idx = int(idx)
                break
        if self.sarcastic_idx is None:
            # Default: assume label 1 = sarcastic
            self.sarcastic_idx = 1
        print(f"Reward model: sarcastic label index = {self.sarcastic_idx}")
        print(f"Reward model labels: {id2label}")

    @torch.no_grad()
    def score(self, texts: list[str]) -> torch.Tensor:
        """Return reward scores for a batch of generated texts.

        Returns tensor of shape (batch_size,) with values in [0, 1].
        """
        encoded = self.tokenizer(
            texts,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoded).logits
        probs = F.softmax(logits, dim=-1)
        p_sarcastic = probs[:, self.sarcastic_idx]

        if self.direction == "sar-to-non":
            # Reward for being non-sarcastic
            reward = 1.0 - p_sarcastic
        else:
            # Reward for being sarcastic
            reward = p_sarcastic

        return reward


# ---------------------------------------------------------------------------
# KL divergence between policy and reference
# ---------------------------------------------------------------------------
def compute_kl_divergence(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
) -> torch.Tensor:
    """Token-level KL(policy || reference), averaged over sequence.

    Both inputs: (batch, seq_len, vocab_size)
    Returns: scalar tensor.
    """
    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
    ref_logprobs = F.log_softmax(ref_logits, dim=-1)

    # KL(P || Q) = sum P * (log P - log Q)
    kl = F.kl_div(ref_logprobs, policy_logprobs, log_target=True, reduction="none")
    # Sum over vocab, mean over sequence and batch
    return kl.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# REINFORCE loss
# ---------------------------------------------------------------------------
def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float | None = None,
) -> torch.Tensor:
    """Policy gradient loss.

    log_probs: (batch,) — sum of log probs of generated tokens
    rewards: (batch,) — reward scores from classifier
    baseline: optional running mean for variance reduction
    """
    advantages = rewards
    if baseline is not None:
        advantages = rewards - baseline

    # Negative because we maximize reward
    return -(advantages.detach() * log_probs).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    policy_model,
    ref_model,
    reward_model: RewardModel,
    policy_tokenizer,
    dataloader: DataLoader,
    optimizer,
    device: str,
    kl_coeff: float,
    max_gen_length: int,
    reward_baseline: float,
    epoch: int,
    log_interval: int = 50,
) -> tuple[float, float, float, float]:
    """One epoch of REINFORCE training.

    Returns: (avg_reward, avg_policy_loss, avg_kl, updated_baseline)
    """
    policy_model.train()
    ref_model.eval()

    total_reward = 0.0
    total_policy_loss = 0.0
    total_kl = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # --- Generate from policy via sampling ---
        with torch.no_grad():
            generated = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
            )

        # Decode generated text for reward scoring
        gen_texts = policy_tokenizer.batch_decode(generated, skip_special_tokens=True)

        # --- Compute reward ---
        rewards = reward_model.score(gen_texts)  # (batch,)

        # --- Compute log probs of generated tokens under policy ---
        # For seq2seq: decoder input is the generated sequence
        # We need to get log probs token-by-token
        decoder_input_ids = generated[:, :-1]  # shift right
        decoder_labels = generated[:, 1:]  # targets

        policy_outputs = policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        policy_logits = policy_outputs.logits  # (batch, seq_len, vocab)

        # Get reference model logits (frozen)
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            ref_logits = ref_outputs.logits

        # Token-level log probs under policy
        log_probs_all = F.log_softmax(policy_logits, dim=-1)
        # Gather log probs of actual generated tokens
        token_log_probs = log_probs_all.gather(
            2, decoder_labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len)

        # Mask padding tokens
        pad_mask = (decoder_labels != policy_tokenizer.pad_token_id).float()
        if hasattr(policy_model.config, "decoder_start_token_id"):
            pad_mask = pad_mask * (decoder_labels != policy_model.config.decoder_start_token_id).float()

        # Sum log probs per sequence
        seq_log_probs = (token_log_probs * pad_mask).sum(dim=-1)  # (batch,)

        # --- Losses ---
        # REINFORCE
        rl_loss = reinforce_loss(seq_log_probs, rewards, baseline=reward_baseline)

        # KL divergence
        # Trim to same length
        min_len = min(policy_logits.size(1), ref_logits.size(1))
        kl = compute_kl_divergence(
            policy_logits[:, :min_len, :],
            ref_logits[:, :min_len, :],
        )

        # Combined loss
        loss = rl_loss + kl_coeff * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update running baseline (exponential moving average)
        batch_reward = rewards.mean().item()
        reward_baseline = 0.9 * reward_baseline + 0.1 * batch_reward

        total_reward += batch_reward
        total_policy_loss += rl_loss.item()
        total_kl += kl.item()
        num_batches += 1

        if (step + 1) % log_interval == 0:
            print(
                f"  Epoch {epoch} Step {step + 1}: "
                f"reward={batch_reward:.4f}  "
                f"rl_loss={rl_loss.item():.4f}  "
                f"kl={kl.item():.4f}  "
                f"total_loss={loss.item():.4f}"
            )

    n = max(num_batches, 1)
    return total_reward / n, total_policy_loss / n, total_kl / n, reward_baseline


@torch.no_grad()
def evaluate(
    policy_model,
    reward_model: RewardModel,
    policy_tokenizer,
    dataloader: DataLoader,
    device: str,
    max_gen_length: int,
) -> tuple[float, list[str], list[str]]:
    """Evaluate: compute average reward and collect sample outputs."""
    policy_model.eval()
    total_reward = 0.0
    num_batches = 0
    all_inputs = []
    all_outputs = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated = policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_length,
            num_beams=4,
        )

        gen_texts = policy_tokenizer.batch_decode(generated, skip_special_tokens=True)
        inp_texts = policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        rewards = reward_model.score(gen_texts)
        total_reward += rewards.mean().item()
        num_batches += 1

        all_inputs.extend(inp_texts)
        all_outputs.extend(gen_texts)

    avg_reward = total_reward / max(num_batches, 1)
    return avg_reward, all_inputs, all_outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RL fine-tuning for sarcasm style transfer")
    p.add_argument("--sft_checkpoint", type=str, required=True,
                    help="Path to SFT-trained model checkpoint")
    p.add_argument("--classifier_model", type=str, required=True,
                    help="HuggingFace model ID for sarcasm classifier (reward model)")
    p.add_argument("--direction", type=str, default="sar-to-non",
                    choices=["sar-to-non", "non-to-sar"])
    p.add_argument("--epochs", type=int, default=3, help="Number of RL training epochs")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size (smaller for RL)")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower than SFT)")
    p.add_argument("--kl_coeff", type=float, default=0.1,
                    help="KL penalty coefficient (higher = more conservative)")
    p.add_argument("--max_length", type=int, default=128, help="Max generation length")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load SFT model (policy) and create frozen reference copy ---
    print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
    policy_tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    policy_model = AutoModelForSeq2SeqLM.from_pretrained(args.sft_checkpoint).to(device)

    # Frozen reference model (deep copy)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Reference model frozen.")

    # --- Load reward model (sarcasm classifier) ---
    print(f"Loading classifier: {args.classifier_model}")
    reward_model = RewardModel(args.classifier_model, device, args.direction)

    # --- Detect model type for prepare_examples ---
    config_path = Path(args.sft_checkpoint) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    model_type = config.get("_name_or_path", config.get("model_type", ""))

    # --- Load data ---
    data_paths = get_data_paths(args.direction)
    train_records = load_jsonl(data_paths["train"])
    val_records = load_jsonl(data_paths["val"])

    raw_train = prepare_examples(train_records, args.direction, model_type)
    raw_val = prepare_examples(val_records, args.direction, model_type)

    print(f"Train: {len(raw_train)}, Val: {len(raw_val)}")

    # Tokenize inputs only (targets generated by policy during RL)
    def tokenize_inputs(examples):
        return policy_tokenizer(
            examples["input_text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )

    train_ds = Dataset.from_list(raw_train)
    val_ds = Dataset.from_list(raw_val)

    train_ds = train_ds.map(tokenize_inputs, batched=True, remove_columns=["input_text", "target_text"])
    val_ds = val_ds.map(tokenize_inputs, batched=True, remove_columns=["input_text", "target_text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # --- Output directory ---
    if args.output_dir is None:
        model_short = Path(args.sft_checkpoint).parent.parent.name
        args.output_dir = str(PROJECT_ROOT / "outputs" / f"{model_short}-rl" / args.direction)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01)

    # --- Training loop ---
    reward_baseline = 0.5  # Initial baseline for variance reduction
    best_val_reward = -float("inf")

    print(f"\nStarting RL training: {args.epochs} epochs, kl_coeff={args.kl_coeff}, lr={args.lr}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        avg_reward, avg_loss, avg_kl, reward_baseline = train_one_epoch(
            policy_model=policy_model,
            ref_model=ref_model,
            reward_model=reward_model,
            policy_tokenizer=policy_tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            kl_coeff=args.kl_coeff,
            max_gen_length=args.max_length,
            reward_baseline=reward_baseline,
            epoch=epoch,
            log_interval=args.log_interval,
        )

        print(f"\nEpoch {epoch} train: reward={avg_reward:.4f}  rl_loss={avg_loss:.4f}  kl={avg_kl:.4f}")

        # Evaluate
        val_reward, val_inputs, val_outputs = evaluate(
            policy_model=policy_model,
            reward_model=reward_model,
            policy_tokenizer=policy_tokenizer,
            dataloader=val_loader,
            device=device,
            max_gen_length=args.max_length,
        )
        print(f"Epoch {epoch} val:   reward={val_reward:.4f}")

        # Print samples
        print("\n  Sample outputs:")
        for i in range(min(5, len(val_inputs))):
            print(f"    Input:  {val_inputs[i]}")
            print(f"    Output: {val_outputs[i]}")
            print()

        # Save best model
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            save_dir = output_dir / "best"
            policy_model.save_pretrained(str(save_dir))
            policy_tokenizer.save_pretrained(str(save_dir))
            print(f"  New best model saved (reward={val_reward:.4f})")

        print("=" * 70)

    # Save final model
    final_dir = output_dir / "final"
    policy_model.save_pretrained(str(final_dir))
    policy_tokenizer.save_pretrained(str(final_dir))

    # Save training config
    with open(output_dir / "rl_training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nRL training complete. Best val reward: {best_val_reward:.4f}")
    print(f"Best model: {output_dir / 'best'}")
    print(f"Final model: {final_dir}")


if __name__ == "__main__":
    main()
