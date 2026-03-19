import torch
import wandb                                                          # CHANGED: added wandb import
import json                                                           # CHANGED: added json import
from datasets import Dataset                                          # CHANGED: added Dataset import
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

# ──────────────────────────────────────────────────────────────────────────────
# CHANGED: Wandb sweep configuration (new section)
# ──────────────────────────────────────────────────────────────────────────────

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "important_token_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "beta": {
            "values": [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
        },
    },
}

sweep_id = wandb.sweep(sweep_config, project="qwen3-4b-beta-sweep")

# ──────────────────────────────────────────────────────────────────────────────
# CHANGED: Wrapped everything in a train() function for the sweep agent
# ──────────────────────────────────────────────────────────────────────────────

def train():
    wandb.init()                                                      # CHANGED: init wandb run
    BETA = wandb.config.beta                                          # CHANGED: pull beta from sweep instead of hardcoding 3.0

    max_length = 2048

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B",
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Add marker tokens
    special_tokens = {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    ANSWER_START_ID = tokenizer.convert_tokens_to_ids("<|answer_start|>")
    ANSWER_END_ID = tokenizer.convert_tokens_to_ids("<|answer_end|>")

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    # ── Load dataset ──                                              # CHANGED: added dataset loading
    with open("SFT_training_data.json") as f:
        data = json.load(f)
    dataset = Dataset.from_dict(data)

    # ────────────────────────────────────────────────────────────────────────────
    # Weighted Cross-Entropy Loss for Marker-Bounded Regions
    # ────────────────────────────────────────────────────────────────────────────

    important_token_acc = []
    step_count = 0                                                    # CHANGED: added step counter for wandb logging

    def compute_weighted_loss_func(outputs, labels, num_items_in_batch=None):
        nonlocal step_count                                           # CHANGED: access step counter
        logits = outputs.logits

        # Causal LM shift: logits[i] predicts labels[i+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token loss, flattened for CrossEntropyLoss then reshaped back
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        loss_per_token = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        # Identify tokens inside answer regions using cumulative marker counts
        starts = (shift_labels == ANSWER_START_ID).int()
        ends = (shift_labels == ANSWER_END_ID).int()
        inside = (starts.cumsum(dim=1) - ends.cumsum(dim=1)) > 0
        is_marker = (shift_labels == ANSWER_START_ID) | (shift_labels == ANSWER_END_ID)
        inside = inside & ~is_marker

        # Build weights: β inside answer regions, 1.0 elsewhere
        weights = torch.ones_like(loss_per_token)
        weights[inside] = BETA                                        # CHANGED: uses sweep value instead of hardcoded 3.0

        # Weighted average over non-masked tokens
        mask = (shift_labels != -100).float()
        loss = (loss_per_token * weights * mask).sum() / mask.sum()

        # Record loss and accuracy on tokens of interest
        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels) & inside
            important_token_acc.append(correct.sum().item() / max(inside.sum().item(), 1))

            step_count += 1
            # CHANGED: log metrics to wandb
            wandb.log({
                "important_token_accuracy": important_token_acc[-1],
                "total_loss": loss.item(),
                "beta": BETA,
                "step": step_count,
            })

        return loss

    # CHANGED: added training args with wandb reporting
    training_args = SFTConfig(
        output_dir=f"output/beta-{BETA}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=5,
        save_steps=50,
        warmup_ratio=0.05,
        report_to="wandb",
        disable_tqdm=False,                                               # CHANGED: ensure progress bar is shown
        logging_strategy="steps",                                         # CHANGED: log at regular intervals
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_loss_func=compute_weighted_loss_func,
    )
    trainer.train()
    wandb.finish()                                                    # CHANGED: close wandb run


# ──────────────────────────────────────────────────────────────────────────────
# CHANGED: Launch sweep agent (new section)
# ──────────────────────────────────────────────────────────────────────────────

wandb.agent(sweep_id, function=train, count=6)