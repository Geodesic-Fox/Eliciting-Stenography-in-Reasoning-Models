import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

max_length = 2048 # Supports automatic RoPE Scaling, so choose any number

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B",
    max_seq_length=max_length,
    dtype=None,  # For auto-detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
)

#Add marker tokens
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
    lora_dropout=0,  # Dropout = 0 is currently optimized
    bias="none",  # Bias = "none" is currently optimized
    use_gradient_checkpointing=True,
    random_state=3407,
)

# ──────────────────────────────────────────────────────────────────────────────
# Weighted Cross-Entropy Loss for Marker-Bounded Regions
# ──────────────────────────────────────────────────────────────────────────────
#
# Standard causal LM cross-entropy loss with selective emphasis on tokens that
# fall between designated marker tokens (<|answer_start|> and <|answer_end|>).
#
# Tokens outside the marked region receive a weight of 1.0 (standard loss).
# Tokens inside the marked region receive a weight of β (boosted loss).

important_token_loss = []

def compute_weighted_loss_func(outputs, labels, num_items_in_batch=None):
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
    beta = 3.0
    starts = (shift_labels == ANSWER_START_ID).int()
    ends = (shift_labels == ANSWER_END_ID).int()
    inside = (starts.cumsum(dim=1) - ends.cumsum(dim=1)) > 0
    is_marker = (shift_labels == ANSWER_START_ID) | (shift_labels == ANSWER_END_ID)
    inside = inside & ~is_marker

    # Build weights: β inside answer regions, 1.0 elsewhere
    weights = torch.ones_like(loss_per_token)
    weights[inside] = beta

    # Weighted average over non-masked tokens
    mask = (shift_labels != -100).float()
    loss = (loss_per_token * weights * mask).sum() / mask.sum()

    # Record loss on tokens of interest
    important_token_loss.append((inside * loss_per_token).sum().item() / max(inside.sum().item(), 1))
    
    return loss

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_loss_func=compute_weighted_loss_func
)
trainer.train()