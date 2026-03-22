# Required packages:
# pip install unsloth
# pip install transformers==4.56.2
# pip install --no-deps trl==0.22.2
# pip install torch
# pip install datasets

from unsloth import FastLanguageModel
import torch


# ---------- Load base model ----------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B",
    max_seq_length = 2048,   # Context length - can be longer, but uses more memory
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    # token = "YOUR_HF_TOKEN",      # HF Token for gated models
)

# -------Add Custom Tokens -----------

special_tokens = {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

ANSWER_START_ID = tokenizer.convert_tokens_to_ids("<|answer_start|>")
ANSWER_END_ID = tokenizer.convert_tokens_to_ids("<|answer_end|>")

# Initialize new tokens with mean of all existing embeddings (unbiased)
with torch.no_grad():
    mean_embedding = model.model.embed_tokens.weight.mean(dim=0)
    model.model.embed_tokens.weight[ANSWER_START_ID] = mean_embedding.clone()
    model.model.embed_tokens.weight[ANSWER_END_ID] = mean_embedding.clone()


# ---------- Attach LoRA adapters ----------
# Only the adapter weights are trained; base model weights are frozen.

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # LoRA rank — higher = more capacity, more memory. Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Scaling factor; best to set equal to rank or rank*2
    lora_dropout = 0, # Supports any value, but 0 is optimized
    bias = "none",    # Supports any value, but "none" is optimized
    use_gradient_checkpointing = "unsloth", # "unsloth" uses 30% less VRAM; fits 2x larger batch sizes
    random_state = 3407,
    use_rslora = False,   # Rank-stabilized LoRA (alternative to standard LoRA)
    loftq_config = None,  # LoftQ quantization-aware LoRA init (alternative init strategy)
)

# Enable gradients for embedding/unembedding matrices and register hooks so that
# only the two new token rows are updated during training (avoids touching the
# 150k+ existing token rows, which are already well-trained).
# Must be done AFTER get_peft_model, which would otherwise freeze these weights.
# Use get_input/output_embeddings() so the path works regardless of PEFT wrapping.
embed_layer = model.get_input_embeddings()
lm_head     = model.get_output_embeddings()

embed_layer.weight.requires_grad_(True)

def make_new_token_hook(new_token_ids):
    def hook(grad):
        mask = torch.zeros_like(grad)
        for token_id in new_token_ids:
            mask[token_id] = 1
        return grad * mask
    return hook

new_ids = [ANSWER_START_ID, ANSWER_END_ID]
embed_layer.weight.register_hook(make_new_token_hook(new_ids))

# lm_head may share weights with embed_tokens (weight tying); only register
# a separate hook if they are distinct matrices.
if lm_head.weight is not embed_layer.weight:
    lm_head.weight.requires_grad_(True)
    lm_head.weight.register_hook(make_new_token_hook(new_ids))

# ---------- Build training dataset ----------
# Loads stenography prompt/response pairs from JSON, converts them into
# chat-formatted strings using the tokenizer's chat template.

from datasets import Dataset
import json

with open("stenography_training_data.json", "r") as f:
    raw = json.load(f)

reasoning_dataset = Dataset.from_dict(raw)

# Reformat raw prompt/response columns into role-tagged conversation lists
def generate_conversations(examples):
    conversations = []
    for prompt, response in zip(examples["prompt"], examples["response"]):
        user_content = prompt[0]["content"]
        assistant_content = response[0]["content"]
        conversations.append([
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ])
    return {"conversations": conversations}

reasoning_dataset = reasoning_dataset.map(generate_conversations, batched=True)

# Apply the tokenizer's chat template to produce fully-formatted training strings
reasoning_conversations = tokenizer.apply_chat_template(
    list(reasoning_dataset["conversations"]),
    tokenize=False,
)

reasoning_conversation_dataset = Dataset.from_dict({"text": reasoning_conversations})

# ---------- Train ----------

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = reasoning_conversation_dataset,
    eval_dataset = None, # Can set up evaluation!
    # compute_loss_func=compute_weighted_loss_func,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Effective batch size = per_device * grad_accum = 8
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run
        max_steps = 30,        # Quick experiment; increase or switch to num_train_epochs for full run
        learning_rate = 2e-4,  # Reduce to 2e-5 for longer training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",    # Set to "wandb" or "tensorboard" to enable logging
        padding_free  = False, # Set to True if > 17 GB VRAM available
    ),
)

# Snapshot embeddings before training to verify they update
emb_w = model.get_input_embeddings().weight
embed_before = {
    "answer_start": emb_w[ANSWER_START_ID].detach().clone(),
    "answer_end":   emb_w[ANSWER_END_ID].detach().clone(),
    "token_0":      emb_w[0].detach().clone(),  # control: should NOT change
}

trainer_stats = trainer.train()

# Verify new token embeddings changed and existing ones did not
emb_w = model.get_input_embeddings().weight
embed_after = {
    "answer_start": emb_w[ANSWER_START_ID].detach().clone(),
    "answer_end":   emb_w[ANSWER_END_ID].detach().clone(),
    "token_0":      emb_w[0].detach().clone(),
}

print("\n--- Embedding update check ---")
for name in ["answer_start", "answer_end", "token_0"]:
    changed = not torch.allclose(embed_before[name], embed_after[name])
    delta = (embed_after[name] - embed_before[name]).abs().max().item()
    print(f"  {name:15s}: {'CHANGED' if changed else 'unchanged'}  (max delta: {delta:.6f})")
