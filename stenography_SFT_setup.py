# pip install unsloth
# pip install transformers==4.56.2
# pip install --no-deps trl==0.22.2
# pip install torch
# pip install datasets

from unsloth import FastLanguageModel
import torch


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B",
    max_seq_length = 2048,   # Context length - can be longer, but uses more memory
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    # token = "YOUR_HF_TOKEN",      # HF Token for gated models
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

#----------Questionable code

from datasets import Dataset
import json

with open("stenography_training_data.json", "r") as f:
    raw = json.load(f)

reasoning_dataset = Dataset.from_dict(raw)

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


reasoning_conversations = tokenizer.apply_chat_template(
    list(reasoning_dataset["conversations"]),
    tokenize=False,
)

reasoning_conversation_dataset = Dataset.from_dict({"text":reasoning_conversations})

#----------Questionable code 

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = reasoning_conversation_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
        padding_free  = False, # Set to True if > 17 GB VRAM
    ),
)

trainer_stats = trainer.train()

