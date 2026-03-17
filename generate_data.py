# 
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

#-----------------------------load the tokenizer and the model-----------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


#-----------------------------generate the test questions-----------------------------

random.seed(42)
number_of_questions = 1
multiplication_test_problems = []

for num in range(number_of_questions):
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    multiplication_test_problems.append((a,b,a*b))

#-----------------------------testing loop-----------------------------

results = []

for i, problem in enumerate(multiplication_test_problems):

    prompt = "What is "+str(problem[0])+"*"+str(problem[1])+"? Reply with ONLY the number, nothing else."

    # ── Run with thinking ──
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=4096)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    answer_thinking = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # ── Run without thinking ──
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    answer_no_thinking = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    # ── Save result ──
    results.append({
        "problem": f"{problem[0]}*{problem[1]}",
        "correct_answer": problem[2],
        "thinking_mode": {
            "answer": answer_thinking,
            "thinking": thinking_content,
        },
        "no_thinking_mode": {
            "answer": answer_no_thinking,
        },
    })

    print(f"[{i+1}/{number_of_questions}] {problem[0]}*{problem[1]} = {problem[2]}")
    print(f"  Thinking:    {answer_thinking}")
    print(f"  No thinking: {answer_no_thinking}")

#-----------------------------save to JSON-----------------------------

with open("/workspace/Eliciting-Stenography-in-Reasoning-Models/multiplication_test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to multiplication_test_results.json")