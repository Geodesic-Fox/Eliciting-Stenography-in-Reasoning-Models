"""
Comprehensive tests for steganography_SFT_setup.py.

Categories:
  1. Training data file  — structure and content of steganography_training_data.json
  2. Conversation formatting — generate_conversations() logic
  3. Tokenizer integration — special token addition, chat template output
                             (loads tokenizer only, NOT the full model)
  4. Embedding initialization — mean-init correctness (mock tensors)
  5. Gradient hook — hook masks correctly (mock tensors)

Run with:
    python test_steganography_setup.py
"""

import json
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"

def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return passed


# ===========================================================================
# 1. Training data file
# ===========================================================================

def test_data_file_loads():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    return report("data file loads as valid JSON", True)

def test_data_has_required_keys():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    ok = "prompt" in data and "response" in data
    return report("data has 'prompt' and 'response' keys", ok)

def test_data_lengths_match():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    ok = len(data["prompt"]) == len(data["response"])
    return report("prompt and response have equal length",
                  ok, f"{len(data['prompt'])} examples")

def test_data_nonempty():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    ok = len(data["prompt"]) > 0
    return report("dataset is non-empty", ok, f"{len(data['prompt'])} examples")

def test_prompt_entries_are_lists():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    bad = [i for i, p in enumerate(data["prompt"]) if not isinstance(p, list)]
    ok = len(bad) == 0
    return report("every prompt entry is a list", ok,
                  f"bad indices: {bad[:5]}" if bad else "")

def test_response_entries_are_lists():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    bad = [i for i, r in enumerate(data["response"]) if not isinstance(r, list)]
    ok = len(bad) == 0
    return report("every response entry is a list", ok,
                  f"bad indices: {bad[:5]}" if bad else "")

def test_prompt_entries_have_content_field():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    bad = [i for i, p in enumerate(data["prompt"])
           if not isinstance(p, list) or len(p) == 0 or "content" not in p[0]]
    ok = len(bad) == 0
    return report("every prompt[0] has a 'content' field", ok,
                  f"bad indices: {bad[:5]}" if bad else "")

def test_response_entries_have_content_field():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    bad = [i for i, r in enumerate(data["response"])
           if not isinstance(r, list) or len(r) == 0 or "content" not in r[0]]
    ok = len(bad) == 0
    return report("every response[0] has a 'content' field", ok,
                  f"bad indices: {bad[:5]}" if bad else "")

def test_prompt_content_is_nonempty_string():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    bad = [i for i, p in enumerate(data["prompt"])
           if not isinstance(p[0].get("content", ""), str)
           or len(p[0]["content"].strip()) == 0]
    ok = len(bad) == 0
    return report("every prompt content is a non-empty string", ok,
                  f"bad indices: {bad[:5]}" if bad else "")

def test_response_content_is_nonempty_string():
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    bad = [i for i, r in enumerate(data["response"])
           if not isinstance(r[0].get("content", ""), str)
           or len(r[0]["content"].strip()) == 0]
    ok = len(bad) == 0
    return report("every response content is a non-empty string", ok,
                  f"bad indices: {bad[:5]}" if bad else "")

def test_response_contains_answer_tokens():
    """Every response should contain the steganography answer markers."""
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    bad = [i for i, r in enumerate(data["response"])
           if "<|answer_start|>" not in r[0]["content"]
           or "<|answer_end|>" not in r[0]["content"]]
    ok = len(bad) == 0
    return report("every response contains <|answer_start|> and <|answer_end|>", ok,
                  f"{len(bad)} missing" if bad else "")


# ===========================================================================
# 2. Conversation formatting  (generate_conversations logic)
# ===========================================================================

# Replicate the function exactly as it appears in the script
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


MOCK_EXAMPLES = {
    "prompt": [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Name a planet."}],
    ],
    "response": [
        [{"role": "assistant", "content": "4"}],
        [{"role": "assistant", "content": "Mars"}],
    ],
}

def test_conv_output_has_conversations_key():
    out = generate_conversations(MOCK_EXAMPLES)
    ok = "conversations" in out
    return report("generate_conversations returns 'conversations' key", ok)

def test_conv_correct_number_of_examples():
    out = generate_conversations(MOCK_EXAMPLES)
    ok = len(out["conversations"]) == 2
    return report("generate_conversations returns correct number of examples",
                  ok, f"got {len(out['conversations'])}")

def test_conv_each_has_two_turns():
    out = generate_conversations(MOCK_EXAMPLES)
    ok = all(len(c) == 2 for c in out["conversations"])
    return report("each conversation has exactly 2 turns", ok)

def test_conv_first_turn_is_user():
    out = generate_conversations(MOCK_EXAMPLES)
    ok = all(c[0]["role"] == "user" for c in out["conversations"])
    return report("first turn role is 'user'", ok)

def test_conv_second_turn_is_assistant():
    out = generate_conversations(MOCK_EXAMPLES)
    ok = all(c[1]["role"] == "assistant" for c in out["conversations"])
    return report("second turn role is 'assistant'", ok)

def test_conv_user_content_extracted_correctly():
    out = generate_conversations(MOCK_EXAMPLES)
    ok = (out["conversations"][0][0]["content"] == "What is 2+2?" and
          out["conversations"][1][0]["content"] == "Name a planet.")
    return report("user content extracted correctly from nested structure", ok)

def test_conv_assistant_content_extracted_correctly():
    out = generate_conversations(MOCK_EXAMPLES)
    ok = (out["conversations"][0][1]["content"] == "4" and
          out["conversations"][1][1]["content"] == "Mars")
    return report("assistant content extracted correctly from nested structure", ok)

def test_conv_with_real_data():
    """Run generate_conversations on the actual training file."""
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    out = generate_conversations(data)
    ok = len(out["conversations"]) == len(data["prompt"])
    return report("generate_conversations processes full dataset without error",
                  ok, f"{len(out['conversations'])} conversations")


# ===========================================================================
# 3. Tokenizer integration  (tokenizer only — no model weights loaded)
# ===========================================================================

def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("unsloth/Qwen3-4B")

def test_tokenizer_loads():
    tok = _load_tokenizer()
    ok = tok is not None
    return report("tokenizer loads from cache", ok)

def test_special_tokens_added():
    tok = _load_tokenizer()
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
    )
    vocab = tok.get_vocab()
    ok = "<|answer_start|>" in vocab and "<|answer_end|>" in vocab
    return report("special tokens added to tokenizer vocabulary", ok)

def test_special_token_ids_are_unique():
    tok = _load_tokenizer()
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
    )
    start_id = tok.convert_tokens_to_ids("<|answer_start|>")
    end_id   = tok.convert_tokens_to_ids("<|answer_end|>")
    ok = start_id != end_id
    return report("answer_start and answer_end have distinct token IDs", ok,
                  f"start={start_id}, end={end_id}")

def test_special_tokens_not_split():
    """Tokenizer should encode each special token as exactly one token ID."""
    tok = _load_tokenizer()
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
    )
    for token in ["<|answer_start|>", "<|answer_end|>"]:
        ids = tok.encode(token, add_special_tokens=False)
        ok = len(ids) == 1
        if not report(f"'{token}' encodes to exactly 1 token (not split)", ok,
                      f"got {len(ids)} tokens"):
            return False
    return True

def test_chat_template_produces_strings():
    tok = _load_tokenizer()
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
    )
    conversations = [
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    ]
    result = tok.apply_chat_template(conversations, tokenize=False)
    ok = isinstance(result, list) and all(isinstance(s, str) for s in result)
    return report("apply_chat_template returns a list of strings", ok)

def test_chat_template_nonempty():
    tok = _load_tokenizer()
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
    )
    conversations = [
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    ]
    result = tok.apply_chat_template(conversations, tokenize=False)
    ok = all(len(s) > 0 for s in result)
    return report("chat template output strings are non-empty", ok)

def test_chat_template_contains_content():
    """The formatted string should contain the actual user/assistant text."""
    tok = _load_tokenizer()
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
    )
    conversations = [
        [{"role": "user", "content": "unique_user_text_xyz"},
         {"role": "assistant", "content": "unique_assistant_text_xyz"}]
    ]
    result = tok.apply_chat_template(conversations, tokenize=False)
    ok = "unique_user_text_xyz" in result[0] and "unique_assistant_text_xyz" in result[0]
    return report("chat template output contains original user/assistant text", ok)

def test_chat_template_count_matches_dataset():
    """One formatted string should be produced per training example."""
    tok = _load_tokenizer()
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|answer_start|>", "<|answer_end|>"]}
    )
    with open("steganography_training_data.json", "r") as f:
        data = json.load(f)
    out = generate_conversations(data)
    results = tok.apply_chat_template(out["conversations"], tokenize=False)
    ok = len(results) == len(data["prompt"])
    return report("apply_chat_template produces one string per example",
                  ok, f"expected {len(data['prompt'])}, got {len(results)}")


# ===========================================================================
# 4. Embedding initialization (mock tensors)
# ===========================================================================

def make_embedding(vocab_size=10, hidden_dim=4, seed=0):
    torch.manual_seed(seed)
    return nn.Embedding(vocab_size, hidden_dim)

def test_mean_init_correct():
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]
    with torch.no_grad():
        mean_emb = emb.weight.mean(dim=0)
        emb.weight[new_ids[0]] = mean_emb.clone()
        emb.weight[new_ids[1]] = mean_emb.clone()
    expected = emb.weight.mean(dim=0)  # recompute after assignment
    # Verify the assigned rows match the pre-assignment mean
    with torch.no_grad():
        original_mean = make_embedding(vocab_size=10, hidden_dim=4).weight.mean(dim=0)
    for tid in new_ids:
        ok = torch.allclose(emb.weight[tid], original_mean)
        if not report(f"token {tid} initialized to mean embedding", ok):
            return False
    return True

def test_mean_init_both_tokens_equal():
    emb = make_embedding()
    new_ids = [8, 9]
    with torch.no_grad():
        mean_emb = emb.weight.mean(dim=0)
        emb.weight[new_ids[0]] = mean_emb.clone()
        emb.weight[new_ids[1]] = mean_emb.clone()
    ok = torch.allclose(emb.weight[new_ids[0]], emb.weight[new_ids[1]])
    return report("both new token embeddings are identical at init", ok)

def test_mean_init_does_not_change_other_rows():
    emb_orig = make_embedding()
    emb_new  = make_embedding()  # same seed → same initial weights
    new_ids = [8, 9]
    with torch.no_grad():
        mean_emb = emb_new.weight.mean(dim=0)
        emb_new.weight[new_ids[0]] = mean_emb.clone()
        emb_new.weight[new_ids[1]] = mean_emb.clone()
    for i in range(8):  # rows 0-7 should be untouched
        ok = torch.allclose(emb_orig.weight[i], emb_new.weight[i])
        if not report(f"row {i} unchanged after mean init", ok):
            return False
    return True


# ===========================================================================
# 5. Gradient hook (mock tensors)
# ===========================================================================

def make_new_token_hook(new_token_ids):
    """Exact copy of the function in steganography_SFT_setup.py."""
    def hook(grad):
        mask = torch.zeros_like(grad)
        for token_id in new_token_ids:
            mask[token_id] = 1
        return grad * mask
    return hook

def run_backward(weight, new_ids):
    weight.requires_grad_(True)
    weight.register_hook(make_new_token_hook(new_ids))
    weight.sum().backward()
    return weight.grad.clone()

def test_hook_zeroes_non_new_token_rows():
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]
    grad = run_backward(emb.weight, new_ids)
    bad = [i for i in range(10) if i not in new_ids and not torch.all(grad[i] == 0)]
    ok = len(bad) == 0
    return report("hook zeros gradients for all non-new-token rows", ok,
                  f"non-zero rows: {bad}" if bad else "")

def test_hook_preserves_new_token_gradients():
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]
    grad = run_backward(emb.weight, new_ids)
    ok = all(not torch.all(grad[tid] == 0) for tid in new_ids)
    return report("hook preserves non-zero gradients for new token rows", ok)

def test_hook_gradient_values_unchanged_for_new_tokens():
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]

    # Reference gradient without hook
    ref = make_embedding(vocab_size=10, hidden_dim=4)
    ref.weight.requires_grad_(True)
    ref.weight.sum().backward()
    ref_grad = ref.weight.grad.clone()

    # Masked gradient
    grad = run_backward(emb.weight, new_ids)

    ok = all(torch.allclose(grad[tid], ref_grad[tid]) for tid in new_ids)
    return report("hook passes through correct gradient values for new tokens", ok)

def test_hook_with_single_new_token():
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [5]
    grad = run_backward(emb.weight, new_ids)
    non_new_zero = all(torch.all(grad[i] == 0) for i in range(10) if i != 5)
    new_nonzero  = not torch.all(grad[5] == 0)
    ok = non_new_zero and new_nonzero
    return report("hook works correctly with a single new token", ok)

def test_weight_tying_same_object():
    shared = nn.Parameter(torch.randn(10, 4))
    class E: weight = shared
    class H: weight = shared
    ok = E.weight is H.weight
    return report("weight tying detected: same object → 'is' check is True", ok)

def test_weight_tying_different_objects():
    class E: weight = nn.Parameter(torch.randn(10, 4))
    class H: weight = nn.Parameter(torch.randn(10, 4))
    ok = E.weight is not H.weight
    return report("no weight tying: different objects → 'is not' check is True", ok)


# ===========================================================================
# Runner
# ===========================================================================

def run_section(title, tests):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    results = [t() for t in tests]
    passed = sum(results)
    print(f"  {passed}/{len(results)} passed")
    return passed, len(results)


if __name__ == "__main__":
    sections = [
        ("1. Training data file", [
            test_data_file_loads,
            test_data_has_required_keys,
            test_data_lengths_match,
            test_data_nonempty,
            test_prompt_entries_are_lists,
            test_response_entries_are_lists,
            test_prompt_entries_have_content_field,
            test_response_entries_have_content_field,
            test_prompt_content_is_nonempty_string,
            test_response_content_is_nonempty_string,
            test_response_contains_answer_tokens,
        ]),
        ("2. Conversation formatting", [
            test_conv_output_has_conversations_key,
            test_conv_correct_number_of_examples,
            test_conv_each_has_two_turns,
            test_conv_first_turn_is_user,
            test_conv_second_turn_is_assistant,
            test_conv_user_content_extracted_correctly,
            test_conv_assistant_content_extracted_correctly,
            test_conv_with_real_data,
        ]),
        ("3. Tokenizer integration", [
            test_tokenizer_loads,
            test_special_tokens_added,
            test_special_token_ids_are_unique,
            test_special_tokens_not_split,
            test_chat_template_produces_strings,
            test_chat_template_nonempty,
            test_chat_template_contains_content,
            test_chat_template_count_matches_dataset,
        ]),
        ("4. Embedding initialization", [
            test_mean_init_correct,
            test_mean_init_both_tokens_equal,
            test_mean_init_does_not_change_other_rows,
        ]),
        ("5. Gradient hook", [
            test_hook_zeroes_non_new_token_rows,
            test_hook_preserves_new_token_gradients,
            test_hook_gradient_values_unchanged_for_new_tokens,
            test_hook_with_single_new_token,
            test_weight_tying_same_object,
            test_weight_tying_different_objects,
        ]),
    ]

    total_passed = total_tests = 0
    for title, tests in sections:
        p, t = run_section(title, tests)
        total_passed += p
        total_tests  += t

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_passed}/{total_tests} passed")
    print(f"{'='*60}\n")
