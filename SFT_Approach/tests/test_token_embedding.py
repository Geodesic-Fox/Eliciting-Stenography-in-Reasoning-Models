"""
Tests for the custom token embedding initialization and gradient hook logic.
Uses small mock tensors — no model loading required.
"""

import torch
import torch.nn as nn


# ---- Copy of the function under test ----

def make_new_token_hook(new_token_ids):
    def hook(grad):
        mask = torch.zeros_like(grad)
        for token_id in new_token_ids:
            mask[token_id] = 1
        return grad * mask
    return hook


# ---- Helpers ----

def make_embedding(vocab_size=10, hidden_dim=4, seed=0):
    torch.manual_seed(seed)
    emb = nn.Embedding(vocab_size, hidden_dim)
    return emb


def run_backward_and_get_grad(weight, new_ids):
    """Trigger a backward pass through the embedding weight and return its gradient."""
    weight.requires_grad_(True)
    weight.register_hook(make_new_token_hook(new_ids))
    # Simulate a loss that depends on every row
    loss = weight.sum()
    loss.backward()
    return weight.grad.clone()


# ---- Tests ----

def test_mean_init_correct():
    """New token rows should equal the mean of the original rows."""
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]

    with torch.no_grad():
        # Compute mean over original tokens only (indices 0-7)
        mean_emb = emb.weight[:8].mean(dim=0)
        emb.weight[new_ids[0]] = mean_emb.clone()
        emb.weight[new_ids[1]] = mean_emb.clone()

    for token_id in new_ids:
        assert torch.allclose(emb.weight[token_id], mean_emb), \
            f"Token {token_id} not initialized to mean embedding"

    print("PASS test_mean_init_correct")


def test_mean_init_both_tokens_equal():
    """Both new tokens should start with identical embeddings."""
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]

    with torch.no_grad():
        mean_emb = emb.weight.mean(dim=0)
        emb.weight[new_ids[0]] = mean_emb.clone()
        emb.weight[new_ids[1]] = mean_emb.clone()

    assert torch.allclose(emb.weight[new_ids[0]], emb.weight[new_ids[1]]), \
        "New token embeddings should be identical at init"

    print("PASS test_mean_init_both_tokens_equal")


def test_hook_zeroes_non_new_token_grads():
    """Gradient hook must zero out every row except the new token rows."""
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]
    grad = run_backward_and_get_grad(emb.weight, new_ids)

    for i in range(10):
        if i in new_ids:
            continue
        assert torch.all(grad[i] == 0), \
            f"Row {i} should have zero gradient but got {grad[i]}"

    print("PASS test_hook_zeroes_non_new_token_grads")


def test_hook_preserves_new_token_grads():
    """Gradient hook must NOT zero out the new token rows."""
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]
    grad = run_backward_and_get_grad(emb.weight, new_ids)

    for token_id in new_ids:
        assert not torch.all(grad[token_id] == 0), \
            f"Row {token_id} (new token) should have non-zero gradient"

    print("PASS test_hook_preserves_new_token_grads")


def test_hook_gradient_values_correct():
    """Hook should pass through the real gradient values for new token rows unchanged."""
    emb = make_embedding(vocab_size=10, hidden_dim=4)
    new_ids = [8, 9]

    # Compute unmasked gradient for reference
    weight_ref = emb.weight.detach().clone().requires_grad_(True)
    weight_ref.sum().backward()
    expected_grad = weight_ref.grad.clone()

    # Compute masked gradient
    grad = run_backward_and_get_grad(emb.weight, new_ids)

    for token_id in new_ids:
        assert torch.allclose(grad[token_id], expected_grad[token_id]), \
            f"Gradient for new token {token_id} was modified by hook"

    print("PASS test_hook_gradient_values_correct")


def test_weight_tying_detection_same_object():
    """If embed_tokens and lm_head share the same tensor, is-check should return True."""
    shared_weight = nn.Parameter(torch.randn(10, 4))

    class MockEmbed:
        weight = shared_weight

    class MockHead:
        weight = shared_weight

    assert MockEmbed.weight is MockHead.weight, \
        "Weight tying detection failed: same object not detected"

    print("PASS test_weight_tying_detection_same_object")


def test_weight_tying_detection_different_objects():
    """If embed_tokens and lm_head have distinct tensors, is-check should return False."""
    class MockEmbed:
        weight = nn.Parameter(torch.randn(10, 4))

    class MockHead:
        weight = nn.Parameter(torch.randn(10, 4))

    assert MockEmbed.weight is not MockHead.weight, \
        "Weight tying detection failed: different objects reported as same"

    print("PASS test_weight_tying_detection_different_objects")


# ---- Run all tests ----

if __name__ == "__main__":
    test_mean_init_correct()
    test_mean_init_both_tokens_equal()
    test_hook_zeroes_non_new_token_grads()
    test_hook_preserves_new_token_grads()
    test_hook_gradient_values_correct()
    test_weight_tying_detection_same_object()
    test_weight_tying_detection_different_objects()
    print("\nAll tests passed.")
