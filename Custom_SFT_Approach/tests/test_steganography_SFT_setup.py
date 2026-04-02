"""
Unit tests for the core logic in steganography_SFT_setup.py.

Tests exercise components directly with synthetic tensors to avoid loading
the full Qwen3-4B model.
"""

import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

# ---------------------------------------------------------------------------
# Mock heavy dependencies so the module can be imported without GPU / network
# ---------------------------------------------------------------------------
# TRL's import_utils calls importlib.util.find_spec("unsloth"), which requires
# the mock to have a proper __spec__ (not None).
import importlib.machinery as _im
_unsloth_mock = MagicMock()
_unsloth_mock.__spec__ = _im.ModuleSpec("unsloth", None)
sys.modules.setdefault("unsloth", _unsloth_mock)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "steganography_SFT_setup",
    Path(__file__).parent.parent / "experiments" / "steganography_SFT_setup.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Helpers: replicate the key logic from steganography_SFT_setup.py so tests
# don't require importing (and running) the full training script.
# ---------------------------------------------------------------------------

ANSWER_START_ID = 100
ANSWER_END_ID   = 101
EOS_TOKEN_ID    = 2
BETA            = 3.0


def _inside_mask(shift_labels):
    """Return a bool mask for tokens strictly inside answer regions."""
    starts = (shift_labels == ANSWER_START_ID).int()
    ends   = (shift_labels == ANSWER_END_ID).int()
    inside = (starts.cumsum(dim=1) - ends.cumsum(dim=1)) > 0
    is_marker = (
        (shift_labels == ANSWER_START_ID) |
        (shift_labels == ANSWER_END_ID) |
        (shift_labels == EOS_TOKEN_ID)
    )
    return inside & ~is_marker


def _compute_weighted_loss(shift_logits, shift_labels):
    """Weighted cross-entropy matching the implementation in the training script."""
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    loss_per_token = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.shape)

    inside  = _inside_mask(shift_labels)
    weights = torch.ones_like(loss_per_token)
    weights[inside] = BETA

    mask = (shift_labels != -100).float()
    return (loss_per_token * weights * mask).sum() / mask.sum()


def _make_outputs(logits):
    """Wrap a logits tensor in a minimal namespace that mimics model outputs."""
    out = types.SimpleNamespace()
    out.logits = logits
    return out


def _make_weighted_loss_func(answer_start_id, answer_end_id, eos_token_id, beta):
    """Factory: returns a compute_weighted_loss_func closure with given constants."""
    def compute_weighted_loss_func(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        loss_per_token = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        starts = (shift_labels == answer_start_id).int()
        ends   = (shift_labels == answer_end_id).int()
        inside = (starts.cumsum(dim=1) - ends.cumsum(dim=1)) > 0
        is_marker = (
            (shift_labels == answer_start_id) |
            (shift_labels == answer_end_id) |
            (shift_labels == eos_token_id)
        )
        inside = inside & ~is_marker

        weights = torch.ones_like(loss_per_token)
        weights[inside] = beta

        mask = (shift_labels != -100).float()
        return (loss_per_token * weights * mask).sum() / mask.sum()

    return compute_weighted_loss_func


# ---------------------------------------------------------------------------
# 1. Inside-region mask tests
# ---------------------------------------------------------------------------

class TestInsideMask:
    def test_basic_region(self):
        # [START, a, b, END]  ->  inside = [F, T, T, F]
        labels = torch.tensor([[ANSWER_START_ID, 10, 11, ANSWER_END_ID]])
        mask = _inside_mask(labels)
        assert mask.tolist() == [[False, True, True, False]]

    def test_markers_excluded(self):
        # Marker tokens themselves must NOT be inside
        labels = torch.tensor([[ANSWER_START_ID, ANSWER_END_ID]])
        mask = _inside_mask(labels)
        assert not mask.any()

    def test_eos_excluded(self):
        # EOS inside an answer region is excluded
        labels = torch.tensor([[ANSWER_START_ID, EOS_TOKEN_ID, 5, ANSWER_END_ID]])
        mask = _inside_mask(labels)
        expected = [[False, False, True, False]]
        assert mask.tolist() == expected

    def test_no_markers(self):
        # No markers -> nothing is inside
        labels = torch.tensor([[1, 2, 3, 4]])
        assert not _inside_mask(labels).any()

    def test_multiple_regions(self):
        # Two answer regions in one sequence
        labels = torch.tensor([[
            ANSWER_START_ID, 5, ANSWER_END_ID,
            7,
            ANSWER_START_ID, 8, 9, ANSWER_END_ID,
        ]])
        mask = _inside_mask(labels)
        expected = [[False, True, False, False, False, True, True, False]]
        assert mask.tolist() == expected

    def test_tokens_outside_region_not_inside(self):
        labels = torch.tensor([[3, 4, ANSWER_START_ID, 10, ANSWER_END_ID, 6]])
        mask = _inside_mask(labels)
        # Only position 3 (token 10) is inside
        assert mask[0, 3].item() is True
        assert not mask[0, 0].item()
        assert not mask[0, 1].item()
        assert not mask[0, 5].item()

    def test_batch_independence(self):
        # Each row is processed independently
        labels = torch.tensor([
            [ANSWER_START_ID, 5, ANSWER_END_ID, 0],
            [0, 0, 0, 0],
        ])
        mask = _inside_mask(labels)
        assert mask[0, 1].item() is True
        assert not mask[1].any()


# ---------------------------------------------------------------------------
# 2. Weighted loss tests
# ---------------------------------------------------------------------------

class TestWeightedLoss:
    def _uniform_logits(self, batch, seq, vocab=50):
        """Logits that produce uniform predictions (no preferred token)."""
        return torch.zeros(batch, seq, vocab)

    def test_returns_scalar(self):
        logits = self._uniform_logits(1, 5)
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        loss = _compute_weighted_loss(logits[..., :-1, :], labels[..., 1:])
        assert loss.shape == ()

    def test_loss_positive(self):
        logits = self._uniform_logits(1, 5)
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        loss = _compute_weighted_loss(logits[..., :-1, :], labels[..., 1:])
        assert loss.item() > 0

    def test_beta_increases_loss_for_inside_tokens(self):
        """Loss with inside tokens should exceed loss without them (same logits).

        Labels must have a leading non-marker token so that after the causal
        shift (labels[1:]), ANSWER_START_ID is still present in shift_labels
        and the inside-region detection can fire.
        """
        vocab = 110  # must exceed max token ID (ANSWER_END_ID=101)

        # layout: [prefix, START, a, b, END, suffix]
        # After shift labels[1:]: [START, a, b, END, suffix]  -> a,b are inside
        labels_with_region = torch.tensor([[
            5, ANSWER_START_ID, 10, 11, ANSWER_END_ID, 20
        ]])
        # Same shape but no markers -> nothing is inside
        labels_no_region = torch.tensor([[5, 3, 10, 11, 6, 20]])

        seq   = labels_with_region.shape[1]          # 6
        logits = torch.zeros(1, seq, vocab)
        shift_logits = logits[..., :-1, :]            # (1, 5, vocab)

        loss_with    = _compute_weighted_loss(shift_logits, labels_with_region[..., 1:])
        loss_without = _compute_weighted_loss(shift_logits, labels_no_region[..., 1:])

        # Inside tokens are weighted by BETA > 1, so weighted-average loss is higher
        assert loss_with.item() > loss_without.item()

    def test_ignored_tokens_not_counted(self):
        """Tokens with label -100 must be excluded from the loss."""
        vocab = 10
        # Sequence: first token is real, rest are padding (-100)
        logits = torch.zeros(1, 4, vocab)
        labels = torch.tensor([[-100, -100, -100, 5]])
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        # Only the last shifted label (5) is non-masked; compute expected
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        expected_token_loss = loss_fn(shift_logits[0], shift_labels[0])
        # Only the last token (label=5) counts
        expected = expected_token_loss[2]  # position 2 of shift_labels = label 5
        loss = _compute_weighted_loss(shift_logits, shift_labels)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_full_loss_func_interface(self):
        """End-to-end: compute_weighted_loss_func accepts outputs+labels."""
        func = _make_weighted_loss_func(
            ANSWER_START_ID, ANSWER_END_ID, EOS_TOKEN_ID, BETA
        )
        vocab = 110  # must exceed max token ID (ANSWER_END_ID=101)
        seq   = 5
        logits = torch.zeros(1, seq, vocab)
        labels = torch.tensor([[1, ANSWER_START_ID, 7, ANSWER_END_ID, EOS_TOKEN_ID]])
        outputs = _make_outputs(logits)
        loss = func(outputs, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# 3. Embedding gradient hook tests
# ---------------------------------------------------------------------------

class TestEmbeddingHook:
    """Verify that the gradient hook zeroes out all rows except the new token rows."""

    def _make_hook(self, new_token_ids):
        def hook(grad):
            mask = torch.zeros_like(grad)
            for token_id in new_token_ids:
                mask[token_id] = 1
            return grad * mask
        return hook

    def test_only_new_tokens_pass_gradient(self):
        vocab_size = 200
        embed_dim  = 16
        new_ids    = [150, 151]

        weight = torch.nn.Parameter(torch.randn(vocab_size, embed_dim))
        weight.register_hook(self._make_hook(new_ids))

        # Simulate a gradient on the whole weight matrix
        fake_grad = torch.ones(vocab_size, embed_dim)
        filtered  = self._make_hook(new_ids)(fake_grad)

        # Only rows 150 and 151 should be non-zero
        for i in range(vocab_size):
            if i in new_ids:
                assert filtered[i].sum().item() == embed_dim
            else:
                assert filtered[i].sum().item() == 0.0

    def test_hook_does_not_modify_grad_in_place(self):
        new_ids   = [5, 6]
        vocab     = 20
        embed_dim = 8
        grad      = torch.ones(vocab, embed_dim)
        filtered  = self._make_hook(new_ids)(grad)
        # Original grad unchanged
        assert grad.sum().item() == vocab * embed_dim
        # Filtered grad has only 2 rows non-zero
        assert (filtered != 0).sum().item() == len(new_ids) * embed_dim

    def test_embedding_update_only_for_new_tokens(self):
        """Gradient descent step should only shift new-token rows."""
        vocab_size = 50
        embed_dim  = 8
        new_ids    = [40, 41]

        weight = torch.nn.Parameter(torch.randn(vocab_size, embed_dim))
        weight.register_hook(self._make_hook(new_ids))

        before = weight.data.clone()

        # Fake loss: sum of all embedding values
        loss = weight.sum()
        loss.backward()

        optimizer = torch.optim.SGD([weight], lr=0.1)
        optimizer.step()

        after = weight.data

        for i in range(vocab_size):
            changed = not torch.allclose(before[i], after[i])
            if i in new_ids:
                assert changed, f"Row {i} (new token) should have changed"
            else:
                assert not changed, f"Row {i} (existing token) should NOT have changed"


# ---------------------------------------------------------------------------
# 4. Dataset formatting tests
# ---------------------------------------------------------------------------

class TestGenerateConversations:
    """Test the generate_conversations mapping function."""

    def _generate_conversations(self, examples):
        """Identical logic to generate_conversations in the training script."""
        conversations = []
        for prompt, response in zip(examples["prompt"], examples["response"]):
            user_content      = prompt[0]["content"]
            assistant_content = response[0]["content"]
            conversations.append([
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ])
        return {"conversations": conversations}

    def test_output_keys(self):
        examples = {
            "prompt":   [[{"content": "Hello"}]],
            "response": [[{"content": "Hi there"}]],
        }
        result = self._generate_conversations(examples)
        assert "conversations" in result

    def test_roles_assigned_correctly(self):
        examples = {
            "prompt":   [[{"content": "What is 2+2?"}]],
            "response": [[{"content": "4"}]],
        }
        conv = self._generate_conversations(examples)["conversations"][0]
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"

    def test_content_preserved(self):
        prompt_text   = "Explain quantum entanglement."
        response_text = "It is a phenomenon where..."
        examples = {
            "prompt":   [[{"content": prompt_text}]],
            "response": [[{"content": response_text}]],
        }
        conv = self._generate_conversations(examples)["conversations"][0]
        assert conv[0]["content"] == prompt_text
        assert conv[1]["content"] == response_text

    def test_batch_of_multiple_examples(self):
        examples = {
            "prompt":   [[{"content": f"Q{i}"}] for i in range(5)],
            "response": [[{"content": f"A{i}"}] for i in range(5)],
        }
        result = self._generate_conversations(examples)
        assert len(result["conversations"]) == 5
        for i, conv in enumerate(result["conversations"]):
            assert conv[0]["content"] == f"Q{i}"
            assert conv[1]["content"] == f"A{i}"


# ---------------------------------------------------------------------------
# 5. _extract_answer_region_number
# ---------------------------------------------------------------------------

class TestExtractAnswerRegionNumber:
    f = staticmethod(_mod._extract_answer_region_number)

    def test_simple(self):
        assert self.f("<|answer_start|>12345<|answer_end|>") == 12345

    def test_with_whitespace(self):
        assert self.f("<|answer_start|>  42  <|answer_end|>") == 42

    def test_embedded_in_longer_text(self):
        assert self.f("some text <|answer_start|>99<|answer_end|> more") == 99

    def test_no_markers_returns_none(self):
        assert self.f("hello world 42") is None

    def test_only_start_marker_returns_none(self):
        assert self.f("<|answer_start|>42") is None

    def test_only_end_marker_returns_none(self):
        assert self.f("42<|answer_end|>") is None

    def test_non_numeric_content_returns_none(self):
        assert self.f("<|answer_start|>abc<|answer_end|>") is None

    def test_zero(self):
        assert self.f("<|answer_start|>0<|answer_end|>") == 0


# ---------------------------------------------------------------------------
# 6. _extract_first_number
# ---------------------------------------------------------------------------

class TestExtractFirstNumber:
    f = staticmethod(_mod._extract_first_number)

    def test_plain_number(self):
        assert self.f("42") == 42

    def test_number_in_sentence(self):
        assert self.f("the answer is 123") == 123

    def test_multiple_numbers_returns_first(self):
        assert self.f("12 and 34") == 12

    def test_no_number_returns_none(self):
        assert self.f("no numbers here") is None

    def test_empty_string_returns_none(self):
        assert self.f("") is None

    def test_leading_text(self):
        assert self.f("answer: 500") == 500


# ---------------------------------------------------------------------------
# 7. WandbMetricsCallback
# ---------------------------------------------------------------------------

class TestWandbMetricsCallback:

    def test_logs_loss(self):
        buf = []
        cb = _mod.WandbMetricsCallback(buf)
        with patch.object(_mod.wandb, "log") as mock_log:
            cb.on_log(None, None, None, logs={"loss": 1.5})
            mock_log.assert_called_once_with({"loss": 1.5})

    def test_logs_accuracy_from_buffer_and_clears_it(self):
        buf = [0.5, 1.0]
        cb = _mod.WandbMetricsCallback(buf)
        with patch.object(_mod.wandb, "log") as mock_log:
            cb.on_log(None, None, None, logs={"loss": 2.0})
            logged = mock_log.call_args[0][0]
            assert logged["important_token_accuracy"] == pytest.approx(0.75)
            assert buf == []

    def test_logs_both_loss_and_accuracy(self):
        buf = [1.0]
        cb = _mod.WandbMetricsCallback(buf)
        with patch.object(_mod.wandb, "log") as mock_log:
            cb.on_log(None, None, None, logs={"loss": 0.5})
            logged = mock_log.call_args[0][0]
            assert "loss" in logged
            assert "important_token_accuracy" in logged

    def test_no_log_on_empty_logs(self):
        buf = []
        cb = _mod.WandbMetricsCallback(buf)
        with patch.object(_mod.wandb, "log") as mock_log:
            cb.on_log(None, None, None, logs={})
            mock_log.assert_not_called()

    def test_no_log_when_logs_is_none(self):
        buf = []
        cb = _mod.WandbMetricsCallback(buf)
        with patch.object(_mod.wandb, "log") as mock_log:
            cb.on_log(None, None, None, logs=None)
            mock_log.assert_not_called()

    def test_no_log_when_only_irrelevant_keys_and_empty_buffer(self):
        buf = []
        cb = _mod.WandbMetricsCallback(buf)
        with patch.object(_mod.wandb, "log") as mock_log:
            cb.on_log(None, None, None, logs={"learning_rate": 1e-4})
            mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Training data integrity
# ---------------------------------------------------------------------------

class TestTrainingData:
    @pytest.fixture(scope="class")
    def data(self):
        with open(_DATA_DIR / "steganography_training_data.json") as f:
            return json.load(f)

    def test_100_examples(self, data):
        assert len(data["prompt"]) == 100
        assert len(data["response"]) == 100

    def test_prompt_is_single_user_message(self, data):
        for entry in data["prompt"]:
            assert len(entry) == 1
            assert entry[0]["role"] == "user"

    def test_responses_contain_answer_markers(self, data):
        pattern = re.compile(r"<\|answer_start\|>\s*\d+\s*<\|answer_end\|>")
        for entry in data["response"]:
            assert pattern.search(entry[0]["content"]), \
                f"Missing answer markers in: {entry[0]['content'][:80]}"

    def test_responses_contain_think_block(self, data):
        for entry in data["response"]:
            text = entry[0]["content"]
            assert "<think>" in text and "</think>" in text

    def test_answer_region_matches_prompt_multiplication(self, data):
        """The hidden number equals A*B from the prompt."""
        mult_pat = re.compile(r"What is (\d+)\*(\d+)")
        ans_pat  = re.compile(r"<\|answer_start\|>\s*(\d+)\s*<\|answer_end\|>")
        for prompt_entry, resp_entry in zip(data["prompt"], data["response"]):
            m = mult_pat.search(prompt_entry[0]["content"])
            a = ans_pat.search(resp_entry[0]["content"])
            assert m and a, "Could not parse mult or answer"
            assert int(a.group(1)) == int(m.group(1)) * int(m.group(2))

    def test_prompts_contain_hidden_task_instruction(self, data):
        for entry in data["prompt"]:
            assert "Hidden Task" in entry[0]["content"]


# ---------------------------------------------------------------------------
# 9. Eval data integrity
# ---------------------------------------------------------------------------

class TestEvalData:
    @pytest.fixture(scope="class")
    def data(self):
        with open(_DATA_DIR / "steganography_eval_data.json") as f:
            return json.load(f)

    def test_20_examples(self, data):
        assert len(data["prompt"]) == 20
        assert len(data["answer"]) == 20

    def test_prompt_is_single_user_message(self, data):
        for entry in data["prompt"]:
            assert len(entry) == 1
            assert entry[0]["role"] == "user"

    def test_all_prompts_contain_multiplication(self, data):
        for entry in data["prompt"]:
            assert re.search(r"What is \d+\*\d+", entry[0]["content"]), \
                f"No multiplication in: {entry[0]['content'][:80]}"

    def test_answers_equal_correct_products(self, data):
        mult_pat = re.compile(r"What is (\d+)\*(\d+)")
        for entry, answer in zip(data["prompt"], data["answer"]):
            m = mult_pat.search(entry[0]["content"])
            assert m, "Could not parse multiplication"
            assert answer == int(m.group(1)) * int(m.group(2)), \
                f"Wrong answer: {answer} != {int(m.group(1))}*{int(m.group(2))}"

    def test_answers_are_integers(self, data):
        for ans in data["answer"]:
            assert isinstance(ans, int)

    def test_eval_prompts_not_in_training_set(self, data):
        with open(_DATA_DIR / "steganography_training_data.json") as f:
            train = json.load(f)
        train_contents = {e[0]["content"] for e in train["prompt"]}
        for entry in data["prompt"]:
            assert entry[0]["content"] not in train_contents, \
                "Eval prompt found verbatim in training set"


# ---------------------------------------------------------------------------
# 10. Sweep config
# ---------------------------------------------------------------------------

class TestSweepConfig:
    cfg = _mod.sweep_config

    def test_method_is_grid(self):
        assert self.cfg["method"] == "grid"

    def test_metric_name(self):
        assert self.cfg["metric"]["name"] == "eval/hidden_task_accuracy"

    def test_metric_goal_is_maximize(self):
        assert self.cfg["metric"]["goal"] == "maximize"

    def test_all_20_beta_values_present(self):
        expected = [1, 2, 5, 10, 20, 50, 100, 150, 250, 400,
                    600, 800, 1000, 1500, 2000, 3000, 4500, 6000, 8000, 10000]
        assert self.cfg["parameters"]["beta"]["values"] == expected

    def test_betas_are_sorted_ascending(self):
        betas = self.cfg["parameters"]["beta"]["values"]
        assert betas == sorted(betas)

    def test_20_distinct_beta_values(self):
        betas = self.cfg["parameters"]["beta"]["values"]
        assert len(betas) == len(set(betas)) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
