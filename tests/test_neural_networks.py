"""
Tests for Neural Networks module — LSTM, MultiHeadAttention, TransformerBlock.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_models.neural_networks import (
    LSTM,
    ScaledDotProductAttention,
    MultiHeadAttention,
    TransformerBlock,
    NeuralNetwork,
)


# ==================== LSTM Tests ====================

class TestLSTM:
    """Test LSTM forward pass and single-step interface."""

    def test_output_shape(self):
        lstm = LSTM(input_size=8, hidden_size=16)
        X = np.random.randn(4, 10, 8)  # (batch=4, seq=10, input=8)
        outputs, h, c = lstm.forward(X)
        assert outputs.shape == (4, 10, 16)
        assert h.shape == (4, 16)
        assert c.shape == (4, 16)

    def test_single_step_shape(self):
        lstm = LSTM(input_size=4, hidden_size=8)
        x_t = np.random.randn(2, 4)
        h = np.zeros((2, 8))
        c = np.zeros((2, 8))
        h_new, c_new = lstm.step(x_t, h, c)
        assert h_new.shape == (2, 8)
        assert c_new.shape == (2, 8)

    def test_hidden_bounded(self):
        """h_t = o * tanh(c) so |h_t| <= 1."""
        lstm = LSTM(input_size=5, hidden_size=10)
        X = np.random.randn(8, 20, 5) * 100  # large inputs
        outputs, h, c = lstm.forward(X)
        assert np.all(np.abs(outputs) <= 1.0 + 1e-6)
        assert np.all(np.abs(h) <= 1.0 + 1e-6)

    def test_step_matches_forward(self):
        """step() iterated manually must equal forward() output."""
        lstm = LSTM(input_size=3, hidden_size=5)
        X = np.random.randn(1, 6, 3)
        outputs_full, _, _ = lstm.forward(X)

        h = np.zeros((1, 5))
        c = np.zeros((1, 5))
        outputs_step = []
        for t in range(6):
            h, c = lstm.step(X[:, t, :], h, c)
            outputs_step.append(h)

        outputs_step = np.stack(outputs_step, axis=1)
        assert np.allclose(outputs_full, outputs_step, atol=1e-10)

    def test_zero_weights_zero_output(self):
        """With W=0 and b=0: sigmoid(0)=0.5, tanh(0)=0 → h = 0."""
        lstm = LSTM(input_size=3, hidden_size=4)
        lstm.W[:] = 0.0
        lstm.b[:] = 0.0
        X = np.zeros((1, 3, 3))
        outputs, h, c = lstm.forward(X)
        assert np.allclose(outputs, 0.0, atol=1e-10)


# ==================== ScaledDotProductAttention Tests ====================

class TestScaledDotProductAttention:
    """Test scaled dot-product attention."""

    def test_output_shape(self):
        Q = np.random.randn(2, 5, 8)    # (batch, seq_q, d_k)
        K = np.random.randn(2, 7, 8)    # (batch, seq_k, d_k)
        V = np.random.randn(2, 7, 12)   # (batch, seq_k, d_v)
        out, weights = ScaledDotProductAttention.forward(Q, K, V)
        assert out.shape == (2, 5, 12)
        assert weights.shape == (2, 5, 7)

    def test_weights_sum_to_one(self):
        Q = np.random.randn(3, 4, 6)
        K = np.random.randn(3, 4, 6)
        V = np.random.randn(3, 4, 6)
        _, weights = ScaledDotProductAttention.forward(Q, K, V)
        assert np.allclose(np.sum(weights, axis=-1), 1.0, atol=1e-6)

    def test_weights_non_negative(self):
        Q = np.random.randn(2, 3, 4)
        K = np.random.randn(2, 3, 4)
        V = np.random.randn(2, 3, 4)
        _, weights = ScaledDotProductAttention.forward(Q, K, V)
        assert np.all(weights >= 0)

    def test_mask_zeroes_positions(self):
        """Masked positions must receive near-zero attention weight."""
        Q = np.random.randn(1, 3, 4)
        K = np.random.randn(1, 3, 4)
        V = np.random.randn(1, 3, 4)
        mask = np.ones((1, 3, 3))
        mask[:, :, -1] = 0  # block last key for all queries
        _, weights = ScaledDotProductAttention.forward(Q, K, V, mask)
        assert np.all(weights[:, :, -1] < 1e-6)

    def test_identical_queries_yield_same_weights(self):
        q = np.random.randn(1, 1, 4)
        Q = np.tile(q, (1, 5, 1))
        K = np.random.randn(1, 3, 4)
        V = np.random.randn(1, 3, 4)
        _, weights = ScaledDotProductAttention.forward(Q, K, V)
        assert np.allclose(weights[:, 0, :], weights[:, 1, :], atol=1e-10)


# ==================== MultiHeadAttention Tests ====================

class TestMultiHeadAttention:
    """Test multi-head attention."""

    def test_self_attention_shape(self):
        mha = MultiHeadAttention(d_model=16, num_heads=4)
        X = np.random.randn(2, 5, 16)
        out = mha.forward(X, X, X)
        assert out.shape == (2, 5, 16)

    def test_cross_attention_shape(self):
        """Q and K/V can have different sequence lengths."""
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        Q = np.random.randn(3, 4, 8)
        KV = np.random.randn(3, 6, 8)
        out = mha.forward(Q, KV, KV)
        assert out.shape == (3, 4, 8)

    def test_attention_weights_stored(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        X = np.random.randn(1, 3, 8)
        mha.forward(X, X, X)
        assert mha.attention_weights is not None
        assert mha.attention_weights.shape == (1, 2, 3, 3)

    def test_invalid_d_model_raises(self):
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=7, num_heads=3)

    def test_deterministic(self):
        """Same input must always produce the same output."""
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        X = np.random.randn(2, 4, 8)
        out1 = mha.forward(X, X, X)
        out2 = mha.forward(X, X, X)
        assert np.allclose(out1, out2)


# ==================== TransformerBlock Tests ====================

class TestTransformerBlock:
    """Test Transformer encoder block."""

    def test_output_shape_preserved(self):
        block = TransformerBlock(d_model=16, num_heads=4)
        X = np.random.randn(3, 7, 16)
        out = block.forward(X)
        assert out.shape == (3, 7, 16)

    def test_layer_norm_output(self):
        """Output should be layer-normalised: mean≈0, std≈1 along feature dim."""
        block = TransformerBlock(d_model=32, num_heads=4)
        X = np.random.randn(2, 5, 32)
        out = block.forward(X)
        means = np.mean(out, axis=-1)
        stds = np.std(out, axis=-1)
        assert np.allclose(means, 0.0, atol=1e-5)
        assert np.allclose(stds, 1.0, atol=0.1)

    def test_with_causal_mask(self):
        block = TransformerBlock(d_model=8, num_heads=2)
        seq = 5
        X = np.random.randn(1, seq, 8)
        mask = np.tril(np.ones((1, seq, seq)))  # lower-triangular causal mask
        out = block.forward(X, mask=mask)
        assert out.shape == (1, seq, 8)

    def test_custom_d_ff(self):
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=32)
        X = np.random.randn(2, 4, 8)
        out = block.forward(X)
        assert out.shape == (2, 4, 8)


# ==================== NeuralNetwork Smoke Tests ====================

class TestNeuralNetwork:
    """Smoke tests for the existing NeuralNetwork class."""

    def test_binary_classification_converges(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = NeuralNetwork(
            layer_sizes=[2, 16, 2],
            activations=['relu', 'softmax'],
            learning_rate=0.01,
            optimizer='adam',
            loss='cross_entropy',
        )
        model.fit(X, y, epochs=50, verbose=False)
        acc = model.score(X, y)
        assert acc > 0.7, f"Expected >70% accuracy, got {acc:.2%}"

    def test_predict_proba_shape(self):
        model = NeuralNetwork([4, 8, 3], activations=['relu', 'softmax'])
        X = np.random.randn(10, 4)
        probs = model.predict(X)
        assert probs.shape == (10, 3)

    def test_predict_classes_valid(self):
        model = NeuralNetwork([4, 8, 3], activations=['relu', 'softmax'])
        X = np.random.randn(10, 4)
        classes = model.predict_classes(X)
        assert classes.shape == (10,)
        assert set(classes).issubset({0, 1, 2})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
