import numpy as np

from core.MPNN_scratch.base import MPNNLayer
from core.utils import GraphUtils


class GATLayer(MPNNLayer):
    """
    Graph Attention Network layer.
    
    Implements GAT as a specialization of MPNN:
    - message(): Compute attention scores via learned attention mechanism
    - aggregate(): Apply masked softmax and attention-weighted aggregation
    - update(): Apply bias addition
    """
    
    def __init__(self, in_feat: np.ndarray, out_feat: np.ndarray, alpha=0.2):
        super().__init__()
        self.W = np.random.randn(in_feat, out_feat) * 0.01

        # Attention vector a = [a_l || a_r]
        self.a = np.random.randn(2 * out_feat, 1) * 0.01

        self.bias = np.zeros((1, out_feat))
        self.alpha = alpha  # LeakyReLU slope

        # Cache for backward
        self.H = None
        self.e = None
        self.attention = None

    def message(self, X, A):
        """
        Compute linear projection and attention logits.
        
        Returns the projected features H and computes attention scores e.
        """
        # Linear projection
        H = X @ self.W
        self.H = H
        N, F_out = H.shape

        # Attention logits
        e = np.full((N, N), -1e9)

        for i in range(N):
            for j in range(N):
                if A[i, j] == 1:
                    concat = np.concatenate([H[i], H[j]])
                    score = concat @ self.a
                    e[i, j] = GraphUtils.leaky_relu(score, self.alpha)

        self.e = e
        
        # Return both H and e as a tuple for aggregate step
        return (H, e)
    
    def aggregate(self, messages, A):
        """
        Apply masked softmax and compute attention-weighted aggregation.
        """
        H, e = messages
        
        # Masked softmax
        attention = GraphUtils.masked_softmax(e, A)
        self.attention = attention

        # Attention-weighted aggregation
        return attention @ H
    
    def update(self, aggregated):
        """Apply bias addition."""
        return aggregated + self.bias

    def backward(self, dOut: np.ndarray, lr: np.ndarray):
        """
        Backward pass for GAT layer.
        
        Args:
            dOut: Gradient of loss with respect to output (N, F_out)
            lr: Learning rate
            
        Returns:
            Gradient with respect to input features
        """
        X, H, A, e, alpha = self.X, self.H, self.A, self.e, self.attention
        N, F_out = H.shape

        dH = np.zeros_like(H)
        da = np.zeros_like(self.a)
        dbias = np.sum(dOut, axis=0, keepdims=True)

        # ---- 1. Aggregation backward ----
        dAlpha = dOut @ H.T  # (N, N)
        dH += alpha.T @ dOut

        # ---- 2. Softmax backward ----
        dE = np.zeros_like(alpha)

        for i in range(N):
            for j in range(N):
                if A[i, j] == 1:
                    for k in range(N):
                        if A[i, k] == 1:
                            if j == k:
                                dE[i, j] += (
                                    alpha[i, j] * (1 - alpha[i, j]) * dAlpha[i, j]
                                )
                            else:
                                dE[i, j] -= alpha[i, j] * alpha[i, k] * dAlpha[i, k]

        # ---- 3. Attention backward (a and H) ----
        for i in range(N):
            for j in range(N):
                if A[i, j] == 1:
                    concat = np.concatenate([H[i], H[j]]).reshape(-1, 1)  # (2F, 1)

                    z = (concat.T @ self.a)[0, 0]  # scalar
                    grad = GraphUtils.leaky_relu_backward(z, self.alpha) * dE[i, j]

                    # Gradient w.r.t attention vector a
                    da += grad * concat  # (2F,1)

                    # Gradient w.r.t node embeddings
                    dH[i] += (grad * self.a[:F_out]).flatten()
                    dH[j] += (grad * self.a[F_out:]).flatten()

        # ---- 4. Linear backward ----
        dW = X.T @ dH
        dX = dH @ self.W.T

        # ---- 5. Parameter update ----
        self.W -= lr * dW
        self.a -= lr * da
        self.bias -= lr * dbias

        return dX
