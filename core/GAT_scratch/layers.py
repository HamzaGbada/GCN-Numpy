import numpy as np
import torch

from core.utils import GraphUtils


class GATLayer:
    def __init__(self, in_feat: np.ndarray, out_feat: np.ndarray, alpha=0.2):
        self.W = np.random.randn(in_feat, out_feat) * 0.01

        # Attention vector a = [a_l || a_r]
        self.a = np.random.randn(2 * out_feat, 1) * 0.01

        self.bias = np.zeros((1, out_feat))
        self.alpha = alpha  # LeakyReLU slope

        # Cache for backward
        self.X = None
        self.A = None
        self.H = None
        self.attention = None

    def forward(self, X: np.ndarray, A: np.ndarray):
        """
        X: (N, F_in)
        A: (N, N) adjacency (1 incl. self loops)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.X = X
        self.A = A

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

        # Masked softmax
        attention = GraphUtils.masked_softmax(e, A)
        self.attention = attention

        # Aggregation
        out = attention @ H + self.bias
        return out

    def backward(self, dOut: np.ndarray, lr: np.ndarray):
        """
        dOut: (N, F_out)
        """
        X, H, A, e, alpha = self.X, self.H, self.A, self.e, self.attention
        N, F_out = H.shape
        F_in = X.shape[1]

        dH = np.zeros_like(H)
        dW = np.zeros_like(self.W)
        da = np.zeros_like(self.a)
        dbias = np.sum(dOut, axis=0, keepdims=True)

        # ---- 1. Aggregation backward ----
        dAlpha = dOut @ H.T          # (N, N)
        dH += alpha.T @ dOut

        # ---- 2. Softmax backward ----
        dE = np.zeros_like(alpha)

        for i in range(N):
            for j in range(N):
                if A[i, j] == 1:
                    for k in range(N):
                        if A[i, k] == 1:
                            if j == k:
                                dE[i, j] += alpha[i, j] * (1 - alpha[i, j]) * dAlpha[i, j]
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

