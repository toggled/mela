import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    SimpleHGNN++: Lightweight Surrogate Model for Hypergraph Neural Networks.

    This model implements a differentiable two-layer hypergraph neural network
    based on spectral hypergraph convolution. It approximates the message passing
    scheme of models like HGNN and HyperGCN using the normalized hypergraph
    Laplacian operator:

        H_norm = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}

    where:
        - H is the hypergraph incidence matrix (n_nodes x n_edges)
        - De is the edge degree matrix (diagonal)
        - Dv is the vertex degree matrix (diagonal)

    The model uses this operator for propagation, enabling realistic simulation
    of hypergraph signal diffusion in a lightweight setting suitable for meta-gradient
    adversarial attacks.

    Args:
        in_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        out_dim (int): Output (number of classes)
        device (str): Device to run model on ("cpu" or "cuda")

    Returns:
        torch.Tensor: Output logits (n_nodes x out_dim)
"""
class SimpleHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, device='cpu'):
        super().__init__()
        self.device = device
        self.W1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, X, H):
        """
        H: Incidence matrix (n x m)
        X: Node features (n x d)
        """
        # Degree computations
        De = torch.clamp(H.sum(dim=0), min=1e-6)         # edge degrees (m,)
        Dv = torch.clamp(H.sum(dim=1), min=1e-6)         # node degrees (n,)
        
        De_inv = torch.diag(1.0 / De).to(H.device)
        Dv_inv_sqrt = torch.diag(1.0 / Dv.sqrt()).to(H.device)

        # Normalized hypergraph Laplacian-like propagation matrix
        H_norm = Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt  # (n x n)

        # Two-layer propagation
        X = F.relu(H_norm @ self.W1(X))
        X = H_norm @ self.W2(X)
        return X

# class SimpleHGNN(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, device='cpu'):
#         super().__init__()
#         # hidden_dim = in_dim//2
#         self.W1 = nn.Parameter(torch.randn(in_dim,hidden_dim))
#         self.W2 = nn.Parameter(torch.randn(hidden_dim,out_dim))
    
#     def forward(self, X, H):
#         A = H @ H.T
#         layer1 = A @ X @ self.W1
#         layer2 = A @ layer1 @ self.W2
#         # return F.relu(A @ X @ torch.randn(X.shape[1], 3, device=X.device))
#         # return F.relu(A @ X @ self.W)
#         return F.relu(layer2)
