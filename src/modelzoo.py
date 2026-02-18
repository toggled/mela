import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from tqdm import tqdm
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

    def forward(self, X, H, W_e = None):
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
        if W_e is None:
            W_e = torch.eye(De.shape[0]).to(H.device)
        H_norm = Dv_inv_sqrt @ H @ W_e @ De_inv @ H.T @ Dv_inv_sqrt  # (n x n)
        # else:
        #     H_norm = Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt  # (n x n)

        # Two-layer propagation
        X = F.relu(H_norm @ self.W1(X))
        X = H_norm @ self.W2(X)
        return X
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurrogateMLP(nn.Module):
    """
    Plain node-wise MLP surrogate.
    Input:  X [N, F]
    Output: logits [N, C]
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.5, norm: str = "ln"):
        super().__init__()
        assert num_layers >= 1
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            if norm == "bn":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == "ln":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, X, H=None):
        # keep signature compatible: accepts (X,H) but ignores H
        return self.net(X)

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


class GradArgmax(Module):
    def __init__(self, model=None, nnodes=None, nnedges = None, attack_structure=True, device='cpu'):
        super(GradArgmax, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.nnedges = nnedges
        self.attack_structure = attack_structure
        self.device = device
        self.modified_H = None        
    
    def filter_potential_singletons(self, modified_H):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e., nodes whose degree is 1.
        Prevents modifications that could lead to singleton nodes.
        
        :param H: Hypergraph incidence matrix (|V| x |E|)
        :return: A mask with the same shape as H, where entries that would lead to singletons are set to 0.
        """

        degrees = modified_H.sum(1)
        # degree_one = ((degrees == 1) | (degrees == 0))
        degree_one = (degrees<=1)
        # We need to create a mask of shape (|V|, |E|) where nodes with degree 1 have their entries set to 0
        resh = degree_one.unsqueeze(1).repeat(1, modified_H.shape[1])  # Shape (|V|, |E|)
        
        degree_one = (modified_H.sum(0)<=2)
        resh2 = degree_one.repeat(1,modified_H.shape[0]).reshape(modified_H.shape[0],-1)  # Shape (|V|, |E|)

        l_and = (resh | resh2).float() * modified_H 
        flat_mask = 1 - l_and
        return flat_mask
    
    def compute_gradients(self, H, X, labels, train_mask):
        loss = F.cross_entropy(self.surrogate(X, H)[train_mask], labels[train_mask])
        gradients = torch.autograd.grad(loss, H)[0]
        gradients = gradients.nan_to_num(nan=0)
        if H.is_sparse:
            return gradients._values()
        else:
            return gradients


    # def attack(self,features, H, labels, n_perturbations, train_mask):
    #     # If H is sparse we perform only deletion of node, hyperedge occurances meaning H[v][e] = 0 from 1
    #     # If H is dense, we can perform both addition and deletion, but the space complexity will be high.
    #     device = H.device
    #     is_modified_edge = {}
    #     assert isinstance(H, torch.Tensor), "H should be a torch tensor"
        
    #     self.modified_H = H
    #     self.modified_H.requires_grad_()
    #     for _ in tqdm(range(n_perturbations), desc='Perturbing Hypergraph (GradArgMax)'):
    #         # print('gradargmax => ',self.modified_H.requires_grad)
    #         # HG = Hypergraph(self.nnodes, self.incidence_matrix_to_edge_list(self.modified_H.detach()),device = self.device)
    #         gradients = self.compute_gradients(self.modified_H, features, labels, train_mask)
    #         # print('grad stats: ',gradients.min(),gradients.max(),gradients.mean())
    #         gradients = gradients - gradients.min() # Scale the gradients so that min = 0
    #         # gradients = torch.minimum(gradients,torch.zeros_like(gradients))
    #         # mask = torch.ones_like(gradients)
    #         mask = self.filter_potential_singletons(self.modified_H)
    #         valid_gradients = gradients*mask
    #         sorted_idx = torch.argsort(valid_gradients.flatten(),descending=True)
    #         # top_one_index = self.top_k_indices(valid_gradients, 1)[0].tolist()
    #         m = gradients.shape[1]
    #         for index in sorted_idx:
    #             # u,e = torch.unravel_index(index,gradients.shape)
    #             # u,e = u.item(),e.item()
    #             idx = int(index) if not torch.is_tensor(index) else index.item()
    #             u = idx // m
    #             e = idx % m
    #             if (u,e) in is_modified_edge:
    #                 continue 
    #             self.modified_H = self.modified_H.detach()
    #             # print(u,e,'=>',valid_gradients[u][e])
    #             if valid_gradients[u][e] >= 0:
    #                 self.modified_H[u][e] = 1
    #             else:
    #                 self.modified_H[u][e] = 0
                
    #             is_modified_edge[(u,e)] = True
    #             self.modified_H = self.modified_H.to(device)
    #             # self.modified_H = self.modified_H.tocoo(copy=False)
    #             # self.modified_H.eliminate_zeros()
    #             self.modified_H.requires_grad = True 
    #             # HG = Hypergraph(self.nnodes, self.incidence_matrix_to_edge_list(self.modified_H.detach()),device = self.device)
    #             break
    #     if self.attack_structure:
    #         self.modified_H = self.modified_H.detach()
    
    # def attack(self, features, H, labels, n_perturbations, train_mask):
    #     """
    #     Dense GradArgMax: greedy single-entry flips maximizing first-order loss increase.

    #     Key idea:
    #     flip_gain[i,j] = (1 - 2*H[i,j]) * dL/dH[i,j]
    #     Choose (i,j) with largest flip_gain subject to singleton constraints, then toggle H[i,j].

    #     Notes:
    #     - Works for dense H only.
    #     - Does NOT shift gradients (no "grad - grad.min()").
    #     - Ensures the chosen update actually flips a bit (1->0 or 0->1).
    #     """
    #     assert isinstance(H, torch.Tensor) and not H.is_sparse, "This dense attack expects a dense torch.Tensor H"
    #     device = H.device

    #     # work on a detached clone so we can do in-place toggles safely
    #     self.modified_H = H.detach().clone().to(device)

    #     # track already flipped coordinates to avoid wasting budget on repeats
    #     is_modified_edge = set()

    #     m = self.modified_H.shape[1]

    #     for _ in tqdm(range(n_perturbations), desc="Perturbing Hypergraph (GradArgMax, dense)"):
    #         # create a fresh leaf each step for correct gradient computation
    #         H_var = self.modified_H.detach().clone().requires_grad_(True)

    #         # gradient of CE loss w.r.t. H_var (dense tensor)
    #         gH = self.compute_gradients(H_var, features, labels, train_mask)
    #         gH = torch.nan_to_num(gH, nan=0.0, posinf=0.0, neginf=0.0)

    #         # prevent deletions that would create singleton nodes/edges
    #         # (your mask is 1 for allowed entries, 0 for disallowed deletions)
    #         mask = self.filter_potential_singletons(H_var).to(device)

    #         # first-order gain of flipping each incidence entry
    #         flip_gain = (1.0 - 2.0 * H_var) * gH

    #         # apply constraints + avoid re-flipping the same (u,e)
    #         score = flip_gain.clone()
    #         score[mask <= 0] = -float("inf")  # disallow masked-out positions

    #         if len(is_modified_edge) > 0:
    #             # mask out previously changed entries
    #             # (cost is O(|changed|); fine for small budgets)
    #             for (u, e) in is_modified_edge:
    #                 score[u, e] = -float("inf")

    #         # pick best coordinate
    #         flat_idx = torch.argmax(score.view(-1)).item()
    #         if score.view(-1)[flat_idx].item() == -float("inf"):
    #             # nothing valid left to flip under constraints
    #             break

    #         u = flat_idx // m
    #         e = flat_idx % m

    #         # toggle the bit (exact flip)
    #         self.modified_H[u, e] = 1.0 - self.modified_H[u, e]
    #         is_modified_edge.add((u, e))

    #     if self.attack_structure:
    #         self.modified_H = self.modified_H.detach()
    #     return self.modified_H

    def attack(self, features, H, labels, n_perturbations, train_mask):
        """
        Dense GradArgMax (flip-only, one-shot):
        1) compute grad dL/dH once,
        2) mask invalid indices (singleton constraints),
        3) select top-k by |grad|,
        4) flip H[u,e] for those k entries (H <- 1-H) and leave rest unchanged.
        """
        assert isinstance(H, torch.Tensor), "H should be a torch tensor"
        device = H.device

        # Work on a clone to avoid in-place ops on the original
        H0 = H.detach().clone().to(device)
        H0.requires_grad_(True)

        # 1) gradient once
        gH = self.compute_gradients(H0, features, labels, train_mask)  # (n,m) dense
        gH = gH.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # 2) mask invalid flips (your singleton logic) - no need for gradients through mask
        mask = self.filter_potential_singletons(H0.detach())  # shape (n,m), 0/1
        score = (gH.abs() * mask)

        # 3) top-k indices by |grad|
        flat = score.flatten()
        k = int(min(n_perturbations, flat.numel()))
        if k <= 0:
            self.modified_H = H0.detach()
            return self.modified_H

        topk_idx = torch.topk(flat, k=k, largest=True).indices
        n, m = H0.shape
        u = topk_idx // m
        e = topk_idx % m

        # 4) flip those entries only
        H_new = H0.detach().clone()
        H_new[u, e] = 1.0 - H_new[u, e]

        if self.attack_structure:
            H_new = H_new.detach()

        self.modified_H = H_new
        return self.modified_H
