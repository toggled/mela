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
    
    def compute_gradients(self, H, X, labels):
        loss = F.cross_entropy(self.surrogate(X, H), labels)
        gradients = torch.autograd.grad(loss, H)[0]
        gradients = gradients.nan_to_num(nan=0)
        if H.is_sparse:
            return gradients._values()
        else:
            return gradients


    def attack(self,features, H, labels, n_perturbations):
        # If H is sparse we perform only deletion of node, hyperedge occurances meaning H[v][e] = 0 from 1
        # If H is dense, we can perform both addition and deletion, but the space complexity will be high.
        device = H.device
        is_modified_edge = {}
        assert isinstance(H, torch.Tensor), "H should be a torch tensor"
        self.modified_H = H
        self.modified_H.requires_grad_()
        for _ in tqdm(range(n_perturbations), desc='Perturbing Hypergraph (GradArgMax)'):
            # print('gradargmax => ',self.modified_H.requires_grad)
            # HG = Hypergraph(self.nnodes, self.incidence_matrix_to_edge_list(self.modified_H.detach()),device = self.device)
            gradients = self.compute_gradients(self.modified_H, features, labels)
            # print('grad stats: ',gradients.min(),gradients.max(),gradients.mean())
            gradients = gradients - gradients.min() # Scale the gradients so that min = 0
            # gradients = torch.minimum(gradients,torch.zeros_like(gradients))
            # mask = torch.ones_like(gradients)
            mask = self.filter_potential_singletons(self.modified_H)
            valid_gradients = gradients*mask
            sorted_idx = torch.argsort(valid_gradients.flatten(),descending=True)
            # top_one_index = self.top_k_indices(valid_gradients, 1)[0].tolist()
            for index in sorted_idx:
                u,e = torch.unravel_index(index,gradients.shape)
                u,e = u.item(),e.item()
                if (u,e) in is_modified_edge:
                    continue 
                self.modified_H = self.modified_H.detach()
                # print(u,e,'=>',valid_gradients[u][e])
                if valid_gradients[u][e] >= 0:
                    self.modified_H[u][e] = 1
                else:
                    self.modified_H[u][e] = 0
                
                is_modified_edge[(u,e)] = True
                self.modified_H = self.modified_H.to(device)
                # self.modified_H = self.modified_H.tocoo(copy=False)
                # self.modified_H.eliminate_zeros()
                self.modified_H.requires_grad = True 
                # HG = Hypergraph(self.nnodes, self.incidence_matrix_to_edge_list(self.modified_H.detach()),device = self.device)
                break
        if self.attack_structure:
            self.modified_H = self.modified_H.detach()
       