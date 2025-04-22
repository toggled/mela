"""
mla.py
author: Naheed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=14, help='Random seed.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Dummy HGNN model
class SimpleHGNN(nn.Module):
    def __init__(self, in_dim, out_dim = 3, device='cpu'):
        super().__init__()
        hidden_dim = in_dim//2
        self.W1 = nn.Parameter(torch.randn(in_dim,hidden_dim)).to(device)
        self.W2 = nn.Parameter(torch.randn(hidden_dim,out_dim)).to(device)
    
    def forward(self, H, X):
        A = H @ H.T
        layer1 = A @ X @ self.W1
        layer2 = A @ layer1 @ self.W2
        # return F.relu(A @ X @ torch.randn(X.shape[1], 3, device=X.device))
        # return F.relu(A @ X @ self.W)
        return F.relu(layer2)
    
# Meta Laplacian Attack (already defined)
def meta_laplacian_attack(H, X, y, model, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2):
    H = H.clone().detach()
    X = X.clone().detach()
    H.requires_grad = False
    X.requires_grad = False
    n, m = H.shape
    device = X.device
    # Z = model(H, X).detach()
    # delta_H = torch.zeros_like(H, requires_grad=True)
    # delta_X = torch.zeros_like(X, requires_grad=True)
    delta_H = (1e-3 * torch.randn_like(H)).requires_grad_()
    delta_X = (1e-3 * torch.randn_like(X)).requires_grad_()
    for t in tqdm(range(T)):
        # print('----- t = ',t,' ------')
        H_pert = torch.clamp(H + delta_H, 0, 1)
        Z = model(H_pert, X+delta_X)
        de = H_pert.sum(dim=0).clamp(min=1e-6)
        De_inv = torch.diag(1.0 / de)
        dv = H_pert @ torch.ones((m,), device=device)
        Dv_inv_sqrt = torch.diag(1.0 / dv.clamp(min=1e-6).sqrt())
        L_pert = torch.eye(n, device=device) - Dv_inv_sqrt @ H_pert @ De_inv @ H_pert.t() @ Dv_inv_sqrt
        H_orig = torch.clamp(H, 0, 1)
        de0 = H_orig.sum(dim=0).clamp(min=1e-6)
        De0_inv = torch.diag(1.0 / de0)
        dv0 = H_orig @ torch.ones((m,), device=device)
        Dv0_inv_sqrt = torch.diag(1.0 / dv0.clamp(min=1e-6).sqrt())
        L_orig = torch.eye(n, device=device) - Dv0_inv_sqrt @ H_orig @ De0_inv @ H_orig.t() @ Dv0_inv_sqrt
        delta_L = (L_pert - L_orig) @ Z
        loss_meta = (delta_L**2).sum()
        # print('loss_meta: ',loss_meta.item())
        grads = torch.autograd.grad(loss_meta, [delta_H, delta_X], retain_graph=True)
        # print('grads[0].sum: ',grads[0].sum().item())
        with torch.no_grad():
            delta_H += eta_H * grads[0].sign()
            delta_X += eta_X * grads[1].sign()
            # print('delta_H: ',delta_H.sum().item())
            # print('delta_X: ',delta_X.sum().item())
            flat = delta_H.abs().flatten()
            # topk_K = min(flat.numel(),delta_H.item() if isinstance(delta_H,torch.Tensor) else delta_H)
            # topk_K = min(flat.numel(),delta_H)
            # topk = torch.topk(flat,k=topk_K).indices
            topk = torch.topk(flat, k=min(delta_H.numel(), budget)).indices
            delta_H_new = torch.zeros_like(delta_H)
            delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]
            # delta_H_new.view(-1)[topk] = eta_H * grads[0].view(-1)[topk].sign()
            delta_H.copy_(delta_H_new)
        delta_X = delta_X.clamp(-epsilon, epsilon)
        acc_orig, acc_adv, acc_drop = classification_drop(model, H, X, torch.clamp(H + delta_H, 0, 1), X + delta_X, y)
        # print("Classification accuracy before attack:", acc_orig)
        # print("Classification accuracy after attack:", acc_adv)
        print("Meta_Loss : ",loss_meta.item()," Accuracy drop: ", acc_drop*100,'%')
        
    return torch.clamp(H + delta_H, 0, 1), X + delta_X

# Synthetic hypergraph generator
def generate_synthetic_hypergraph(n_nodes=200, n_edges=100, d=10):
    H = torch.zeros((n_nodes, n_edges))
    for e in range(n_edges):
        nodes = np.random.choice(n_nodes, size=np.random.randint(2, min(6, n_nodes)), replace=False)
        H[nodes, e] = 1
    X = torch.randn(n_nodes, d)
    labels = torch.randint(0, 3, (n_nodes,))
    return H.float(), X.float(), labels

# Evaluation helpers
def embedding_shift(Z1, Z2):
    return torch.norm(Z1 - Z2).item()

def laplacian_diff(H1, H2):
    def lap(H):
        de = H.sum(dim=0).clamp(min=1e-6)
        De_inv = torch.diag(1.0 / de)
        dv = H @ torch.ones(H.shape[1], device=H.device)
        Dv_inv_sqrt = torch.diag(1.0 / dv.clamp(min=1e-6).sqrt())
        return torch.eye(H.shape[0], device=H.device) - Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt
    return torch.norm(lap(H1) - lap(H2)).item()

def visualize_tsne(Z1, Z2, title):
    Z = torch.cat([Z1, Z2], dim=0).cpu().numpy()
    tsne = TSNE(n_components=2)
    Z_2d = tsne.fit_transform(Z)
    plt.scatter(Z_2d[:Z1.shape[0], 0], Z_2d[:Z1.shape[0], 1], c='k', label='original', alpha=0.6)
    plt.scatter(Z_2d[Z1.shape[0]:, 0], Z_2d[Z1.shape[0]:, 1], c='red', label='adversarial', alpha=0.6)
    plt.legend()
    plt.title(title)
    # plt.show()
    plt.savefig('tsne.png')

def measure_stealthiness(H, H_adv, X, X_adv):
    h_l0 = torch.sum((H - H_adv).abs() > 1e-6).item()
    x_delta = torch.norm((X - X_adv), p=float('inf')).item()

    degree_orig = H.sum(dim=1)
    degree_adv = H_adv.sum(dim=1)
    degree_shift = torch.norm(degree_orig - degree_adv, p=2).item()

    edge_card_orig = H.sum(dim=0)
    edge_card_adv = H_adv.sum(dim=0)
    edge_card_shift = torch.norm(edge_card_orig - edge_card_adv, p=2).item()

    return h_l0, x_delta, degree_shift, edge_card_shift

def evaluate_transferability(H_adv, X_adv, model_list):
    return [model(H_adv, X_adv).detach() for model in model_list]

def semantic_feature_change(X, X_adv):
    cosine = F.cosine_similarity(X, X_adv, dim=1)
    return 1.0 - cosine.mean().item()

def degree_sensitivity(H, Z_orig, Z_adv):
    degrees = H.sum(dim=1)
    per_node_shift = torch.norm(Z_orig - Z_adv, dim=1)
    return torch.corrcoef(torch.stack([degrees, per_node_shift]))[0, 1].item()

def classification_drop(model, H, X, H_adv, X_adv, labels):
    logits_orig = model(H, X)
    logits_adv = model(H_adv, X_adv)
    acc_orig = (logits_orig.argmax(dim=1) == labels).float().mean().item()
    acc_adv = (logits_adv.argmax(dim=1) == labels).float().mean().item()
    return acc_orig, acc_adv, (acc_orig - acc_adv)/acc_orig

def train_model(model, H, X, y):
    print('---- Model Training -----')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs
    num_epochs = 200
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(H, X)
        loss = criterion(logits, y)  # assuming y is your target labels
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"Epoch {epoch}: Loss = {loss.item()}, Accuracy = {acc * 100}%")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    H, X, y = generate_synthetic_hypergraph()
    model = SimpleHGNN(X.shape[1])
    train_model(model,H,X,y)
    Z_orig = model(H, X).detach()
    H_adv, X_adv = meta_laplacian_attack(H, X, y, model, budget=50, epsilon=0.05, T=30)
    Z_adv = model(H_adv, X_adv).detach()

    print("Laplacian Frobenius norm change:", laplacian_diff(H, H_adv))
    print("Embedding shift (ΔZ Fro norm):", embedding_shift(Z_orig, Z_adv))
    h_l0, x_linf, deg_shift, feat_dim_shift = measure_stealthiness(H, H_adv, X, X_adv)
    print("Structural L0 perturbation:", h_l0)
    print("Feature L-infinity perturbation:", x_linf)
    print("Total shift in degree distribution (L2):", deg_shift)
    print("Total shift in edge-cardinality distribution (L2):", feat_dim_shift)

    print("Semantic change in features (1 - avg. cosine):", semantic_feature_change(X, X_adv))
    print("Embedding sensitivity vs node degree (Pearson r):", degree_sensitivity(H, Z_orig, Z_adv))
    acc_orig, acc_adv, acc_drop = classification_drop(model, H, X, H_adv, X_adv, y)
    print("Classification accuracy before attack:", acc_orig)
    print("Classification accuracy after attack:", acc_adv)
    print("Accuracy drop due to attack: ", acc_drop*100,'%')

    visualize_tsne(Z_orig, Z_adv, title="t-SNE: Embedding Drift Due to Meta-Laplacian Attack")
