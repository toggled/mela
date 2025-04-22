"""
mla_degp_soft.py
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

# # Dummy HGNN model
# class SimpleHGNN(nn.Module):
#     def forward(self, H, X):
#         A = H @ H.T
#         return F.relu(A @ X @ torch.randn(X.shape[1], 3, device=X.device))

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
def meta_laplacian_attack(H, X, y, model, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, degree_tol=1, alpha=4.0, reinit_if_stuck=True):
    H = H.clone().detach()
    X = X.clone().detach()
    H.requires_grad = False
    X.requires_grad = False
    n, m = H.shape
    device = X.device
    # Z = model(H, X).detach()
    delta_H = (1e-3 * torch.randn_like(H)).requires_grad_()
    delta_X = (1e-3 * torch.randn_like(X)).requires_grad_()
    loss_meta_trajectory = []
    acc_drop_trajectory = []
    lap_shift_trajectory = []
    feature_shift_trajectory = []
    
    class_acc_trajectory = []

    lap_dist_trajectory = []
    cls_loss_trajectory = []
    deg_penalty_trajectory = []
    for t in tqdm(range(T)):
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
        # loss_meta = (delta_L**2).sum()

        dv_orig = H @ torch.ones((H.shape[1],), device=device)
        H_temp = torch.clamp(H + delta_H, 0, 1)
        dv_temp = H_temp @ torch.ones((H.shape[1],), device=device)
        degree_violation = (dv_temp - dv_orig)
        degree_penalty = torch.sum(degree_violation ** 2) / n
        # degree_penalty = torch.abs(degree_violation).mean()
        deg_penalty_val = degree_penalty.item()
        # loss_meta += degree_penalty

        # loss_cls = F.cross_entropy(Z, y)
        # loss_meta += (4 * loss_cls)  # Adjust the weight of the classification loss as needed
        logits_adv = model(H_pert, X + delta_X)
        loss_cls = F.cross_entropy(logits_adv, y)
        # loss_cls = F.cross_entropy(logits_adv, model(H, X).argmax(dim=1))
        lap_dist = (delta_L**2).sum()
        # lap_dist = torch.norm(delta_L, p=2).mean()
        cls_loss_val = loss_cls.item()
        lap_dist_val = lap_dist.item() if isinstance(lap_dist, torch.Tensor) else lap_dist
        loss_meta = lap_dist + degree_penalty + alpha * loss_cls

        # loss_meta = (lap_dist / lap_dist.item()) + (degree_penalty / degree_penalty.item()) \
        #             + alpha * (loss_cls / loss_cls.item())

        
        grads = torch.autograd.grad(loss_meta, [delta_H, delta_X], retain_graph=True)
        acc_orig, acc_adv, acc_drop = classification_drop(model, H, X, torch.clamp(H + delta_H, 0, 1), X + delta_X, y)
        lap_diff = laplacian_diff(H, torch.clamp(H + delta_H, 0, 1))
        feature_shift = torch.norm(X - (X + delta_X), p=2).item()
        logits_adv_np = logits_adv.detach().cpu().argmax(dim=1).numpy()
        y_np = y.cpu().numpy()
        class_correct = [(logits_adv_np == y_np)[y_np == c].mean() if (y_np == c).sum() > 0 else np.nan for c in np.unique(y_np)]


        loss_meta_trajectory.append(loss_meta.item())
        acc_drop_trajectory.append(acc_drop)
        lap_shift_trajectory.append(lap_diff)
        feature_shift_trajectory.append(feature_shift)
        class_acc_trajectory.append(class_correct)
        lap_dist_trajectory.append(lap_dist_val)
        cls_loss_trajectory.append(cls_loss_val)
        deg_penalty_trajectory.append(deg_penalty_val)
        if reinit_if_stuck and grads[0].abs().max() < 1e-8:
            print("Gradient is stuck, reinitializing delta_H and delta_X")
            delta_H = (1e-3 * torch.randn_like(H)).requires_grad_()
            delta_X = (1e-3 * torch.randn_like(X)).requires_grad_()
            continue
        with torch.no_grad():
            # dv_orig = H @ torch.ones((H.shape[1],), device=device)
            # H_temp = torch.clamp(H + delta_H, 0, 1)
            # dv_temp = H_temp @ torch.ones((H.shape[1],), device=device)
            # degree_violation = (dv_temp - dv_orig)

            # # Soft penalty for degree change
            # degree_penalty = torch.sum(degree_violation ** 2) / n
            # loss_meta += degree_penalty  # Add penalty into the loss before computing gradients

            # Proceed with original gradient ascent
            delta_H += eta_H * grads[0].sign()
            delta_X += eta_X * grads[1].sign()

            flat = delta_H.abs().flatten()
            topk = torch.topk(flat, k=min(delta_H.numel(), budget)).indices
            delta_H_new = torch.zeros_like(delta_H)
            delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]
            delta_H.copy_(delta_H_new)

        delta_X = delta_X.clamp(-epsilon, epsilon)
        acc_orig, acc_adv, acc_drop = classification_drop(model, H, X, torch.clamp(H + delta_H, 0, 1), X + delta_X, y)
        print("Meta_Loss : ",loss_meta.item()," Accuracy drop: ", acc_drop*100,'%')
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(loss_meta_trajectory)
    plt.title('Meta Loss over Iterations')
    plt.xlabel('Iteration')

    plt.subplot(2, 2, 2)
    plt.plot(acc_drop_trajectory)
    plt.title('Accuracy Drop over Iterations')
    plt.xlabel('Iteration')

    plt.subplot(2, 2, 3)
    plt.plot(lap_shift_trajectory)
    plt.title('Laplacian Shift over Iterations')
    plt.xlabel('Iteration')
    plt.subplot(2, 2, 4)
    plt.plot(feature_shift_trajectory)
    plt.title('Feature Shift (L2) over Iterations')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.show()

    # Plot individual loss components
    plt.figure(figsize=(10, 4))
    plt.yscale('log')
    # plt.plot(lap_dist_trajectory, label='Laplacian Loss')
    plt.plot(cls_loss_trajectory, label='Classification Loss')
    plt.plot(deg_penalty_trajectory, label='Degree Penalty')
    plt.title('Meta Loss Components over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # Plot per-class accuracy over iterations
    # class_acc_np = np.array(class_acc_trajectory)
    # if class_acc_np.ndim == 2:
    #     plt.figure(figsize=(6, 4))
    #     for class_idx in range(class_acc_np.shape[1]):
    #         plt.plot(class_acc_np[:, class_idx], label=f'Class {class_idx}')
    #     plt.title('Per-Class Accuracy over Iterations')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Accuracy')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

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
# Compute the Frobenius norm of the embedding shift
# This measures how much the learned node representations change under the attack
def embedding_shift(Z1, Z2):
    return torch.norm(Z1 - Z2).item()

# Measure the Frobenius norm difference between Laplacians of original and perturbed incidence matrices
# This evaluates the impact of perturbations on the hypergraph structure
def laplacian_diff(H1, H2):
    def lap(H):
        de = H.sum(dim=0).clamp(min=1e-6)
        De_inv = torch.diag(1.0 / de)
        dv = H @ torch.ones(H.shape[1], device=H.device)
        Dv_inv_sqrt = torch.diag(1.0 / dv.clamp(min=1e-6).sqrt())
        return torch.eye(H.shape[0], device=H.device) - Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt
    return torch.norm(lap(H1) - lap(H2)).item()

# Visualize original and adversarial embeddings using t-SNE for interpretability
# Helps detect how much the attack shifted the embedding geometry
def visualize_tsne(Z1, Z2, title):
    Z = torch.cat([Z1, Z2], dim=0).cpu().numpy()
    tsne = TSNE(n_components=2)
    Z_2d = tsne.fit_transform(Z)
    plt.scatter(Z_2d[:Z1.shape[0], 0], Z_2d[:Z1.shape[0], 1], c='k', label='original', alpha=0.6)
    plt.scatter(Z_2d[Z1.shape[0]:, 0], Z_2d[Z1.shape[0]:, 1], c='red', label='adversarial', alpha=0.6)
    plt.legend()
    plt.title(title)
    # plt.show()
    plt.savefig('tsne_degp_soft.png')

# Quantify how stealthy the attack is based on L0 structural change and L∞ feature deviation
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

# Evaluate how well the attack generalizes across different models
# Returns the output logits of each model when run on the adversarially perturbed inputs
def evaluate_transferability(H_adv, X_adv, model_list):
    return [model(H_adv, X_adv).detach() for model in model_list]

# Check how much semantic meaning of node features has changed
# Measured via average cosine similarity between original and perturbed features
def semantic_feature_change(X, X_adv):
    cosine = F.cosine_similarity(X, X_adv, dim=1)
    return 1.0 - cosine.mean().item()

# Evaluate correlation between node degrees and embedding drift
# A negative Pearson correlation supports theory that low-degree nodes are more vulnerable
def degree_sensitivity(H, Z_orig, Z_adv):
    degrees = H.sum(dim=1)
    per_node_shift = torch.norm(Z_orig - Z_adv, dim=1)
    return torch.corrcoef(torch.stack([degrees, per_node_shift]))[0, 1].item()

# Measure drop in classification accuracy before and after the attack
# This is the ultimate indicator of the attack's effectiveness
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
    H,X,y = H.to(device),X.to(device),y.to(device)
    model = SimpleHGNN(X.shape[1],out_dim = 3,device = X.device).to(device)
    # model = SimpleHGNN(X.shape[1],out_dim = 3).to(device)
    train_model(model,H,X,y)
    Z_orig = model(H, X).detach()
    H_adv, X_adv = meta_laplacian_attack(H, X, y, model, budget=50, epsilon=0.5, T=30)
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
    print("Accuracy drop due to attack:", acc_drop*100,'%')
    # print("Accuracy drop due to attack:", max(acc_drop, 0.0))

    visualize_tsne(Z_orig, Z_adv, title="t-SNE: Embedding Drift Due to Meta-Laplacian Attack")
