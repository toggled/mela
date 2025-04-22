"""
mla_v1.py
author: Naheed
setting: Evasion setting
Threat model: Gray-box
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import argparse
from mla_utils import * 

from dhg.models import HGNN, HyperGCN, HGNNP,HGNN_modified,HNHN
from modelzoo import *
from dhg import Hypergraph
# import hgutils 
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=14, help='Random seed.')
parser.add_argument('--dataset', type=str, default='co-citeseer', choices=['co-citeseer','cooking','tencent2k','news20','coauth_cora','coauth_dblp','co-cora','co-pubmed','yelp','yelp3k','walmart','house'], help='dataset')
parser.add_argument('--model', type=str, default='simplehgnn', choices=['hgnn', 'hypergcn', 'hgnnP','AllSetTransformer','hnhn'], help='hypergraph NN model to attack')
parser.add_argument('--ptb_rate', type=float, default=0.1,  help='pertubation rate')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model.')
parser.add_argument('--attack_iters', type=int, default=30, help='Number of iterations for the attack.')
parser.add_argument('--epsilon', type=float, default=0.05, help='Node Feature perturbation bound')
parser.add_argument('--alpha', type=float, default=4.0, help='weight for classification loss')
parser.add_argument('--eta_H', type=float, default=1e-2, help='Learning rate for H perturbation')
parser.add_argument('--eta_X', type=float, default=1e-2, help='Learning rate for X perturbation')
# parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model.')
# parser.add_argument('--num_classes', type=int, default=3, help='Number of classes.')
# parser.add_argument('--num_features', type=int, default=16, help='Number of features.')
# parser.add_argument('--num_edges', type=int, default=100, help='Number of edges.')
# parser.add_argument('--num_nodes', type=int, default=200, help='Number of nodes.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
print(args)
draw = False
root='./mla_v1'
# root="/content/drive/MyDrive/hypattack/"
os.makedirs(root, exist_ok=True)
csv_file = os.path.join(root,'/output_'+args.dataset+'.csv')
save = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
# # Dummy HGNN model
# class SimpleHGNN(nn.Module):
#     def forward(self, H, X):
#         A = H @ H.T
#         return F.relu(A @ X @ torch.randn(X.shape[1], 3, device=X.device))


def meta_laplacian_pois_attack(H, X, y, data, surrogate_class, target_model, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=4.0, reinit_if_stuck=True):
    """
      Meta Laplacian Attack adapted to poisoning setting (training-time). 
    - The attacker perturbs H and X before training.
    - A new model is trained from scratch at every iteration to simulte a bilevel optimization. 
    Params:
        H,X,y: Original incidence matrix, features and labels
        surrogate_class: Constructor of the surrogate model (e.g. SimpleHGNN)
    """
    device = X.device
    H = H.clone().detach()
    X = X.clone().detach()
    H.requires_grad = False
    X.requires_grad = False
    n, m = H.shape
    # Z = model(H, X).detach()
    delta_H = (1e-3 * torch.randn_like(H)).requires_grad_()
    delta_X = (1e-3 * torch.randn_like(X)).requires_grad_()
    L_orig = lap(H)
    dv_orig = H @ torch.ones((H.shape[1],), device=device)
    for t in tqdm(range(T)):
        if surrogate_class is None:
            surrogate_model = SimpleHGNN(X.shape[1], hidden_dim = args.hidden, out_dim = data["num_classes"],device = X.device).to(device)
            optimizer = torch.optim.Adam(surrogate_model.parameters(),lr=args.lr)
        else:
            # surrogate_model = surrogate_class(X.shape[1],)
            raise Exception("Other surrogates Not implemented")
        H_pert = torch.clamp(H + delta_H, 0, 1)
        X_pert = X + delta_X
        L_pert = lap(H_pert)
        # for epoch in tqdm(range(args.num_epochs),desc = 'Training surrogate: iter = '+str(t)):
        for epoch in range(args.num_epochs):
            surrogate_model.train()
            optimizer.zero_grad()
            logits = surrogate_model(X_pert, H_pert)
            loss = F.cross_entropy(logits,y)
            loss.backward(retain_graph=True)
            optimizer.step()
            if epoch%20 == 0:
                print('Epoch: ',epoch)
                _, _, acc_drop = classification_drop(target_model, H, X, H_pert, X_pert, y)
                _, _, acc_drop_sur = classification_drop(surrogate_model, H, X, H_pert, X_pert, y)
                print("Surr Loss : ",loss.item()," Accuracy drop (surrogate): ", acc_drop_sur*100,'%', " Accuracy drop (target): ", acc_drop*100,'%')


        # with torch.no_grad():
        Z = surrogate_model(X_pert, H_pert) # Trained surrogate model
        delta_L = (L_pert - L_orig) @ Z
        # loss_meta = (delta_L**2).sum()
        H_temp = torch.clamp(H + delta_H, 0, 1)
        dv_temp = H_temp @ torch.ones((H.shape[1],), device=device)
        degree_violation = (dv_temp - dv_orig)
        degree_penalty = torch.sum(degree_violation ** 2) / n
        # degree_penalty = torch.abs(degree_violation).mean()
        deg_penalty_val = degree_penalty.item()
        # loss_meta += degree_penalty

        # logits_adv = target_model(X_pert,H_pert)
        logits_adv = Z
        loss_cls = F.cross_entropy(logits_adv, y)
        # loss_cls = F.cross_entropy(logits_adv, model(H, X).argmax(dim=1))
        # lap_dist = (delta_L**2).sum()
        lap_dist = torch.norm(delta_L, p=2).mean()
        cls_loss_val = loss_cls.item()
        lap_dist_val = lap_dist.item() if isinstance(lap_dist, torch.Tensor) else lap_dist
        loss_meta = lap_dist + degree_penalty + alpha * loss_cls

        grads = torch.autograd.grad(loss_meta,[delta_H,delta_X])
        with torch.no_grad():
            # Proceed with original gradient ascent
            delta_H += eta_H * grads[0].sign()
            delta_X += eta_X * grads[1].sign()

            flat = delta_H.abs().flatten()
            topk = torch.topk(flat, k=min(delta_H.numel(), budget)).indices
            delta_H_new = torch.zeros_like(delta_H)
            delta_H_new.view(-1)[topk] = delta_H.view(-1)[topk]
            delta_H.copy_(delta_H_new)

        delta_X = delta_X.clamp(-epsilon, epsilon)
    return torch.clamp(H + delta_H, 0, 1), X + delta_X

# Meta Laplacian Attack (already defined)
def meta_laplacian_attack(H, X, y, data, budget=20, epsilon=0.05, T=20, eta_H=1e-2, eta_X=1e-2, alpha=4.0, reinit_if_stuck=True):
    device = X.device
    model = SimpleHGNN(X.shape[1], hidden_dim = args.hidden, out_dim = data["num_classes"],device = X.device).to(device)
    H = H.clone().detach()
    X = X.clone().detach()
    H.requires_grad = False
    X.requires_grad = False
    n, m = H.shape
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
    H_orig = torch.clamp(H, 0, 1)
    # de0 = H_orig.sum(dim=0).clamp(min=1e-6)
    # De0_inv = torch.diag(1.0 / de0)
    # dv0 = H_orig @ torch.ones((m,), device=device)
    # Dv0_inv_sqrt = torch.diag(1.0 / dv0.clamp(min=1e-6).sqrt())
    # L_orig = torch.eye(n, device=device) - Dv0_inv_sqrt @ H_orig @ De0_inv @ H_orig.t() @ Dv0_inv_sqrt
    L_orig = lap(H_orig)
    for t in tqdm(range(T)):
        H_pert = torch.clamp(H + delta_H, 0, 1)
        Z = model(X+delta_X, H_pert)
        # de = H_pert.sum(dim=0).clamp(min=1e-6)
        # De_inv = torch.diag(1.0 / de)
        # dv = H_pert @ torch.ones((m,), device=device)
        # Dv_inv_sqrt = torch.diag(1.0 / dv.clamp(min=1e-6).sqrt())
        # L_pert = torch.eye(n, device=device) - Dv_inv_sqrt @ H_pert @ De_inv @ H_pert.t() @ Dv_inv_sqrt
        L_pert = lap(H_pert)
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
        logits_adv = model(X + delta_X,H_pert)
        loss_cls = F.cross_entropy(logits_adv, y)
        # loss_cls = F.cross_entropy(logits_adv, model(H, X).argmax(dim=1))
        # lap_dist = (delta_L**2).sum()
        lap_dist = torch.norm(delta_L, p=2).mean()
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
    
    if draw:
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
        plt.plot(lap_dist_trajectory, label='Laplacian Loss')
        plt.plot(cls_loss_trajectory, label='Classification Loss')
        plt.plot(deg_penalty_trajectory, label='Degree Penalty')
        plt.title('Meta Loss Components over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(root,args.dataset+'_'+args.model+'_'+str(args.ptb_rate)+'_loss_components.png'))

        # Plot per-class accuracy over iterations
        class_acc_np = np.array(class_acc_trajectory)
        if class_acc_np.ndim == 2:
            plt.figure(figsize=(6, 4))
            for class_idx in range(class_acc_np.shape[1]):
                plt.plot(class_acc_np[:, class_idx], label=f'Class {class_idx}')
            plt.title('Per-Class Accuracy over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(root,args.dataset+'_'+args.model+'_'+str(args.ptb_rate)+'_class_acc.png'))

    return torch.clamp(H + delta_H, 0, 1), X + delta_X


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    data, X, y, train_mask, val_mask, test_mask = get_dataset(args, device)
    print(f"length of train is : {sum(train_mask)}, length of val is: {sum(val_mask)},length of test is: {sum(test_mask)}")
    
    HG = Hypergraph(data["num_vertices"], data["edge_list"],device = device)
    H = HG.H.to_dense().to(device)
    W_e = HG.W_e.to_dense()
    print('W_e stats: ',W_e.min(), W_e.mean(), W_e.std(), W_e.max())
    print('# 0-degree nodes: ',(HG.H.sum(1).to_dense() == 0).sum(),' #0-degree hyperedges: ',(HG.H.sum(0).to_dense() == 0).sum())
    perturbations = int(args.ptb_rate * (HG.num_e))
    print('#hyperedges to perturb: ',perturbations)


    # H, X, y = generate_synthetic_hypergraph()
    # H,X,y = H.to(device),X.to(device),y.to(device)
    use_bn = False
    if args.model == 'hypergcn':
        model = HyperGCN(X.shape[1], args.hidden, data["num_classes"],use_bn=use_bn).to(device)
    elif args.model == 'hgnn':
        # from models import HGNN
        # gcn = HGNN(in_ch=X.shape[1],n_class=data["num_classes"],n_hid=args.hidden,dropout=args.dropout).to(device)
        model = HGNN_modified(X.shape[1], args.hidden, data["num_classes"],use_bn=use_bn).to(device)
    elif args.model == 'hgnnP':
        model = HGNNP(X.shape[1], args.hidden, data["num_classes"],use_bn=use_bn).to(device)
    elif args.model == 'hnhn':
        use_H = True
        if use_H:
            model = HNHN(X.shape[1], args.hidden, data["num_classes"],use_bn=use_bn, use_H = True).to(device)
        else:
            model = HNHN(X.shape[1], args.hidden, data["num_classes"],use_bn=use_bn, use_H = False).to(device)
    else:
        model = SimpleHGNN(X.shape[1],hidden_dim = args.hidden, out_dim = data["num_classes"],device = X.device).to(device)
        # model = SimpleHGNN(X.shape[1],out_dim = 3).to(device)
    train_model(args,model,H,X,y)
    
    Z_orig = model(X, H).detach()
    # H_adv, X_adv = \
    #     meta_laplacian_attack(H, X, y, data, budget=perturbations, \
    #                           epsilon=args.epsilon, T=args.attack_iters, \
    #                         eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.alpha, \
    #                         reinit_if_stuck=True)

    H_adv, X_adv = meta_laplacian_pois_attack(H, X, y, data, surrogate_class=None, target_model= model, budget=perturbations, \
                            epsilon=args.epsilon, T=args.attack_iters, \
                        eta_H=args.eta_H, eta_X=args.eta_X, alpha=args.alpha, \
                        reinit_if_stuck=True)

    Z_adv = model(X_adv,H_adv).detach()

    print("Laplacian Frobenius norm change:", laplacian_diff(H, H_adv))
    print("Embedding shift (ΔZ Fro norm):", embedding_shift(Z_orig, Z_adv))
    h_l0, x_linf, deg_shift_l1, edge_card_shift_l1,deg_shift_l2, edge_card_shift_l2, deg_shift_linf, edge_card_shift_linf = measure_stealthiness(H, H_adv, X, X_adv)
    print("Structural L0 perturbation:", h_l0)
    print("Feature L-infinity perturbation:", x_linf)
    print("Total shift in degree distribution (Linf):", deg_shift_linf)
    print("Total shift in degree distribution (L1):", deg_shift_l1)
    print("Total shift in degree distribution (L2):", deg_shift_l2)
    print("Total shift in edge-cardinality distribution (Linf):", edge_card_shift_linf)
    print("Total shift in edge-cardinality distribution (L1):", edge_card_shift_l1)
    print("Total shift in edge-cardinality distribution (L2):", edge_card_shift_l2)

    print("Semantic change in features (1 - avg. cosine):", semantic_feature_change(X, X_adv))
    print("Embedding sensitivity vs node degree (Pearson r):", degree_sensitivity(H, Z_orig, Z_adv))
    acc_orig, acc_adv, acc_drop = classification_drop(model, H, X, H_adv, X_adv, y)
    print("Classification accuracy before attack:", acc_orig)
    print("Classification accuracy after attack:", acc_adv)
    print("Accuracy drop due to attack:", acc_drop*100,'%')
    # print("Accuracy drop due to attack:", max(acc_drop, 0.0))
    if draw:
        visualize_tsne(args,Z_orig, Z_adv, title="t-SNE: Embedding Drift Due to Meta-Laplacian Attack")
