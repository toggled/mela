import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm 
import os
from dhg.data import Cooking200, CoauthorshipCora,CocitationCora,CocitationCiteseer,CoauthorshipDBLP, CocitationPubmed,\
                     Tencent2k, News20,YelpRestaurant, WalmartTrips,HouseCommittees, Yelp3k
from dhg.utils import split_by_ratio
# Synthetic hypergraph generator
# def generate_synthetic_hypergraph(n_nodes=200, n_edges=100, d=10):
#     H = torch.zeros((n_nodes, n_edges))
#     for e in range(n_edges):
#         nodes = np.random.choice(n_nodes, size=np.random.randint(2, min(6, n_nodes)), replace=False)
#         H[nodes, e] = 1
#     X = torch.randn(n_nodes, d)
#     labels = torch.randint(0, 3, (n_nodes,))
#     return H.float(), X.float(), labels

# Evaluation helpers
# Compute the Frobenius norm of the embedding shift
# This measures how much the learned node representations change under the attack
def embedding_shift(Z1, Z2):
    return torch.norm(Z1 - Z2).item()

def lap(H):
    de = H.sum(dim=0).clamp(min=1e-6)
    De_inv = torch.diag(1.0 / de)
    dv = H @ torch.ones(H.shape[1], device=H.device)
    Dv_inv_sqrt = torch.diag(1.0 / dv.clamp(min=1e-6).sqrt())
    return torch.eye(H.shape[0], device=H.device) - Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt
# Measure the Frobenius norm difference between Laplacians of original and perturbed incidence matrices
# This evaluates the impact of perturbations on the hypergraph structure
def laplacian_diff(H1, H2):
    return torch.norm(lap(H1) - lap(H2)).item()

# Visualize original and adversarial embeddings using t-SNE for interpretability
# Helps detect how much the attack shifted the embedding geometry
def visualize_tsne(args, Z1, Z2, title):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    Z = torch.cat([Z1, Z2], dim=0).cpu().numpy()
    tsne = TSNE(n_components=2)
    Z_2d = tsne.fit_transform(Z)
    plt.scatter(Z_2d[:Z1.shape[0], 0], Z_2d[:Z1.shape[0], 1], c='k', label='original', alpha=0.6, s = 2)
    plt.scatter(Z_2d[Z1.shape[0]:, 0], Z_2d[Z1.shape[0]:, 1], c='red', label='adversarial', alpha=0.6, s = 2)
    plt.legend()
    plt.title(title)
    # plt.show()
    plt.savefig(args.dataset+'_'+args.model+'_'+str(args.ptb_rate)+'_tsne.png')

# Quantify how stealthy the attack is based on L0 structural change and L∞ feature deviation
def measure_stealthiness(H, H_adv, X, X_adv):
    h_l0 = torch.sum((H - H_adv).abs() > 1e-6).item()
    x_delta = torch.norm((X - X_adv), p=float('inf')).item()

    degree_orig = H.sum(dim=1)
    degree_adv = H_adv.sum(dim=1)
    deg_shift_inf = torch.norm(degree_orig - degree_adv, p=float('inf')).item()
    deg_shift_l1 = torch.norm(degree_orig - degree_adv, p=1).item()
    deg_shift_l2 = torch.norm(degree_orig - degree_adv, p=2).item()

    edge_card_orig = H.sum(dim=0)
    edge_card_adv = H_adv.sum(dim=0)
    edge_card_shift_inf = torch.norm(edge_card_orig - edge_card_adv, p=float('inf')).item()
    edge_card_shift_l1 = torch.norm(edge_card_orig - edge_card_adv, p=1).item()
    edge_card_shift_l2 = torch.norm(edge_card_orig - edge_card_adv, p=2).item()

    return h_l0, x_delta, deg_shift_l1, edge_card_shift_l1,deg_shift_l2, edge_card_shift_l2, deg_shift_inf, edge_card_shift_inf

# Evaluate how well the attack generalizes across different models
# Returns the output logits of each model when run on the adversarially perturbed inputs
def evaluate_transferability(H_adv, X_adv, model_list):
    return [model(X_adv,H_adv).detach() for model in model_list]

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
@torch.no_grad
def classification_drop(model, H, X, H_adv, X_adv, labels):
    logits_orig = model(X, H)
    logits_adv = model(X_adv,H_adv)
    acc_orig = (logits_orig.argmax(dim=1) == labels).float().mean().item()
    acc_adv = (logits_adv.argmax(dim=1) == labels).float().mean().item()
    return acc_orig, acc_adv, (acc_orig - acc_adv)/acc_orig

@torch.no_grad
def classification_drop_pois(model, model_pois, H, X, H_adv, X_adv, labels):
    logits_orig = model(X, H)
    logits_adv = model_pois(X_adv,H_adv)
    acc_orig = (logits_orig.argmax(dim=1) == labels).float().mean().item()
    acc_adv = (logits_adv.argmax(dim=1) == labels).float().mean().item()
    return acc_orig, acc_adv, (acc_orig - acc_adv)/acc_orig


def train_model(args, model, H, X, y):
    print('---- Model Training -----')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs
    num_epochs = args.num_epochs
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(X, H)
        loss = criterion(logits, y)  # assuming y is your target labels
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"Epoch {epoch}: Loss = {loss.item()}, Accuracy = {acc * 100}%")

def get_dataset(args, device):
    if args.dataset == 'co-citeseer': #cocitation-citeseer
        data = CocitationCiteseer()
    if args.dataset == 'coauth_cora':
        data = CoauthorshipCora()
    if args.dataset == 'coauth_dblp':
        data = CoauthorshipDBLP()
    if args.dataset == 'co-cora': # cocitation_cora
        data = CocitationCora()
    if args.dataset == 'co-pubmed': #cocitation-pubmed
        data = CocitationPubmed()
    if args.dataset == 'yelp':
        data = YelpRestaurant()
    if args.dataset == 'yelp3k':
        data = Yelp3k()
        train_ratio, val_ratio, test_ratio = 0.1, 0.1, 0.8
        num_v = data["labels"].shape[0]
        train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
        # train_mask = train_mask.to(device)
        # val_mask = val_mask.to(device)
        # test_mask = test_mask.to(device)
        val_mask = test_mask | val_mask
        val_mask = val_mask.to(device)
        test_mask = val_mask.to(device)
        data["train_mask"] = train_mask
        data["val_mask"] = val_mask
        data["test_mask"] = test_mask

    if args.dataset == "cooking":
        data = Cooking200()
        train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
        num_v = data["labels"].shape[0]
        train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
        # data["train_mask"] = train_mask
        # data["val_mask"] = val_mask
        # data["test_mask"] = test_mask
        val_mask = test_mask | val_mask
        val_mask = val_mask.to(device)
        test_mask = val_mask.to(device)
        data["train_mask"] = train_mask
        data["val_mask"] = val_mask
        data["test_mask"] = test_mask
    if args.dataset == 'tencent2k':
        data = Tencent2k()
        # random.seed(1)
        # train_ratio, val_ratio, test_ratio = 0.5, 0.1, 0.4
        # num_v = data["labels"].shape[0]
        # train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
        # train_mask = train_mask.to(device)
        # val_mask = test_mask | val_mask
        # val_mask = val_mask.to(device)
        # test_mask = val_mask.to(device)
        # data["train_mask"] = train_mask
        # data["val_mask"] = val_mask
        # data["test_mask"] = test_mask
    if args.dataset == 'walmart':
        data = WalmartTrips()
        train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
        num_v = data["labels"].shape[0]
        train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
    if args.dataset == 'house':
        data = HouseCommittees()
        train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
        num_v = data["labels"].shape[0]
        train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
    if args.dataset == 'news20':
        data = News20()
        num_v = data["labels"].shape[0]
        train_ratio, val_ratio, test_ratio = 0.1, 0.1, 0.8
        train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
        # train_mask = train_mask.to(device)
        # val_mask = val_mask.to(device)
        # test_mask = test_mask.to(device)
        val_mask = test_mask | val_mask
        val_mask = val_mask.to(device)
        test_mask = val_mask.to(device)
        data["train_mask"] = train_mask
        data["val_mask"] = val_mask
        data["test_mask"] = test_mask

    labels = data["labels"].to(device)

    if 'features' not in data.content:
        print(args.dataset,' does not have features.')
        features = torch.ones(data["num_vertices"],1).to(device)
    else:
        features = data["features"].to(device)
    if "train_mask" in data.content:
        train_mask = data["train_mask"].to(device)
    if "val_mask" in data.content:
        val_mask = data["val_mask"].to(device)
    if "test_mask" in data.content:
        test_mask = data["test_mask"].to(device)
    return data, features, labels, train_mask, val_mask, test_mask

