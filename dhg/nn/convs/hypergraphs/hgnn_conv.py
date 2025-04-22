import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.utils.sparse import sparse_dropout,dense_dropout
from torch.sparse import mm as sparse_mm
from dhg.structure.hypergraphs import Hypergraph

class HGNNConv(nn.Module):
    r"""The HGNN convolution layer proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} 
        \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).

    where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{H}` is the hypergraph incidence matrix, 
    :math:`\mathbf{W}_e` is a diagonal hyperedge weight matrix, :math:`\mathbf{D}_v` is a diagonal vertex degree matrix, 
    :math:`\mathbf{D}_e` is a diagonal hyperedge degree matrix, :math:`\mathbf{\Theta}` is the learnable parameters.

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.theta(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X

class HGNNConv2(nn.Module):
    r"""The HGNN convolution layer proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} 
        \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).

    where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{H}` is the hypergraph incidence matrix, 
    :math:`\mathbf{W}_e` is a diagonal hyperedge weight matrix, :math:`\mathbf{D}_v` is a diagonal vertex degree matrix, 
    :math:`\mathbf{D}_e` is a diagonal hyperedge degree matrix, :math:`\mathbf{\Theta}` is the learnable parameters.

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        # self.drop_rate = drop_rate
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def smoothing_with_HGNN(self, X: torch.Tensor, H: torch.Tensor, laplacian_drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
    """
        # if self.device != X.device:
        #     X = X.to(self.device)
        # D_v_inv_sqrt_values = torch.pow(H.sum(dim=1).to_dense(), -0.5)
        # D_v_inv_sqrt_indices = torch.arange(0, H.size(0)).to(H.device)
        # D_v_neg_1_2 = torch.sparse_coo_tensor(
        #     torch.stack([D_v_inv_sqrt_indices, D_v_inv_sqrt_indices]),
        #     D_v_inv_sqrt_values,
        #     (H.size(0), H.size(0))
        # )
        D_v_neg_1_2 = torch.pow(H.sum(dim=1),-0.5)
        D_v_neg_1_2 = torch.nan_to_num(D_v_neg_1_2)
        H_t = H.transpose(0, 1)  # E x N       
        # W_e = torch.eye(H_t.size(0)).to_sparse().to(H.device)
        W_e = torch.eye(H_t.size(0)).to(H.device)


        # D_e_inv_values = torch.pow(H_t.sum(dim=1).to_dense(), -1)
        # D_e_inv_indices = torch.arange(0, H_t.size(0)).to(H_t.device)
        # D_e_neg_1 = torch.sparse_coo_tensor(
        #     torch.stack([D_e_inv_indices, D_e_inv_indices]),
        #     D_e_inv_values,
        #     (H_t.size(0), H_t.size(0))
        # )
        D_e_neg_1 = torch.pow(H_t.sum(dim=1), -1)
        D_e_neg_1=torch.nan_to_num(D_e_neg_1)
        # H_normalized = sparse_mm(D_v_neg_1_2, sparse_mm(H, W_e))
        # L_HGNN = sparse_mm(H_normalized, sparse_mm(D_e_neg_1, sparse_mm(H_t, D_v_neg_1_2)))  # N x in_features
        # print(D_v_neg_1_2.shape, H.shape, W_e.shape, D_e_neg_1.shape, H_t.shape)
        L_HGNN = torch.diag(D_v_neg_1_2).mm(H).mm(W_e).mm(torch.diag(D_e_neg_1)).mm(H_t).mm(torch.diag(D_v_neg_1_2))
        # if self.drop_rate > 0.0:
        #     L_HGNN = sparse_dropout(L_HGNN, self.drop_rate)
        if laplacian_drop_rate > 0.0:
            L_HGNN = dense_dropout(L_HGNN, self.drop_rate)
        # return sparse_mm(L_HGNN,X)
        return L_HGNN.mm(X)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.theta(X)
        X = self.smoothing_with_HGNN(X,H)

        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X
