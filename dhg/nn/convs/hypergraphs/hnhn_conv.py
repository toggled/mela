import torch
import torch.nn as nn
from typing import Optional
from dhg.structure.hypergraphs import Hypergraph
from dhg.utils.sparse import sparse_dropout

class HNHNConv(nn.Module):
    r"""The HNHN convolution layer proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

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
        self.theta_v2e = nn.Linear(in_channels, out_channels, bias=bias)
        self.theta_e2v = nn.Linear(out_channels, out_channels, bias=bias)

    # def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
    def forward(self, X: torch.Tensor, hg: torch.Tensor) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        # v -> e
        X = self.theta_v2e(X)
        Y = self.act(hg.v2e(X, aggr="mean"))
        # e -> v
        Y = self.theta_e2v(Y)
        X = hg.e2v(Y, aggr="mean")
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X


class HNHNConv2(nn.Module):
    r"""The HNHN convolution layer proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

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
        self.theta_v2e = nn.Linear(in_channels, out_channels, bias=bias)
        self.theta_e2v = nn.Linear(out_channels, out_channels, bias=bias)

    def v2e(
        self,
        H: torch.Tensor,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ) -> torch.Tensor:
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyperedges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        H_T = H.t()
        D_e_neg_1 = torch.pow(H_T.sum(dim=1), -1)
        # Message aggregation
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(H_T, drop_rate)
            else:
                P = H_T
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(torch.diag(D_e_neg_1), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            raise ValueError("not implemented")
            # assert v2e_weight.shape[0] == self.v2e_weight.shape[0], \
            #     "The size of v2e_weight must be equal to the size of self.v2e_weight."
            # P = torch.sparse_coo_tensor(self.H_T._indices(), v2e_weight, self.H_T.shape, device=self.device)
            # if drop_rate > 0.0:
            #     P = sparse_dropout(P, drop_rate)
            # if aggr == "mean":
            #     X = torch.sparse.mm(P, X)
            #     D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
            #     D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
            #     X = D_e_neg_1 * X
            # elif aggr == "sum":
            #     X = torch.sparse.mm(P, X)
            # elif aggr == "softmax_then_sum":
            #     P = torch.sparse.softmax(P, dim=1)
            #     X = torch.sparse.mm(P, X)
            # else:
            #     raise ValueError(f"Unknown aggregation method {aggr}.")

        return X
    def e2v(
        self,
        H: torch.Tensor,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ) -> torch.Tensor:
        """
        Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Args:
            X (torch.Tensor): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            aggr (str): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            e2v_weight (Optional[torch.Tensor]): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            v_weight (Optional[torch.Tensor]): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            drop_rate (float): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.

        Returns:
            torch.Tensor: Updated vertex feature matrix after message passing.
        """

        # Handle dropout
        if drop_rate > 0.0:
            P = sparse_dropout(H, drop_rate)
        else:
            P = H

        # Check if e2v_weight is provided
        if e2v_weight is None:
            # Initialize message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                # D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                # D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                D_v_neg_1 = torch.pow(P.sum(dim=1),-1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                # print('X.shape: ',X.shape, ' - ', D_v_neg_1.shape)
                X = D_v_neg_1.view(-1, 1) * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            raise ValueError("not implemented")
        
        return X
    
    # def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
    def forward(self, X: torch.Tensor, hg: torch.Tensor) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        # v -> e
        X = self.theta_v2e(X)
        # Y = self.act(hg.v2e(X, aggr="mean"))
        # print(hg.shape,X.shape)
        Y = self.act(self.v2e(hg,X,aggr="mean"))
        # print('Y.shape: ',Y.shape)
        # e -> v
        Y = self.theta_e2v(Y)
        # X = hg.e2v(Y, aggr="mean")
        X = self.e2v(hg,Y, aggr="mean")
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X
