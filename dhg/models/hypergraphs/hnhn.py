import torch
import torch.nn as nn

import dhg
from dhg.nn import HNHNConv,HNHNConv2


class HNHN(nn.Module):
    r"""The HNHN model proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        use_H = False,
    ) -> None:
        super().__init__()
        self.use_H = use_H
        self.layers = nn.ModuleList()
        if use_H:
            self.layers.append(
            HNHNConv2(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
            )
            self.layers.append(
                HNHNConv2(hid_channels, num_classes, use_bn=use_bn, is_last=True)
            )

        else:            
            self.layers.append(
                HNHNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
            )
            self.layers.append(
                HNHNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
            )
    def forward(self, X, hg):
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X
