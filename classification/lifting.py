import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from wavelet_util import build_graph, even_odd_split, batched_index_select

import pdb


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, adapted from https://github.com/tkipf/gcn, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input, adj):
        """
        :param input: (b, n_pts, c)
        :param adj: (b, n_pts, n_pts), row represents neighbors
        :return: (b, n_pts, c)
        """
        support = self.layer(input)
        output = torch.bmm(adj, support)

        return output


class GCN(nn.Module):
    def __init__(self, nin, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nin, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

        self.bn1 = nn.BatchNorm1d(nhid, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(nout, momentum=0.1)

    def forward(self, x, adj1, adj2):
        """
        :param x: (b, c, n_pts)
        :param adj: (b, n_pts, n_pts), row represents neighbors
        :return: (b, c, n_pts)
        """
        x = x.transpose(2, 1).contiguous()   # (b, n_pts, c)

        x = F.relu(self.bn1(self.gc1(x, adj1).transpose(2, 1).contiguous())).transpose(2, 1).contiguous()
        x = F.dropout(x, self.dropout)
        x = self.bn2(self.gc2(x, adj2).transpose(2, 1).contiguous())

        return x


class LiftingScheme(nn.Module):
    def __init__(self, in_planes, modified=True, splitting=True, dropout=0.2):
        super(LiftingScheme, self).__init__()
        self.in_planes = in_planes
        self.modified = modified
        self.splitting = splitting
        self.dropout = dropout

        self.P = GCN(self.in_planes, self.in_planes * 2, self.in_planes, dropout=self.dropout)
        self.U = GCN(self.in_planes, self.in_planes * 2, self.in_planes, dropout=self.dropout)

    def forward(self, x):
        """
        :param x: (b, c, n_pts)
        :return: adj_x: (b, n_pts, n_pts), c / d: (b, c, n_pts / 2), odd_idx / even_idx: (b, n_pts / 2)
        """
        if self.splitting:  # True
            adj_uw, adj_w, p_dist, _, split_output = even_odd_split(x, k=64, split_adj=True)

            odd_idx, even_idx = split_output['idx'][0], split_output['idx'][1]
            x_odd, x_even = split_output['nodes'][0], split_output['nodes'][1]
            split_adj_w = split_output['adj'][1]
        else:
            raise NotImplementedError

        odd_adj_w, even_adj_w, odd_even_adj_w, even_odd_adj_w = split_adj_w

        if self.modified:  # True
            c = x_even + self.U(x_odd, odd_adj_w, even_odd_adj_w)  # coarse
            d = x_odd - torch.tanh(self.P(c, even_adj_w, odd_even_adj_w))  # detail

            return (adj_uw, adj_w, p_dist), (c, d), (odd_idx, even_idx)  # same shape as x_even and x_odd
        else:
            # not used and tested
            d = x_odd - self.P(x_even, odd_even_adj_w)
            c = x_even + torch.tanh(self.U(d, even_odd_adj_w))

            return (adj_uw, adj_w, p_dist), (c, d), (odd_idx, even_idx)


class LevelDAWN(nn.Module):
    def __init__(self, in_planes, regu_details, regu_approx, dropout=0.2):
        super(LevelDAWN, self).__init__()
        self.in_planes = in_planes
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.dropout = dropout

        self.wavelet = LiftingScheme(self.in_planes, modified=True, splitting=True, dropout=self.dropout)

    def forward(self, x):
        """
        x: (b, c, n_pts)
        x_pt: (b, 3, n_pts)
        """

        (adj_uw, adj_w, p_dist), (c, d), (odd_idx, even_idx) = self.wavelet(x)   # adj: (b, n_pts, n_pts); c, d: (b, c, n_pts / 2)

        if self.regu_approx != 0.0 and self.regu_details != 0.0:
            # constraint on the details
            if self.regu_details:
                rd = self.regu_details * F.smooth_l1_loss(d, torch.zeros_like(d), beta=1.0)
            else:
                rd = 0

            # constrain on the approximation
            if self.regu_approx:
                x_bar = torch.bmm(x, adj_w.transpose(2, 1).contiguous())  # (b, c, n_pts)
                rc = self.regu_approx * F.smooth_l1_loss(c, batched_index_select(x_bar, 2, even_idx), beta=1.0)
            else:
                rc = 0

            if self.regu_approx == 0.0:
                # only the details
                r = rd
            elif self.regu_details == 0.0:
                # only the approximation
                r = rc
            else:
                # both
                r = rd + rc
        else:
            r = 0

        return (c, d), r, (odd_idx, even_idx), adj_uw, p_dist

