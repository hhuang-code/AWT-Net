# Lines 27 to 31, lines 44 to 48, lines 63 to 89 and lines 161 to 187 are referred from: https://github.com/mutianxu/GDANet/blob/main/model/GDANet_cls.py:

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../classification')

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_util import knn, local_operator, local_operator_withnorm, index_points

from lifting import LevelDAWN
from transformer import TransformerEncoder
from wavelet_util import batched_index_select

import pdb


class AWTNet(nn.Module):
    def __init__(self, opts, num_classes):
        super(AWTNet, self).__init__()

        self.opts = opts

        # scale 1 --------------------------------------------------------------------------
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv1 = nn.Sequential(nn.Conv2d(9, 64, kernel_size=1, bias=True), self.bn1)

        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True), self.bn11)

        self.lifting1 = LevelDAWN(64, regu_details=0.1, regu_approx=0.1, dropout=self.opts.dropout[0])

        self.tfm_c1 = TransformerEncoder(key_size=64, query_size=64, value_size=64, num_hiddens=64, norm_shape=[64],
                                         ffn_num_input=64, ffn_num_hiddens=64, num_heads=2, num_layers=2, dropout=self.opts.dropout[0])
        self.tfm_d1 = TransformerEncoder(key_size=64, query_size=64, value_size=64, num_hiddens=64, norm_shape=[64],
                                         ffn_num_input=64, ffn_num_hiddens=64, num_heads=2, num_layers=2, dropout=self.opts.dropout[0])

        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)
        self.conv12 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True), self.bn12)

        # scale 2 ----------------------------------------------------------------
        self.bn2 = nn.BatchNorm2d(64 + 64, momentum=0.1)
        self.conv2 = nn.Sequential(nn.Conv2d((64 + 64) * 2, 64 + 64, kernel_size=1, bias=True, groups=2), self.bn2)

        self.bn20 = nn.BatchNorm2d(64 + 64, momentum=0.1)
        self.conv20 = nn.Sequential(nn.Conv2d(64 + 64, 64 + 64, kernel_size=1, bias=True, groups=2), self.bn20)

        self.bn21 = nn.BatchNorm2d(64 + 64, momentum=0.1)
        self.conv21 = nn.Sequential(nn.Conv2d(64 + 64, 64 + 64, kernel_size=1, bias=True, groups=2), self.bn21)

        self.lifting2 = LevelDAWN(64, regu_details=0.1, regu_approx=0.1, dropout=self.opts.dropout[0])

        self.tfm_c2 = TransformerEncoder(key_size=64, query_size=64, value_size=64, num_hiddens=64, norm_shape=[64],
                                         ffn_num_input=64, ffn_num_hiddens=64, num_heads=2, num_layers=2, dropout=self.opts.dropout[0])
        self.tfm_d2 = TransformerEncoder(key_size=64, query_size=64, value_size=64, num_hiddens=64, norm_shape=[64],
                                         ffn_num_input=64, ffn_num_hiddens=64, num_heads=2, num_layers=2, dropout=self.opts.dropout[0])

        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)
        self.conv22 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True), self.bn22)

        # tailing ----------------------------------------------------------------
        self.bnt = nn.BatchNorm2d(128, momentum=0.1)
        self.convt = nn.Sequential(nn.Conv2d((64 + 64 + 64) * 2, 128, kernel_size=1, bias=True), self.bnt)

        self.bnt1 = nn.BatchNorm2d(128, momentum=0.1)
        self.convt1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True), self.bnt1)

        self.bnt2 = nn.BatchNorm1d(128, momentum=0.1)
        self.convt2 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True), self.bnt2)

        self.bnc = nn.BatchNorm1d(512, momentum=0.1)
        self.convc = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True), self.bnc)

        self.bncls = nn.BatchNorm1d(64, momentum=0.1)
        self.convcls = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=True), self.bncls)

        self.bn5 = nn.BatchNorm1d(256, momentum=0.1)
        self.conv5 = nn.Sequential(nn.Conv1d(256 + 512 + 64, 256, kernel_size=1, bias=True), self.bn5)
        self.dp1 = nn.Dropout(self.opts.dropout[1])

        self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=True), self.bn6)
        self.dp2 = nn.Dropout(self.opts.dropout[1])

        self.bn7 = nn.BatchNorm1d(128, momentum=0.1)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=True), self.bn7)

        self.conv8 = nn.Conv1d(128, num_classes, kernel_size=1, bias=True)

    def forward(self, x, norm_plt, cls_label):
        """
        :param x: (b, 3, n_pts)
        :param norm_plt: (b, 3, n_pts)
        :param cls_label: (b, n_parts)
        :return:
        """
        bz, c, n_pts = x.size()

        # scale 1 -----------------------------------------------------------------------------
        knn_idx, _ = knn(x, k=32)
        x1 = local_operator_withnorm(x, knn_idx, norm_plt, k=32)  # (b, 9, n_pts, k)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1) + x1)  # (b, c, n_pts, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]   # (b, c, n_pts)

        (c1, d1), r1, (odd_idx, even_idx), adj_uw, p_dist = self.lifting1(x1)  # c1, d1: (b, c, n_pts / 2)
        d1 = torch.abs(d1)

        # for numerical stability
        d1_max, _ = torch.max(d1, dim=1, keepdim=True)
        d1 = d1 - d1_max.detach()

        d1 = batched_index_select(x1, 2, odd_idx) * torch.exp(d1)

        zd1 = self.tfm_d1(x1, d1, batched_index_select(adj_uw, dim=2, index=odd_idx))  # same as x1
        zc1 = self.tfm_c1(x1, c1, batched_index_select(adj_uw, dim=2, index=even_idx))

        z1 = torch.cat([zd1, zc1], 1)   # (b, c, n_pts)
        z1 = F.relu(self.conv12(z1))

        # scale 2 -----------------------------------------------------------------------------
        odd_even_dist = batched_index_select(batched_index_select(p_dist, 1, odd_idx), 2, even_idx)
        odd_even_dist, idx = odd_even_dist.sort(dim=-1, descending=False)  # (b, n_pts / 2, n_pts / 2)
        odd_even_dist, idx = odd_even_dist[:, :, :3], idx[:, :, :3]

        odd_even_dist_recip = 1.0 / (odd_even_dist + 1e-8)
        norm = torch.sum(odd_even_dist_recip, dim=2, keepdim=True)
        weight = odd_even_dist_recip / norm
        interp_c1 = torch.sum(index_points(c1.transpose(1, 2), idx) * weight.view(bz, n_pts // 2, 3, 1), dim=2).transpose(1, 2) # (b, n_pts / 2, c)
        up_c1 = batched_index_select(torch.cat([c1, interp_c1], dim=2), 2, torch.cat([even_idx, odd_idx], dim=1).sort(dim=1)[1])

        xz1 = torch.cat([up_c1, z1], dim=1)
        x2 = local_operator(xz1, knn_idx, k=32)
        x2 = x2.split(x2.shape[1] // 4, dim=1)
        x2 = torch.cat([x2[0], x2[2], x2[1], x2[3]], dim=1)

        x2_id = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv20(x2_id))
        x2 = F.relu(self.conv21(x2) + x2_id)  # (b, c, n_pts, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]  # (b, c, n_pts)

        loc_c1, loc_z1 = x2.split(x2.shape[1] // 2, dim=1)

        (c2, d2), r2, (odd_idx, even_idx), adj_uw, _ = self.lifting2(loc_c1)  # c, d: (b, c, n_pts)
        d2 = torch.abs(d2)

        # for numerical stability
        d2_max, _ = torch.max(d2, dim=1, keepdim=True)
        d2 = d2 - d2_max.detach()

        d2 = batched_index_select(loc_c1, 2, odd_idx) * torch.exp(d2)

        zd2 = self.tfm_d2(loc_z1, d2, batched_index_select(adj_uw, dim=2, index=odd_idx))  # same as x1
        zc2 = self.tfm_c2(loc_z1, c2, batched_index_select(adj_uw, dim=2, index=even_idx))

        z2 = torch.cat([zd2, zc2], 1)  # (b, c, n_pts)
        z2 = F.relu(self.conv22(z2))

        # tailing ----------------------------------------------------------------------
        xzt = torch.cat([xz1, z2], dim=1)
        xt = local_operator(xzt, knn_idx, k=32)
        xt = F.relu(self.convt(xt))
        xt = F.relu(self.convt1(xt) + xt)
        xt = xt.max(dim=-1, keepdim=False)[0]   # (b, c, n_pts)
        zt = F.relu(self.convt2(xt))

        ###############
        xx = torch.cat((z1, z2, zt), dim=1)

        xc = F.relu(self.convc(xx))
        xc = F.adaptive_max_pool1d(xc, 1).view(bz, -1)

        cls_label = cls_label.view(bz, 16, 1)
        cls_label = F.relu(self.convcls(cls_label))
        cls = torch.cat((xc.view(bz, 512, 1), cls_label), dim=1)
        cls = cls.repeat(1, 1, n_pts)

        x = torch.cat((xx, cls), dim=1)
        x = F.relu(self.conv5(x))
        x = self.dp1(x)
        x = F.relu(self.conv6(x))
        x = self.dp2(x)
        x = F.relu(self.conv7(x))
        x = self.conv8(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1).contiguous()  # (b, n_pts, n_parts)

        return x, r1 + r2
