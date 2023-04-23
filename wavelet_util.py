import torch

from model_util import knn

import pdb


def batched_index_select(input, dim, index):
    """
    :param input: (B, *, ..., *)
    :param dim: 0 < scalar
    :param index: (B, M)
    :return:
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    return torch.gather(input, dim, index)


def build_graph(pts, k, sigma=2):
    """
    :param pts: (b, c, n_pts)
    :param k: a scalar
    :param sigma: a scalar
    :return:
    """
    bz, _, n_pts = pts.shape

    idx, p_dist = knn(pts, k=k)

    p_dist = p_dist / (sigma * sigma)
    w = torch.exp(torch.neg(p_dist))

    adj_uw = torch.zeros(bz, n_pts, n_pts, device=pts.device)
    adj_uw.scatter_(dim=-1, index=idx, value=1)

    adj_w = adj_uw * w

    row_sum = 1 / (torch.sum(adj_w, dim=-1, keepdim=True) + 1e-8)

    adj_w = adj_w * row_sum

    return adj_uw, adj_w, p_dist    # unweighted, weighted


def split_helper(nodes, adj_uw, adj_w, labels, split_adj=True):
    """
    :param nodes: (b, c, n_pts)
    :param adj: (b, n_pts, n_pts)
    :param labels: 0 - even, 1 - odd
    :return:
    """
    bz, c, n_nodes = nodes.shape

    # rearrange nodes
    odd_idx = torch.stack((labels == 1).nonzero(as_tuple=False)[:, -1].split(n_nodes // 2), dim=0)   # (b, n_nodes / 2)
    even_idx = torch.stack((labels == 0).nonzero(as_tuple=False)[:, -1].split(n_nodes // 2), dim=0)

    odd_nodes = batched_index_select(nodes, 2, odd_idx)     # (b, c, n_pts / 2)
    even_nodes = batched_index_select(nodes, 2, even_idx)

    if split_adj:
        # select sub-adjacent matrix
        odd_adj_uw = batched_index_select(batched_index_select(adj_uw, 1, odd_idx), 2, odd_idx)
        even_adj_uw = batched_index_select(batched_index_select(adj_uw, 1, even_idx), 2, even_idx)
        odd_even_adj_uw = batched_index_select(batched_index_select(adj_uw, 1, odd_idx), 2, even_idx)
        even_odd_adj_uw = batched_index_select(batched_index_select(adj_uw, 1, even_idx), 2, odd_idx)

        odd_adj_w = batched_index_select(batched_index_select(adj_w, 1, odd_idx), 2, odd_idx)
        even_adj_w = batched_index_select(batched_index_select(adj_w, 1, even_idx), 2, even_idx)
        odd_even_adj_w = batched_index_select(batched_index_select(adj_w, 1, odd_idx), 2, even_idx)
        even_odd_adj_w = batched_index_select(batched_index_select(adj_w, 1, even_idx), 2, odd_idx)

        split_adj_uw = (odd_adj_uw, even_adj_uw, odd_even_adj_uw, even_odd_adj_uw)
        split_adj_w = (odd_adj_w, even_adj_w, odd_even_adj_w, even_odd_adj_w)
    else:
        # odd_adj, even_adj, odd_even_adj, even_odd_adj = None, None, None, None
        split_adj_uw = (None, None, None, None)
        split_adj_w = (None, None, None, None)

    return {'idx': (odd_idx, even_idx), 'nodes': (odd_nodes, even_nodes), 'adj': (split_adj_uw, split_adj_w)}


def even_odd_split(x, k=64, split_adj=False):
    """
    :param x: (b, c, n_pts)
    :return:
    """
    adj_uw, adj_w, p_dist = build_graph(x, k)

    bz, c, n_nodes = x.shape
    # randomly assign initial label to each node, -1 - even, 1 - odd
    labels = torch.randint(0, 2, size=(bz, n_nodes), device=x.device)
    labels[labels == 0] = -1

    # change labels of activated nodes to minimize its conflict with neighboring nodes
    max_iter = 10
    for i in range(max_iter):
        labels_rp = labels.unsqueeze(1).repeat(1, n_nodes, 1)

        act = torch.rand(bz, n_nodes, device=x.device) >= 0.5
        conflict_flag = torch.sum(adj_uw * (labels.unsqueeze(-1) == labels_rp), -1) > torch.sum(adj_uw, -1) / 2
        labels[act * conflict_flag] *= -1

    # keep the same number of even & odd nodes
    for i in range(bz):
        # flip extra odd to even
        odd_flag = labels[i] == 1
        odd_idx = torch.nonzero(odd_flag, as_tuple=False).squeeze(-1)
        odd_extra = (odd_flag.sum(-1) - n_nodes // 2).item()
        if odd_extra > 0:
            flip_idx = odd_idx[torch.randperm(odd_idx.size(0))[:odd_extra]]
            labels[i][flip_idx] *= -1

        # flip extra even to odd
        even_flag = labels[i] == -1
        even_idx = torch.nonzero(even_flag, as_tuple=False).squeeze(-1)
        even_extra = (even_flag.sum(-1) - n_nodes // 2).item()
        if even_extra > 0:
            flip_idx = even_idx[torch.randperm(even_idx.size(0))[:even_extra]]
            labels[i][flip_idx] *= -1

    labels[labels == -1] = 0    # 0 - even, 1 - odd

    split_output = split_helper(x, adj_uw, adj_w, labels, split_adj=split_adj)

    return adj_uw, adj_w, p_dist, labels, split_output


if __name__ == '__main__':
    import open3d as o3d
    import numpy as np

    pcd_1 = o3d.io.read_point_cloud('../preliminary/chair_1_sampled.ply')
    pcd_1 = torch.Tensor(np.asarray(pcd_1.points))

    pcd_2 = o3d.io.read_point_cloud('../preliminary/chair_2_sampled.ply')
    pcd_2 = torch.Tensor(np.asarray(pcd_2.points))

    pcd = torch.stack([pcd_1, pcd_2], dim=0).cuda()

    even_odd_split(pcd)
