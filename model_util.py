# local_operator() and local_operator_withnorm() are referred from: https://github.com/mutianxu/GDANet/blob/main/util/GDANet_util.py
# index_points() are referred from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py

import torch

import pdb


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def knn1(x, k):
    """
    Not used.
    :param x: (b, c, n_pts)
    :param k: a scalar
    :return: (b, n_pts, k), (b, n_pts, n_pts)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx, pairwise_distance


def knn(x, k):
    """
    :param x: (b, c, n_pts)
    :param k: a scalar
    :return: (b, n_pts, k), (b, n_pts, n_pts)
    """
    x = x.transpose(1, 2).contiguous()
    pairwise_distance = torch.cdist(x, x)
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)

    return idx, pairwise_distance


def local_operator1(x, k):
    """
    Not used.
    :param x: (b, c, n_pts)
    :param k: a scalar
    :return: (b, 2 * c, n_pts, k)
    """
    batch_size, _, num_points = x.shape

    x = x.view(batch_size, -1, num_points)
    idx, pairwise_distance = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor = neighbor.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((neighbor - x, neighbor), dim=3).permute(0, 3, 1, 2).contiguous()  # local and global all in

    return feature, pairwise_distance


def local_operator(x, idx, k):
    """
    :param x: (b, c, n_pts)
    :param k: a scalar
    :return: (b, 2 * c, n_pts, k)
    """
    batch_size, _, num_points = x.shape

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor = neighbor.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((neighbor - x, neighbor), dim=3).permute(0, 3, 1, 2).contiguous()  # local and global all in

    return feature


def local_operator_withnorm1(x, norm_plt, k):
    """
    Not used.
    :param x: (b, c=3, n_pts)
    :param norm_plt: (b, c=3, n_pts)
    :param k: a scalar
    :return: (b, c, n_pts, k)
    """
    batch_size, _, num_points = x.shape

    x = x.view(batch_size, -1, num_points)
    norm_plt = norm_plt.view(batch_size, -1, num_points)
    idx, pairwise_distance = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    norm_plt = norm_plt.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_norm = norm_plt.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)
    neighbor_norm = neighbor_norm.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor - x, neighbor, neighbor_norm), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, pairwise_distance


def local_operator_withnorm(x, idx, norm_plt, k):
    """
    :param x: (b, c=3, n_pts)
    :param norm_plt: (b, c=3, n_pts)
    :param k: a scalar
    :return: (b, c, n_pts, k)
    """
    batch_size, _, num_points = x.shape

    x = x.view(batch_size, -1, num_points)
    norm_plt = norm_plt.view(batch_size, -1, num_points)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    norm_plt = norm_plt.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_norm = norm_plt.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)
    neighbor_norm = neighbor_norm.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor - x, neighbor, neighbor_norm), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


if __name__ == '__main__':
    import time

    x = torch.randn((48, 3, 1024)).cuda()

    st = time.time()
    idx1, pdist1 = knn1(x, 32)
    stop1 = time.time()
    idx2, pdist2 = knn(x, 32)
    stop2 = time.time()

    print(stop1 - st, stop2 - stop1)

    compare = torch.isclose(-pdist1, pdist2 ** 2)
    ratio = 1 - compare.sum().float() / compare.nelement()
    print(ratio)

    print('Done.')
