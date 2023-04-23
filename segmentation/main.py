# Lines 168 to 221, lines 447 to 483 are referred from: https://github.com/mutianxu/GDANet/blob/main/main_ptseg.py

import os
import json
import random
import logging
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from config_s import opts
from shapenet import PartNormalDataset
from ptseg_model import AWTNet
from weight_init import weight_init
from misc_util import to_categorical, compute_overall_iou

import pdb

# torch.backends.cudnn.benchmark = True


manual_seed = 123

random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


LOG_FORMAT = '%(asctime)s - %(message)s'
DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


classes_str = ['aero', 'bag', 'cap', 'car', 'chair', 'ear', 'guitar', 'knife',
               'lamp', 'lapt', 'moto', 'mug', 'Pistol', 'rock', 'stake', 'table']

color_map_file = os.path.join(opts.data_path, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))


def backup(opts):
    if not opts.eval:
        os.system('cp scripts/main.sh ' + opts.exp_name + '/' + 'main.sh.backup')

        os.system('cp config_s.py ' + opts.exp_name + '/' + 'config_s.py.backup')
        os.system('cp main.py ' + opts.exp_name + '/' + 'main.py.backup')
        os.system('cp ptseg_model.py ' + opts.exp_name + '/' + 'ptseg_model.py.backup')
        os.system('cp shapenet.py ' + opts.exp_name + '/' + 'shapenet.py.backup')
        os.system('cp ../classification/lifting.py ' + opts.exp_name + '/' + 'lifting.py.backup')
        os.system('cp ../classification/transformer.py ' + opts.exp_name + '/' + 'transformer.py.backup')

        os.system('cp ../data_utils.py ' + opts.exp_name + '/' + 'data_utils.py.backup')
        os.system('cp ../model_util.py ' + opts.exp_name + '/' + 'model_util.py.backup')
        os.system('cp ../wavelet_util.py ' + opts.exp_name + '/' + 'wavelet_util.py.backup')


def output_color_point_cloud(data, seg, color_map, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def train(gpu, opts):
    logging.basicConfig(filename=os.path.join(opts.exp_name, opts.log_path, 'train_{}.log'.format(gpu)),
                        filemode='a', level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info(opts)

    torch.cuda.set_device(gpu)

    rank = opts.n_rank * opts.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=opts.world_size, rank=rank)

    train_data = PartNormalDataset(opts.data_path, opts.n_pts, 'trainval', normalize=opts.normalize)
    train_sampler = DistributedSampler(train_data, num_replicas=opts.world_size, rank=rank) if opts.gpus > 1 else None
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=opts.batch_size,
                              shuffle=train_sampler is None, num_workers=4, drop_last=True)

    test_data = PartNormalDataset(opts.data_path, opts.n_pts, 'test', normalize=opts.normalize)
    test_sampler = DistributedSampler(test_data, num_replicas=opts.world_size, rank=rank) if opts.gpus > 1 else None
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=opts.test_batch_size,
                             shuffle=False, num_workers=4, drop_last=False)

    num_part = 50
    model = AWTNet(opts, num_part)
    model.apply(weight_init)
    device = torch.device('cuda', gpu)
    model = model.to(device)

    if opts.gpus > 1:
        if opts.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=False)
        print('{} gpus are used.'.format(opts.gpus))

    if opts.optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr * 100, momentum=opts.momentum, weight_decay=0)
    elif opts.optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        raise ValueError('Optimizer {} is not supported.'.format(opts.optim_type))

    if opts.sched_type == 'step':
        scheduler = StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)
    elif opts.sched_type == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=opts.epochs,
                                      eta_min=opts.lr if opts.optim_type == 'sgd' else opts.lr / 100)
    else:
        raise ValueError('Scheduler {} is not supported.'.format(opts.sched_type))

    start_epoch = 0

    if opts.resume:
        checkpoint = torch.load(os.path.join(opts.exp_name, opts.ck_path, 'best_{}_model.pth'.format(opts.model_type)),
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = checkpoint['scheduler']

        start_epoch = checkpoint['epoch'] + 1

        random_state = torch.load(os.path.join(opts.exp_name, opts.ck_path, 'best_{}_random_state_{}.pth'.format(opts.model_type, gpu)))
        torch.set_rng_state(random_state['torch_rng_state'])
        torch.cuda.set_rng_state(random_state['torch_cuda_rng_state'])

    best_acc = 0
    best_class_iou = 0
    best_instance_iou = 0
    num_part = 50
    num_classes = 16
    for epoch in range(start_epoch, opts.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_epoch(opts, train_loader, model, optimizer, scheduler, epoch, num_part, num_classes)

        torch_rng_state = torch.get_rng_state()
        torch_cuda_rng_state = torch.cuda.get_rng_state()

        test_metrics, total_per_cat_iou = test_epoch(opts, test_loader, model, epoch, num_part, num_classes)

        # 1. when get the best accuracy, save the model:
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            msg = 'Max Acc: {:.6f}'.format(best_acc)
            print(msg)
            logging. info(msg)

            if (opts.gpus > 1 and dist.get_rank() == 0) or opts.gpus == 1:
                state = {
                    'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler,
                    'epoch': epoch, 'best_acc': best_acc}
                torch.save(state, os.path.join(opts.exp_name, opts.ck_path, 'best_acc_model.pth'))
            torch.save({'torch_rng_state': torch_rng_state, 'torch_cuda_rng_state': torch_cuda_rng_state},
                       os.path.join(opts.exp_name, opts.ck_path, 'best_acc_random_state_{}.pth'.format(gpu)))

        # 2. when get the best instance_iou, save the model:
        if test_metrics['shape_avg_iou'] > best_instance_iou:
            best_instance_iou = test_metrics['shape_avg_iou']
            msg = 'Max instance iou: {:.6f}'.format(best_instance_iou)
            print(msg)
            logging.info(msg)

            if (opts.gpus > 1 and dist.get_rank() == 0) or opts.gpus == 1:
                state = {
                    'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler,
                    'epoch': epoch, 'best_instance_iou': best_instance_iou}
                torch.save(state, os.path.join(opts.exp_name, opts.ck_path, 'best_insiou_model.pth'))
            torch.save({'torch_rng_state': torch_rng_state, 'torch_cuda_rng_state': torch_cuda_rng_state},
                       os.path.join(opts.exp_name, opts.ck_path, 'best_insiou_random_state_{}.pth'.format(gpu)))

        # 3. when get the best class_iou, save the model:
        class_iou = 0
        for cat_idx in range(num_classes):
            class_iou += total_per_cat_iou[cat_idx]
        avg_class_iou = class_iou / num_classes

        if avg_class_iou > best_class_iou:
            best_class_iou = avg_class_iou
            # print the iou of each class:
            for cat_idx in range(num_classes):
                msg = classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx])
                print(msg)
                logging.info(msg)
            msg = 'Max class iou: {:.6f}'.format(best_class_iou)
            print(msg)
            logging.info(msg)

            if (opts.gpus > 1 and dist.get_rank() == 0) or opts.gpus == 0:
                state = {
                    'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler,
                    'epoch': epoch, 'best_class_iou': best_class_iou}
                torch.save(state, os.path.join(opts.exp_name, opts.ck_path, 'best_clsiou_model.pth'))
            torch.save({'torch_rng_state': torch_rng_state, 'torch_cuda_rng_state': torch_cuda_rng_state},
                       os.path.join(opts.exp_name, opts.ck_path, 'best_clsiou_random_state_{}.pth'.format(gpu)))

    # report best acc, ins_iou, cls_iou
    msg = 'Final Max Acc: {:.6f}'.format(best_acc)
    print(msg)
    logging.info(msg)
    msg = 'Final Max instance iou: {:.6f}'.format(best_instance_iou)
    print(msg)
    logging.info(msg)
    msg = 'Final Max class iou: {:.6f}'.format(best_class_iou)
    print(msg)
    logging.info(msg)


def train_epoch(opts, train_loader, model, optimizer, scheduler, epoch, num_part, num_classes):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    metrics = defaultdict(lambda: list())

    model.train()
    for batch_id, (points, label, target, norm_plt) in enumerate(train_loader):
        batch_size, num_point, _ = points.shape
        points = points.transpose(2, 1)         # (b, 3, n_pts)
        norm_plt = norm_plt.transpose(2, 1)     # (b, 3, n_pts)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        seg_pred, regus = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: (b, n_pts, n_parts)

        # loss
        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.long().view(-1, 1)[:, 0]) + torch.sum(regus)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy
        pred_choice = seg_pred.contiguous().max(-1)[1]
        correct = pred_choice.eq(target.contiguous()).sum()

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # batch_iou list:[iou1,iou2,...,iou#b_size]
        # total iou of current batch in each process:
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)

        # sum
        shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    if opts.sched_type == 'cos':
        scheduler.step()
    elif opts.sched_type == 'step':
        if opts.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opts.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opts.param_groups:
                param_group['lr'] = 0.9e-5
    msg = 'Learning rate: {:.6f}'.format(optimizer.param_groups[0]['lr'])
    print(msg)
    logging.info(msg)

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    msg = 'Train {}, loss: {:.6f}, train acc: {:.6f}, train ins_iou: {:.6f}'.format(
        epoch + 1, train_loss * 1.0 / count, metrics['accuracy'], metrics['shape_avg_iou'])
    print(msg)
    logging.info(msg)


def test_epoch(opts, test_loader, model, epoch, num_part, num_classes):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    num_classes = 16
    final_total_per_cat_iou = np.zeros(num_classes).astype(np.float32)
    final_total_per_cat_seen = np.zeros(num_classes).astype(np.int32)
    metrics = defaultdict(lambda: list())

    model.eval()
    with torch.no_grad():
        for batch_id, (points, label, target, norm_plt) in enumerate(test_loader):
            batch_size, num_point, _ = points.shape
            points = points.transpose(2, 1)     # (b, 3, n_pts)
            norm_plt = norm_plt.transpose(2, 1) # (b, 3, n_pts)
            points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                              target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

            seg_pred, regus = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: (b, n_pts, n_parts)

            seg_pred_list = [seg_pred for _ in range(opts.world_size)]
            label_list = [label for _ in range(opts.world_size)]
            target_list = [target for _ in range(opts.world_size)]
            dist.all_gather(seg_pred_list, seg_pred)
            dist.all_gather(label_list, label)
            dist.all_gather(target_list, target)

            # instance iou without considering the class average at each batch_size:
            batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # batch_iou list:[iou1,iou2,...,iou#b_size]

            # per category iou at each batch_size:
            for shape_idx in range(seg_pred.size(0)):  # sample_idx
                cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
                final_total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]  # add the iou belongs to this cat
                final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen

            # total iou of current batch in each process:
            batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)

            # prepare seg_pred and target for later calculating loss and acc:
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]

            loss = F.nll_loss(seg_pred.contiguous(), target.contiguous().long()) + torch.sum(regus)

            # accuracy
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).sum()  # total number of correct-predict pts

            loss = torch.mean(loss)
            shape_ious += batch_ious.item()  # count the sum of ious in each iteration
            count += batch_size  # count the total number of samples in each iteration
            test_loss += loss.item() * batch_size
            accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    for cat_idx in range(num_classes):
        if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
            final_total_per_cat_iou[cat_idx] = \
                final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[cat_idx]  # avg class iou across all samples

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    msg = 'Test {}, loss: {:.6f}, test acc: {:.6f}  test ins_iou: {:.6f}'.format(
        epoch + 1, test_loss * 1.0 / count, metrics['accuracy'], metrics['shape_avg_iou'])
    print(msg)
    logging.info(msg)

    return metrics, final_total_per_cat_iou


def test(gpu, opts):
    logging.basicConfig(filename=os.path.join(opts.exp_name, opts.log_path, 'test.log'),
                        filemode='a', level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info(opts)

    torch.cuda.set_device(gpu)

    rank = opts.n_rank * opts.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=opts.world_size, rank=rank)

    test_data = PartNormalDataset(opts.data_path, opts.n_pts, 'test', normalize=opts.normalize)
    test_sampler = DistributedSampler(test_data, num_replicas=opts.world_size, rank=rank) if opts.gpus > 1 else None
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=opts.test_batch_size,
                             shuffle=False, num_workers=4, drop_last=False)

    # load models
    num_part = 50
    model = AWTNet(opts, num_part)
    device = torch.device('cuda', gpu)
    model = model.to(device)

    if opts.gpus > 1:
        if opts.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)
    print('{} gpus are used.'.format(opts.gpus))

    checkpoint = torch.load(os.path.join(opts.exp_name, opts.ck_path, 'best_{}_model.pth'.format(opts.model_type)),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    random_state = torch.load(os.path.join(opts.exp_name, opts.ck_path, 'best_{}_random_state_{}.pth'.format(opts.model_type, gpu)))
    torch.set_rng_state(random_state['torch_rng_state'])
    torch.cuda.set_rng_state(random_state['torch_cuda_rng_state'])

    num_part = 50
    num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    shape_ious = []
    total_per_cat_iou = np.zeros((num_classes)).astype(np.float32)
    total_per_cat_seen = np.zeros((num_classes)).astype(np.int32)

    model.eval()
    with torch.no_grad():
        for batch_id, (points, label, target, norm_plt) in enumerate(test_loader):
            batch_size, num_point, _ = points.shape
            points = points.transpose(2, 1)
            norm_plt = norm_plt.transpose(2, 1)
            points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(non_blocking=True), \
                                              target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

            seg_pred, _ = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: (b, n_pts, n_parts)

            seg_pred_list = [seg_pred for _ in range(opts.world_size)]
            label_list = [label for _ in range(opts.world_size)]
            target_list = [target for _ in range(opts.world_size)]
            dist.all_gather(seg_pred_list, seg_pred)
            dist.all_gather(label_list, label)
            dist.all_gather(target_list, target)

            # visualize segmentation output
            if opts.visualize:
                path = os.path.join(opts.exp_name, opts.out_path)
                if not os.path.exists(path):
                    os.makedirs(path)

                for i, (pts, cls, seg_val, seg_pred_val) in enumerate(zip(points, label, target, seg_pred)):
                    pts = pts.transpose(1, 0).cpu().numpy()
                    seg_val = seg_val.cpu().numpy()
                    seg_pred_val = seg_pred_val.max(1)[1].cpu().numpy()
                    output_color_point_cloud(pts, seg_val, color_map,
                                             os.path.join(path, classes_str[cls] + '_' + str(i) + '_gt.obj'))
                    output_color_point_cloud(pts, seg_pred_val, color_map,
                                             os.path.join(path, classes_str[cls] + '_' + str(i) + '_pred.obj'))
                    output_color_point_cloud_red_blue(pts, np.int32(seg_val == seg_pred_val),
                                                      os.path.join(path, classes_str[cls] + '_' + str(i) + '_diff.obj'))

            # instance iou without considering the class average at each batch_size:
            batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # seg_pred: (b, n_pts, n_parts)
            shape_ious += batch_shapeious

            # per category iou at each batch_size:
            for shape_idx in range(seg_pred.size(0)):  # sample_idx
                cur_gt_label = label[shape_idx]  # label[sample_idx]
                total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
                total_per_cat_seen[cur_gt_label] += 1

            # accuracy
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['shape_avg_iou'] = np.mean(shape_ious)
    for cat_idx in range(num_classes):
        if total_per_cat_seen[cat_idx] > 0:
            total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # calculate the iou of each class and the avg class iou
    class_iou = 0
    for cat_idx in range(num_classes):
        class_iou += total_per_cat_iou[cat_idx]
        msg = classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]) # print the iou of each class
        print(msg)
        logging.info(msg)

    avg_class_iou = class_iou / num_classes
    msg = 'test acc: {:.6f}  test class mIOU: {:.6f}, test instance mIOU: {:.6f}'.format(
        metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
    print(msg)
    logging.info(msg)


if __name__ == '__main__':
    opts.world_size = opts.gpus * opts.nodes

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '23456'

    opts.exp_name = os.path.join('exps', opts.exp_name)

    if not os.path.exists(os.path.join(opts.exp_name, opts.log_path)):
        os.makedirs(os.path.join(opts.exp_name, opts.log_path))

    if not os.path.exists(os.path.join(opts.exp_name, opts.img_path)):
        os.makedirs(os.path.join(opts.exp_name, opts.img_path))

    if not os.path.exists(os.path.join(opts.exp_name, opts.ck_path)):
        os.makedirs(os.path.join(opts.exp_name, opts.ck_path))

    backup(opts)

    if not opts.eval:
        if opts.gpus > 1:
            mp.spawn(train, nprocs=opts.gpus, args=(opts,))
        else:
            train(0, opts)
    else:
        if opts.gpus > 1:
            mp.spawn(test, nprocs=opts.gpus, args=(opts,))
        else:
            test(0, opts)