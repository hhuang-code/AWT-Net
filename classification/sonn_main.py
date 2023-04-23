import os
import random
import logging
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from config import opts
from scanobjectnn import ScanObjectNN
from model import AWTNet
from weight_init import weight_init
from misc_util import cal_loss

import pdb


# torch.autograd.set_detect_anomaly(True)

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


def backup(opts):
    if not opts.eval:
        os.system('cp config_s.py ' + opts.exp_name + '/' + 'config_s.py.backup')
        os.system('cp sonn_main.py ' + opts.exp_name + '/' + 'sonn_main.py.backup')
        os.system('cp lifting.py ' + opts.exp_name + '/' + 'lifting.py.backup')
        os.system('cp model.py ' + opts.exp_name + '/' + 'model.py.backup')
        os.system('cp scanobjectnn.py ' + opts.exp_name + '/' + 'scanobjectnn.py.backup')
        os.system('cp transformer.py ' + opts.exp_name + '/' + 'transformer.py.backup')

        os.system('cp ../data_utils.py ' + opts.exp_name + '/' + 'data_utils.py.backup')
        os.system('cp ../model_util.py ' + opts.exp_name + '/' + 'model_util.py.backup')
        os.system('cp ../wavelet_util.py ' + opts.exp_name + '/' + 'wavelet_util.py.backup')


def train(gpu, opts):
    logging.basicConfig(filename=os.path.join(opts.exp_name, opts.log_path, 'train_{}.log'.format(gpu)),
                        filemode='a', level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    print(opts)
    logging.info(opts)

    torch.cuda.set_device(gpu)

    rank = opts.n_rank * opts.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=opts.world_size, rank=rank)

    train_set = ScanObjectNN(opts.data_path, opts.n_pts, 'train', opts.with_bg)
    train_sampler = DistributedSampler(train_set, num_replicas=opts.world_size, rank=rank) if opts.gpus > 1 else None
    train_loader = DataLoader(train_set, sampler=train_sampler, num_workers=4, batch_size=opts.batch_size,
                              shuffle=train_sampler is None, drop_last=True)

    test_set = ScanObjectNN(opts.data_path, opts.n_pts, 'test', opts.with_bg)
    test_sampler = DistributedSampler(test_set, num_replicas=opts.world_size, rank=rank) if opts.gpus > 1 else None
    test_loader = DataLoader(test_set, sampler=test_sampler, num_workers=4, batch_size=opts.test_batch_size,
                             shuffle=False, drop_last=False)

    model = AWTNet(opts)
    model.apply(weight_init)
    device = torch.device('cuda', gpu)
    model = model.to(device)

    if opts.gpus > 1:
        if opts.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=False)
    print('{} gpus are used.'.format(opts.gpus))

    if opts.optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr * 100, momentum=opts.momentum, weight_decay=1e-4)
    elif opts.optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    else:
        raise ValueError('Optimizer {} is not supported.'.format(opts.optim_type))

    if opts.sched_type == 'step':
        scheduler = StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)
    elif opts.sched_type == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=opts.epochs, eta_min=opts.lr)
    else:
        raise ValueError('Scheduler {} is not supported.'.format(opts.sched_type))

    criterion = cal_loss

    best_test_acc = 0

    for epoch in range(opts.epochs):
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred, train_true = [], []
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, (data, label) in enumerate(train_loader):
            data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze()
            data = data.permute(0, 2, 1)    # (b, 3, n_pts)
            batch_size = data.shape[0]

            optimizer.zero_grad()
            logits, regus = model(data)
            loss = criterion(logits, label) + torch.sum(regus)
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        scheduler.step()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        msg = 'Train: {}, loss: {:.6f}, train acc: {:.6f}, train avg acc: {:.6f}'.format(
            epoch, train_loss * 1.0 / count, metrics.accuracy_score(train_true, train_pred),
            metrics.balanced_accuracy_score(train_true, train_pred))
        print(msg)
        logging.info(msg)

        # Test ------------------------------------------------------------------------------------
        torch_rng_state = torch.get_rng_state()
        torch_cuda_rng_state = torch.cuda.get_rng_state()

        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred, test_true = [], []
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.shape[0]

                logits, regus = model(data)
                loss = criterion(logits, label) + torch.sum(regus)
                preds = logits.max(dim=1)[1]

                preds_list = [preds for _ in range(opts.world_size)]
                label_list = [label for _ in range(opts.world_size)]
                dist.all_gather(preds_list, preds)
                dist.all_gather(label_list, label)

                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        msg = 'Test: {}, loss: {:.6f}, test acc: {:.6f}, test avg acc: {:.6f}'.format(
            epoch, test_loss * 1.0 / count, test_acc, avg_per_class_acc)
        print(msg)
        logging.info(msg)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            msg = 'Best Acc: {:.6f}'.format(best_test_acc)
            print(msg)
            logging.info(msg)
            if opts.gpus > 1:
                if dist.get_rank() == 0:
                    torch.save({'model_state_dict': model.state_dict()},
                               os.path.join(opts.exp_name, opts.ck_path, 'best_model.pth'))
            else:
                torch.save({'model_state_dict': model.state_dict()},
                           os.path.join(opts.exp_name, opts.ck_path, 'best_model.pth'))
            torch.save({'torch_rng_state': torch_rng_state,
                        'torch_cuda_rng_state': torch_cuda_rng_state},
                       os.path.join(opts.exp_name, opts.ck_path, 'best_random_state_{}.pth'.format(gpu)))


def test(gpu, opts):
    logging.basicConfig(filename=os.path.join(opts.exp_name, opts.log_path, 'test.log'),
                        filemode='a', level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info(opts)

    torch.cuda.set_device(gpu)

    rank = opts.n_rank * opts.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=opts.world_size, rank=rank)

    test_set = ScanObjectNN(opts.data_path, opts.n_pts, 'test', opts.with_bg)
    test_sampler = DistributedSampler(test_set, num_replicas=opts.world_size, rank=rank) if opts.gpus > 1 else None
    test_loader = DataLoader(test_set, sampler=test_sampler, num_workers=4, batch_size=opts.test_batch_size,
                             shuffle=False, drop_last=False)

    model = AWTNet(opts)
    device = torch.device('cuda', gpu)
    model = model.to(device)

    if opts.gpus > 1:
        if opts.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=False)
    print('{} gpus are used.'.format(opts.gpus))

    checkpoint = torch.load(os.path.join(opts.exp_name, opts.ck_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_true = []
    test_pred = []

    random_state = torch.load(os.path.join(opts.exp_name, opts.ck_path, 'best_random_state_{}.pth'.format(gpu)))
    torch.set_rng_state(random_state['torch_rng_state'])
    torch.cuda.set_rng_state(random_state['torch_cuda_rng_state'])

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze()
            data = data.permute(0, 2, 1)
            logits, _ = model(data)
            preds = logits.max(dim=1)[1]

            preds_list = [preds for _ in range(opts.world_size)]
            label_list = [label for _ in range(opts.world_size)]
            dist.all_gather(preds_list, preds)
            dist.all_gather(label_list, label)

            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

    msg = 'test acc: {:.6f}, test avg acc: {:.6f}'.format(test_acc, avg_per_class_acc)
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