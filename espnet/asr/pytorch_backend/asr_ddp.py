#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""

import copy
import json
import logging
import math
import os
import sys
import pdb
import random
import time

from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest

class AverageMeter(object):
    """Compute and storesthe average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt +'} ({avg' + self.fmt +'})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches)+']'

class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, load_fn, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.load_fn = load_fn

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, device, subsampling_factor=1, dtype=torch.float32):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype
        self.device = device

    def __call__(self, batch):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        xs, ys = batch
        ys = list(ys)

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == 'c':
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0).to(self.dtype).cuda(self.device, non_blocking=True)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0).to(self.dtype).cuda(self.device, non_blocking=True)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {'real': xs_pad_real, 'imag': xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(self.dtype).cuda(self.device, non_blocking=True)

        ilens = torch.from_numpy(ilens).cuda(self.device, non_blocking=True)
        # NOTE: this is for multi-task learning (e.g., speech translation)
        if isinstance(ys[0], tuple):
            ys_pad_0 = pad_list([torch.from_numpy(np.array(y[0])).long() for y in ys],
                                self.ignore_id).cuda(self.device, non_blocking=True)
            ys_pad_1 = pad_list([torch.from_numpy(np.array(y[1])).long() for y in ys],
                                0).cuda(self.device, non_blocking=True)
            return xs_pad, ilens, ys_pad_0, ys_pad_1
        else:
            ys_pad = pad_list([torch.from_numpy(y).long()
                               for y in ys], self.ignore_id).cuda(self.device, non_blocking=True)

        return xs_pad, ilens, ys_pad

def dist_train(gpu, args):
    """Initialize torch.distributed."""
    args.gpu = gpu
    args.rank = gpu
    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))

    init_method = "tcp://localhost:{port}".format(port=args.port)

    torch.distributed.init_process_group(
        backend='nccl', world_size=args.ngpu, rank=args.gpu,
        init_method=init_method)
    torch.cuda.set_device(args.gpu)
    print('initialize model on gpu: {}'.format(args.gpu))
    if args.enc_init is not None or args.dec_init is not None:
        model = load_trained_modules(83, 106, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(83, 106, args)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)
    print('initialize data sampler')
    converter = CustomConverter(args.gpu)
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=False,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)
    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    num_samples = int(math.ceil(len(train) * 1.0 / args.ngpu))
    total_len = num_samples * args.ngpu
    g = torch.Generator()
    g.manual_seed(args.seed)
    indices = torch.randperm(len(train), generator=g).tolist()
    indices += indices[:(total_len - len(indices))]
    assert len(indices) == total_len
    indices = indices[args.rank:total_len:args.ngpu]
    assert len(indices) == num_samples
    train_dataset = []
    for i in indices:
        train_dataset.append(load_tr(train[i]))
    train_dataset = TransformDataset(train_dataset, converter)
    valid_dataset = TransformDataset(valid, lambda data: converter(load_cv(data)))
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

    start_epoch = 0

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        train_epoch(train_loader, model, optimizer, epoch, args)

def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    acc_meter = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_meter],
        prefix="Epoch: [{}] GPU: [{}]".format(epoch, args.gpu))
    model.train()
    start = time.time()
    for i, (xs, ilens, ys, ys_asr_pad) in enumerate(train_loader):
        """
        xs = xs[0].cuda(args.gpu, non_blocking=True)
        ilens = ilens[0].cuda(args.gpu, non_blocking=True)
        ys = ys[0].cuda(args.gpu, non_blocking=True)
        ys_asr_pad = ys_asr_pad.cuda(args.gpu, non_blocking=True)
        """
        data_time.update(time.time() - start)
        loss, acc = model(xs[0], ilens[0], ys[0])
        losses.update(loss.item())
        acc_meter.update(acc)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - start)
        if i % args.report_interval_iters == 0:
            progress.display(i)                
            #print(dict(model.module.named_parameters())['decoder.output_layer.weight'])
            #return


def train(args):
    """Main training program."""
    if args.ngpu == 0:
        logging.warning('distributed training only supported for GPU training')

    args.ngpu = torch.cuda.device_count()
    if args.ngpu == 0:
        logging.warning('no gpu detected')
        exit(0)
    args.port = random.randint(10000, 20000)
    """
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=False,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)
    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    train_dataset
    train_dataset = TransformDataset(train, lambda data: converter(load_tr(data)))
    valid_dataset = TransformDataset(valid, lambda data: converter(load_cv(data)))
    """
    mp.spawn(dist_train, nprocs=args.ngpu, args=(args,))

    #dist_main(0, args)
