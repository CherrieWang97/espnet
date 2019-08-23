#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import pdb
import json
import logging
import os
import sys
import random

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
import numpy as np
from tensorboardX import SummaryWriter
import torch

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
import espnet.lm.pytorch_backend.lm as lm_pytorch
from espnet.nets.mt_interface import MTInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_mtbatchset
from espnet.utils.training.batchfy import sequence_to_id
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

# duplicate
from espnet.asr.pytorch_backend.asr import CustomEvaluator
from espnet.asr.pytorch_backend.asr import CustomUpdater
from espnet.asr.pytorch_backend.asr import load_trained_model

import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest

REPORT_INTERVAL = 100


class CustomConverter(object):
    """Custom batch converter for Pytorch

    :param int idim : index for <pad> in the source language
    """

    def __init__(self, idim):
        self.pad = 2
        self.ignore_id = -1

    def __call__(self, batch, device):
        """Transforms a batch and send it to a device

        :param list batch: The batch to transform
        :param torch.device device: The device to send to
        :return: a tuple xs_pad, ilens, ys_pad
        :rtype (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ilens, ys = batch[0]
        xs = xs.to(device)
        ys = ys.to(device)
        ilens = ilens.to(device)

        return xs, ilens, ys

    def transform(self, batch, add_noise=True):
        src = []
        tgt = []
        lens = []
        if add_noise:
            new_batch = []
            for item in batch:
                xs, ys = item
                ilen = len(xs)
                xs_n = []
                i = 0
                repeat = False
                word_len = 0
                while i < ilen:
                    xs_n.append(xs[i])
                    word_len += 1
                    if xs[i] == 30 and repeat == False:
                        blank_len = random.randint(0, 4 * word_len)
                        xs_n.extend([114] * blank_len)
                        word_len = 0
                    if random.random() > 0.55:
                        i += 1
                        repeat = False
                    else:
                        if repeat == True:
                            if random.random() < 0.85:
                                i += 1
                                repeat = False
                        else:
                            repeat = True
                xs = np.asarray(xs_n)
                new_batch.append((xs, ys))
            new_batch.sort(key=lambda x: -len(x[0]))
            batch = new_batch
        for item in batch:
            xs, ys = item
            ilens = len(xs)
            src.append(xs)
            tgt.append(ys)
            lens.append(ilens)

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).long() for x in src], self.pad)
        lens = torch.from_numpy(np.array(lens))
        ys_pad = pad_list([torch.from_numpy(y).long() for y in tgt], self.ignore_id)

        return xs_pad, lens, ys_pad

def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    idim = args.src_vocab
    odim = args.tgt_vocab
    if args.share_dict:
        idim = odim
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))
    
    # specify model architecture
    model_class = dynamic_import(args.model_module)
    model = model_class(idim, odim, args)
    assert isinstance(model, MTInterface)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # Setup an optimizer
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

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))
    
    # Setup a converter
    converter = CustomConverter(idim=idim)

    # read json data

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_mtbatchset(args.train_src, args.train_trg, args.batch_size)
    valid = make_mtbatchset(args.valid_src, args.valid_trg, args.batch_size)

    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, converter.transform),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = ToggleableShufflingSerialIterator(
            TransformDataset(train, converter.transform),
            batch_size=1, shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingSerialIterator(
            TransformDataset(valid, converter.transform),
            batch_size=1, repeat=False, shuffle=False)
    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer, converter, device, args.ngpu)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_load(args.resume, model)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device))
    

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_att', 'validation/main/loss_att'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))
    trainer.extend(extensions.PlotReport(['main/ppl', 'validation/main/ppl'],
                                         'epoch', file_name='ppl.png'))

    # Save best models
    trainer.extend(snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(snapshot_object(model, 'model.acc.best'),
                   trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(10000, 'iteration'))
    
    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
    
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
                   'main/acc', 'validation/main/acc',
                   'main/ppl', 'validation/main/ppl',
                   'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(REPORT_INTERVAL, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))
    set_early_stop(trainer, args)
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)

def id_to_sentence(char_list, sent):
    ret = []
    for i in sent:
        ret.append(char_list[i])
    return ' '.join(ret)


def trans(args):
    """Decode with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, MTInterface)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()


    # read json data
    sentences = []
    with open(args.recog_path, 'rb') as f:
        for line in f:
            line = line.decode().strip().split()
            sent = np.asarray(list(map(int, line)), dtype=np.int32)
            sentences.append(sent)
    # read json data
    one_best = []
    if args.batchsize == 0:
        with torch.no_grad():
            for idx, sent in enumerate(sentences):
                nbest_hyps = model.translate([sent], args, train_args.char_list)
                best = nbest_hyps[0]['yseq']
                if best[0] == 1:
                    best = best[1:]
                if best[-1] == 2:
                    best = best[:-1]
                one_best.append(best)
                #one_best.append(id_to_sentence(train_args.char_list, best))
    else:
        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)
        
        lens = [len(sent) for sent in sentences]
        sorted_index = sorted(range(len(lens)), key=lambda i: -lens[i])
        sorted_sents = [sentences[i] for i in sorted_index]
        start = 0
        sort_one_best = []
        with torch.no_grad():
            while True:
                end = min(len(sentences), start + args.batchsize)
                pdb.set_trace()
                y = model.translate_batch(sorted_sents[start:end], args, train_args.char_list)
                for ret in y:
                    best = ret[0]['yseq']
                    if best[0] == 1:
                        best = best[1:]
                    if best[-1] == 2:
                        best = best[:-1]
                    sort_one_best.append(best)
                start += args.batchsize
                if end == len(sentences):
                    break
        one_best = [sort_one_best[sorted_index.index(i)] for i in range(len(sentences))]

    with open(args.result_label, 'w', encoding="utf-8") as f:
        for r in one_best:
            f.write(' '.join(list(map(str, r)))+"\n")
