#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import pdb
import copy
import json
import logging
import math
import os

import chainer
import kaldiio
import numpy as np
import torch

from chainer.datasets import TransformDataset
from chainer.iterators import SerialIterator
from chainer import training
from chainer.training import extensions

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import PlotAttentionReport
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import torch_snapshot
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.pytorch_backend.e2e_s2s import Translacotron
from espnet.nets.pytorch_backend.e2e_s2s import TranslacotronLoss
from espnet.transform.transformation import using_transform_config
from espnet.s2s.s2s_utils import DataFeeder
from espnet.utils.io_utils import LoadInputsAndTargets

from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet.utils.training.iterators import ShufflingEnabler


import matplotlib

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

matplotlib.use('Agg')

REPORT_INTERVAL = 100


class CustomEvaluator(extensions.Evaluator):
    """Custom Evaluator for Tacotron2 training

    :param torch.nn.Model model : The model to evaluate
    :param chainer.dataset.Iterator iterator : The validation iterator
    :param target :
    :param CustomConverter converter : The batch converter
    :param torch.device device : The device to use
    """

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with chainer.reporter.report_scope(observation):
                    # convert to torch tensor
                    src_mel, src_phoneme, src_mel_len, src_phoneme_len, trg_mel, trg_phoneme, trg_mel_len, \
        trg_phoneme_len, trg_linear, token_targets = self.converter(batch, self.device)
                    self.model(src_mel, src_mel_len, trg_mel, token_targets, trg_mel_len, src_seq=src_phoneme,src_seq_len=src_phoneme_len, trg_seq=trg_phoneme, trg_seq_len=trg_phoneme_len)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    """Custom updater for Tacotron2 training

    :param torch.nn.Module model: The model to update
    :param float grad_clip : The gradient clipping value to use
    :param chainer.dataset.Iterator train_iter : The training iterator
    :param optimizer :
    :param CustomConverter converter : The batch converter
    :param torch.device device : The device to use
    """

    def __init__(self, model, grad_clip, train_iter, optimizer, converter, device):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip = grad_clip
        self.converter = converter
        self.device = device
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        src_mel, src_phoneme, src_mel_len, src_phoneme_len, trg_mel, trg_phoneme, trg_mel_len, \
        trg_phoneme_len, trg_linear, token_targets = self.converter(batch, self.device)

        # compute loss and gradient
        loss = self.model(src_mel, src_mel_len, trg_mel, token_targets, trg_mel_len, src_seq=src_phoneme,
                          src_seq_len=src_phoneme_len, trg_seq=trg_phoneme, trg_seq_len=trg_phoneme_len)
        optimizer.zero_grad()
        loss.backward()

        # compute the gradient norm to check if it is normal or not
        grad_norm = self.clip_grad_norm(self.model.parameters(), self.grad_clip)
        logging.debug('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()


class CustomConverter(object):
    """Custom converter for Translacotron training

    :param bool return_targets:
    :param bool use_speaker_embedding:
    :param bool use_second_target:
    """

    def __init__(self, token_pad=1.0, mel_pad=0.0, bos_id=0, eos_id=0, reduction=1):
        super(CustomConverter, self).__init__()
        self.token_pad = token_pad
        self.mel_pad = mel_pad
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.reduction = reduction


    def transform(self, batch):
        # load batch
        src_mel = []
        src_phoneme = []
        src_mel_len = []
        src_phoneme_len = []
        trg_mel = []
        trg_phoneme = []
        trg_mel_len = []
        trg_phoneme_len = []
        token_targets = []
        trg_linear = []


        for item in batch:
            src_mel_, src_phoneme_, trg_mel_, trg_phoneme_, trg_linear_, token_target_ = item
            src_mel.append(src_mel_)
            src_phoneme.append(src_phoneme_)
            src_mel_len.append(len(src_mel_))
            src_phoneme_len.append(len(src_phoneme_) + 1)
            trg_mel.append(trg_mel_)
            trg_phoneme.append(trg_phoneme_)
            trg_mel_len.append(len(trg_mel_))
            trg_phoneme_len.append(len(trg_phoneme_) + 1)
            trg_linear.append(trg_linear_)
            token_targets.append(token_target_)
        src_mel = self._prepare_mels(src_mel, 1)
        trg_mel = self._prepare_mels(trg_mel, self.reduction)
        src_phoneme = self._prepare_text(src_phoneme)
        trg_phoneme = self._prepare_text(trg_phoneme)
        trg_linear = self._prepare_mels(trg_linear, self.reduction)
        token_targets = self._prepare_token_targets(token_targets, self.reduction)
        src_mel_len = np.asarray(src_mel_len, dtype=np.int32)
        trg_mel_len = np.asarray(trg_mel_len, dtype=np.int32)
        src_phoneme_len = np.asarray(src_phoneme_len, dtype=np.int32)
        trg_phoneme_len = np.asarray(trg_phoneme_len, dtype=np.int32)

        return (src_mel, src_phoneme, src_mel_len, src_phoneme_len, trg_mel, trg_phoneme, trg_mel_len,
                trg_phoneme_len, trg_linear, token_targets)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        inputs_and_targets = batch[0]

        # parse inputs and targets
        src_mel, src_phoneme, src_mel_len, src_phoneme_len, trg_mel, trg_phoneme, trg_mel_len, \
        trg_phoneme_len, trg_linear, token_targets = inputs_and_targets

        src_mel = torch.from_numpy(src_mel).float().to(device)
        src_phoneme = torch.from_numpy(src_phoneme).long().to(device)
        src_mel_len = torch.from_numpy(src_mel_len).long().to(device)
        src_phoneme_len = torch.from_numpy(src_phoneme_len).long().to(device)
        trg_mel = torch.from_numpy(trg_mel).float().to(device)
        trg_linear = torch.from_numpy(trg_linear).float().to(device)
        trg_mel_len = torch.from_numpy(trg_mel_len).long().to(device)
        trg_phoneme = torch.from_numpy(trg_phoneme).long().to(device)
        trg_phoneme_len = torch.from_numpy(trg_phoneme_len).long().to(device)
        token_targets = torch.from_numpy(token_targets).float().to(device)

        return (src_mel, src_phoneme, src_mel_len, src_phoneme_len, trg_mel, trg_phoneme, trg_mel_len,
                trg_phoneme_len, trg_linear, token_targets)

    def _prepare_text(self, inputs):
        max_len = max([len(x) for x in inputs]) + 1
        return np.stack([self._pad_text(x, max_len) for x in inputs])


    def _prepare_mels(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_mels(t, data_len) for t in targets])

    def _prepare_token_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets]) + 1
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_token_target(t, data_len) for t in targets])

    def _pad_text(self, x, length):
        x = np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self.eos_id)
        x = np.pad(x, (1, 0), mode='constant', constant_values=self.bos_id)
        return x

    def _pad_mels(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self.mel_pad)

    def _pad_token_target(self, t, length):
        return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self.token_pad)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder


def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output
    src_path = os.path.join(args.base_dir, args.source)
    trg_path = os.path.join(args.base_dir, args.target)
    feeder = DataFeeder(src_path, trg_path, args)
    
    # Setup a converter
    converter = CustomConverter()

    train_batchset = feeder.make_train_batches()
    valid_batchset = feeder.make_test_batches()

    train_iter = SerialIterator(
        TransformDataset(train_batchset, converter.transform),
        batch_size=1, shuffle=True)
    valid_iter = SerialIterator(
        TransformDataset(valid_batchset, converter.transform),
        batch_size=1, repeat=False, shuffle=False)

    #pdb.set_trace()
    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        f.write(json.dumps(vars(args), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    translacotron = Translacotron(args)
    logging.info(translacotron)
    logging.info("Make Translacotron Model Done")
    # check the use of multi-gpu
    if args.ngpu > 1:
        translacotron = torch.nn.DataParallel(translacotron, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    translacotron = translacotron.to(device)

    # define loss
    model = TranslacotronLoss(translacotron, args.use_masking, args.bce_pos_weight)
    reporter = model.reporter

    # Setup an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, eps=args.eps,
        weight_decay=args.weight_decay)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, 'target', reporter)
    setattr(optimizer, 'serialize', lambda s: reporter.serialize(s))


    # Set up a trainer
    updater = CustomUpdater(model, args.grad_clip, train_iter, optimizer, converter, device)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device))

    # Save snapshot for each epoch
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # Save best models
    trainer.extend(extensions.snapshot_object(translacotron, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

    # Save attention figure for each epoch

    # Make a plot for training and validation values
    plot_keys = ['main/loss', 'validation/main/loss',
                 'main/l1_loss', 'validation/main/l1_loss',
                 'main/mse_loss', 'validation/main/mse_loss',
                 'main/bce_loss', 'validation/main/bce_loss']
    trainer.extend(extensions.PlotReport(['main/l1_loss', 'validation/main/l1_loss'],
                                         'epoch', file_name='l1_loss.png'))
    trainer.extend(extensions.PlotReport(['main/mse_loss', 'validation/main/mse_loss'],
                                         'epoch', file_name='mse_loss.png'))
    trainer.extend(extensions.PlotReport(['main/bce_loss', 'validation/main/bce_loss'],
                                         'epoch', file_name='bce_loss.png'))
    if args.use_cbhg:
        plot_keys += ['main/cbhg_l1_loss', 'validation/main/cbhg_l1_loss',
                      'main/cbhg_mse_loss', 'validation/main/cbhg_mse_loss']
        trainer.extend(extensions.PlotReport(['main/cbhg_l1_loss', 'validation/main/cbhg_l1_loss'],
                                             'epoch', file_name='cbhg_l1_loss.png'))
        trainer.extend(extensions.PlotReport(['main/cbhg_mse_loss', 'validation/main/cbhg_mse_loss'],
                                             'epoch', file_name='cbhg_mse_loss.png'))
    trainer.extend(extensions.PlotReport(plot_keys, 'epoch', file_name='loss.png'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = plot_keys[:]
    report_keys[0:0] = ['epoch', 'iteration', 'elapsed_time']
    trainer.extend(extensions.PrintReport(report_keys), trigger=(REPORT_INTERVAL, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    set_early_stop(trainer, args)
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(args.tensorboard_dir)
        trainer.extend(TensorboardLogger(writer, None))


    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def decode(args):
    """Decode with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info('args: ' + key + ': ' + str(vars(args)[key]))

    # define model
    translacotron = Translacotron(idim, odim, train_args)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    torch_load(args.model, translacotron)
    translacotron.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    translacotron = translacotron.to(device)

    # read json data
    with open(args.json, 'rb') as f:
        js = json.load(f)['utts']

    # check directory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='tts', load_input=False, sort_in_input_length=False,
        use_speaker_embedding=train_args.use_speaker_embedding,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf)

    with torch.no_grad(), kaldiio.WriteHelper('ark,scp:{o}.ark,{o}.scp'.format(o=args.out)) as f:
        for idx, utt_id in enumerate(js.keys()):
            batch = [(utt_id, js[utt_id])]
            with using_transform_config({'train': False}):
                data = load_inputs_and_targets(batch)
            if train_args.use_speaker_embedding:
                spemb = data[1][0]
                spemb = torch.FloatTensor(spemb).to(device)
            else:
                spemb = None
            x = data[0][0]
            x = torch.LongTensor(x).to(device)

            # decode and write
            outs, _, _ = translacotron.inference(x, args, spemb)
            if outs.size(0) == x.size(0) * args.maxlenratio:
                logging.warning("output length reaches maximum length (%s)." % utt_id)
            logging.info('(%d/%d) %s (size:%d->%d)' % (
                idx + 1, len(js.keys()), utt_id, x.size(0), outs.size(0)))
            f[utt_id] = outs.cpu().numpy()
