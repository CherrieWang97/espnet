#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import pdb
import json
import logging
import math
import os
import sys

from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
import numpy as np
from tensorboardX import SummaryWriter
import torch

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
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
import espnet.lm.pytorch_backend.lm as lm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.mt_interface import MTInterface
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_utils import FileWriterWrapper
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.batchfy import make_mtbatchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.asr.pytorch_backend.asr import CustomConverter as ASRConverter
from espnet.mt.pytorch_backend.mt import CustomConverter as MTConverter

from espnet.nets.pytorch_backend.e2e_multitask import E2E
import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest

REPORT_INTERVAL = 100


class CustomEvaluator(extensions.Evaluator):
    """Custom Evaluator for Pytorch

    :param torch.nn.Module model : The model to evaluate
    :param chainer.dataset.Iterator : The train iterator
    :param target :
    :param CustomConverter converter : The batch converter
    :param torch.device device : The device used
    """

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    x = self.converter(batch, self.device)
                    self.model(*x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    """Custom Updater for Pytorch

    :param torch.nn.Module model : The model to update
    :param int grad_clip_threshold : The gradient clipping value to use
    :param chainer.dataset.Iterator train_iter : The training iterator
    :param torch.optim.optimizer optimizer: The training optimizer
    :param CustomConverter converter: The batch converter
    :param torch.device device : The device to use
    :param int ngpu : The number of gpus to use
    """

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.forward_count = 0
        self.iteration = 0

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        st_iter = self.get_iterator('main')
        asr_iter = self.get_iterator('asr')
        mt_iter = self.get_iterator('mt')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = st_iter.next()
        x = self.converter(batch, self.device)
        loss = self.model(*x).mean()
        #elif self.iteration % 3 == 1:
        #    batch = asr_iter.next()
        #    x = self.converter(batch, self.device)
        #    loss = self.model(*x, task="asr").mean()
        #else:
        #    batch = mt_iter.next()
        #    xs, ilens, ys = batch[0]
        #    xs = xs.to(self.device)
        #    ilens = ilens.to(self.device)
        #    ys = ys.to(self.device)
        #    loss = self.model(xs, ilens, ys, task="mt").mean()
        if math.isnan(float(loss)) or float(loss) > 10000.0:
            return
        loss.backward()

        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
        optimizer.zero_grad()



def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][-1])
    logging.info('#input dims : ' + str(idim))

    asr_model, mt_model = None, None
    # Initialize encoder with pre-trained ASR encoder
    if args.asr_model:
        asr_model = E2E(idim, args.src_vocab, args.trg_vocab, args)
        torch_load(args.asr_model, asr_model)

    # Initialize decoder with pre-trained MT decoder
    if args.mt_model:
        mt_model = E2E(idim, args.src_vocab, args.trg_vocab, args)
        torch_load(args.mt_model, mt_model)

    # specify model architecture
    model = E2E(idim, args.src_vocab, args.trg_vocab, args, asr_model=asr_model, mt_model=mt_model)

    subsampling_factor = model.subsample[0]

    # delete pre-trained models
    if args.asr_model:
        del asr_model
    if args.mt_model:
        del mt_model

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, args.src_vocab, args.trg_vocab, vars(args)),
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
    converter = ASRConverter(subsampling_factor=subsampling_factor)
    mt_converter = MTConverter(args.src_vocab)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    with open(args.asr_json, 'rb') as f:
        asr_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad,
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
    asr_train = make_batchset(asr_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out, args.minibatches,
                              min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                              shortest_first=use_sortagrad,
                              count=args.batch_count,
                              batch_bins=args.batch_bins,
                              batch_frames_in=args.batch_frames_in,
                              batch_frames_out=args.batch_frames_out,
                              batch_frames_inout=args.batch_frames_inout
                              )
    mt_train = make_mtbatchset(args.train_src, args.train_trg, args.mt_batch_size)


    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    train_iter = ToggleableShufflingSerialIterator(
        TransformDataset(train, load_tr),
        batch_size=1, shuffle=not use_sortagrad)
    valid_iter = ToggleableShufflingSerialIterator(
        TransformDataset(valid, load_cv),
        batch_size=1, repeat=False, shuffle=False)
    asr_iter = ToggleableShufflingSerialIterator(
        TransformDataset(asr_train, load_tr),
        batch_size=1, shuffle=not use_sortagrad)
    mt_iter = ToggleableShufflingSerialIterator(
            TransformDataset(mt_train, mt_converter.transform),
            batch_size=1, shuffle=not use_sortagrad)
    iters = {"main": train_iter,
             "asr": asr_iter,
             "mt": mt_iter}
    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, iters, optimizer, converter, device, args.ngpu)
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
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device), trigger=(5000, 'iteration'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/stloss', 'validation/main/stloss'],
                                         'epoch', file_name='stloss.png'), trigger=(5000, 'iteration'))
    trainer.extend(extensions.PlotReport(['main/stacc', 'validation/main/stacc'],
                                         'epoch', file_name='stacc.png'), trigger=(5000, 'iteration'))


    # Save best models
    trainer.extend(snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger('validation/main/stloss', trigger=(5000, 'iteration')))

    trainer.extend(snapshot_object(model, 'model.acc.best'),
                   trigger=training.triggers.MaxValueTrigger('validation/main/stacc', trigger=(5000, 'iteration')))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(5000, 'iteration'))

    # epsilon decay in the optimizer
    """
    if args.opt == 'adadelta':
        if args.criterion == 'acc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/stacc',
                               lambda best_value, current_value: best_value > current_value,trigger=(5000, 'iteration')))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/stacc',
                               lambda best_value, current_value: best_value > current_value, trigger=(5000, 'iteration')))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/stloss',
                               lambda best_value, current_value: best_value < current_value, trigger=(5000, 'iteration')))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/stloss',
                               lambda best_value, current_value: best_value < current_value, trigger=(5000, 'iteration')))
    """
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/stloss', 'main/stacc','validation/main/stloss', 
                     'validation/main/stacc', 'main/asrloss', 'main/asracc', 
                      'main/mtloss',
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


def recog(args):
    """Decode with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    idim, src_vocab, trg_vocab, train_args = get_model_conf(args.model, os.path.join(os.path.dirname(args.model), 'model.json'))
    logging.info('reading model parameters from ' + args.model)
    st_model = None
    if args.st_model:
        st_model, _ = load_trained_model(args.st_model)
        assert isinstance(st_model, ASRInterface)
    else:
        st_model = None
  
    model = E2E(idim, src_vocab+1, trg_vocab, train_args, st_model=st_model,bias=True)
    torch_load(args.model, model)
    if args.st_model:
        del st_model
    assert isinstance(model, ASRInterface)
    model.recog_args = args

    if args.char_list is not None:
        char_list = []
        with open(args.char_list, 'rb') as f:
            dictionary = f.readlines()
        for entry in dictionary:
            entry = entry.decode('utf-8').split(' ')
            word = entry[0]
            char_list.append(word)
    else:
        char_list = train_args.char_list
    char_list.append("<blank>")

    # read rnnlm
    rnnlm = None
    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}
    results = []
    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})
    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)[0][0]
                if args.streaming_mode == 'window':
                    logging.info('Using streaming recognizer with window size %d frames', args.streaming_window)
                    se2e = WindowStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    for i in range(0, feat.shape[0], args.streaming_window):
                        logging.info('Feeding frames %d - %d', i, i + args.streaming_window)
                        se2e.accept_input(feat[i:i + args.streaming_window])
                    logging.info('Running offline attention decoder')
                    se2e.decode_with_attention_offline()
                    logging.info('Offline attention decoder finished')
                    nbest_hyps = se2e.retrieve_recognition()
                elif args.streaming_mode == 'segment':
                    logging.info('Using streaming recognizer with threshold value %d', args.streaming_min_blank_dur)
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({'yseq': [], 'score': 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i:i + r])
                        if hyps is not None:
                            text = ''.join([char_list[int(x)]
                                            for x in hyps[0]['yseq'][1:-1] if int(x) != -1])
                            text = text.replace('\u2581', ' ').strip()  # for SentencePiece
                            text = text.replace(model.space, ' ')
                            text = text.replace(model.blank, '')
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]['yseq'].extend(hyps[n]['yseq'])
                                nbest_hyps[n]['score'] += hyps[n]['score']
                else:
                    nbest_hyps = model.recognize(feat, args, char_list, rnnlm)
                results.append(nbest_hyps[0])
                new_js[name] = add_results_to_json(js[name], nbest_hyps[0].cpu(), char_list)

    else:
        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data
        keys = list(js.keys())
        feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
        sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
        keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = load_inputs_and_targets(batch)[0]
                nbest_hyps = model.recognize_batch(feats, args, train_args.char_list, rnnlm=rnnlm)

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)
    with open(args.result_label +".txt", 'w', encoding='utf-8') as f:
        for r in results:
            f.write(' '.join(list(map(str, np.asarray(r.cpu()))))+"\n")


    with open(args.result_label + '.json', 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))


def enhance(args):
    """Dumping enhanced speech and mask

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, ASRInterface)
    torch_load(args.model, model)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=None  # Apply pre_process in outer func
    )
    if args.batchsize == 0:
        args.batchsize = 1

    # Creates writers for outputs from the network
    if args.enh_wspecifier is not None:
        enh_writer = FileWriterWrapper(args.enh_wspecifier,
                                       filetype=args.enh_filetype)
    else:
        enh_writer = None

    # Creates a Transformation instance
    preprocess_conf = (
        train_args.preprocess_conf if args.preprocess_conf is None
        else args.preprocess_conf)
    if preprocess_conf is not None:
        logging.info('Use preprocessing'.format(preprocess_conf))
        transform = Transformation(preprocess_conf)
    else:
        transform = None

    # Creates a IStft instance
    istft = None
    frame_shift = args.istft_n_shift  # Used for plot the spectrogram
    if args.apply_istft:
        if preprocess_conf is not None:
            # Read the conffile and find stft setting
            with open(preprocess_conf) as f:
                # Json format: e.g.
                #    {"process": [{"type": "stft",
                #                  "win_length": 400,
                #                  "n_fft": 512, "n_shift": 160,
                #                  "window": "han"},
                #                 {"type": "foo", ...}, ...]}
                conf = json.load(f)
                assert 'process' in conf, conf
                # Find stft setting
                for p in conf['process']:
                    if p['type'] == 'stft':
                        istft = IStft(win_length=p['win_length'],
                                      n_shift=p['n_shift'],
                                      window=p.get('window', 'hann'))
                        logging.info('stft is found in {}. '
                                     'Setting istft config from it\n{}'
                                     .format(preprocess_conf, istft))
                        frame_shift = p['n_shift']
                        break
        if istft is None:
            # Set from command line arguments
            istft = IStft(win_length=args.istft_win_length,
                          n_shift=args.istft_n_shift,
                          window=args.istft_window)
            logging.info('Setting istft config from the command line args\n{}'
                         .format(istft))

    # sort data
    keys = list(js.keys())
    feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
    sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
    keys = [keys[i] for i in sorted_index]

    def grouper(n, iterable, fillvalue=None):
        kargs = [iter(iterable)] * n
        return zip_longest(*kargs, fillvalue=fillvalue)

    num_images = 0
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    for names in grouper(args.batchsize, keys, None):
        batch = [(name, js[name]) for name in names]

        # May be in time region: (Batch, [Time, Channel])
        org_feats = load_inputs_and_targets(batch)[0]
        if transform is not None:
            # May be in time-freq region: : (Batch, [Time, Channel, Freq])
            feats = transform(org_feats, train=False)
        else:
            feats = org_feats

        with torch.no_grad():
            enhanced, mask, ilens = model.enhance(feats)

        for idx, name in enumerate(names):
            # Assuming mask, feats : [Batch, Time, Channel. Freq]
            #          enhanced    : [Batch, Time, Freq]
            enh = enhanced[idx][:ilens[idx]]
            mas = mask[idx][:ilens[idx]]
            feat = feats[idx]

            # Plot spectrogram
            if args.image_dir is not None and num_images < args.num_images:
                import matplotlib.pyplot as plt
                num_images += 1
                ref_ch = 0

                plt.figure(figsize=(20, 10))
                plt.subplot(4, 1, 1)
                plt.title('Mask [ref={}ch]'.format(ref_ch))
                plot_spectrogram(plt, mas[:, ref_ch].T, fs=args.fs,
                                 mode='linear', frame_shift=frame_shift,
                                 bottom=False, labelbottom=False)

                plt.subplot(4, 1, 2)
                plt.title('Noisy speech [ref={}ch]'.format(ref_ch))
                plot_spectrogram(plt, feat[:, ref_ch].T, fs=args.fs,
                                 mode='db', frame_shift=frame_shift,
                                 bottom=False, labelbottom=False)

                plt.subplot(4, 1, 3)
                plt.title('Masked speech [ref={}ch]'.format(ref_ch))
                plot_spectrogram(
                    plt, (feat[:, ref_ch] * mas[:, ref_ch]).T,
                    frame_shift=frame_shift,
                    fs=args.fs, mode='db', bottom=False, labelbottom=False)

                plt.subplot(4, 1, 4)
                plt.title('Enhanced speech')
                plot_spectrogram(plt, enh.T, fs=args.fs,
                                 mode='db', frame_shift=frame_shift)

                plt.savefig(os.path.join(args.image_dir, name + '.png'))
                plt.clf()

            # Write enhanced wave files
            if enh_writer is not None:
                if istft is not None:
                    enh = istft(enh)
                else:
                    enh = enh

                if args.keep_length:
                    if len(org_feats[idx]) < len(enh):
                        # Truncate the frames added by stft padding
                        enh = enh[:len(org_feats[idx])]
                    elif len(org_feats) > len(enh):
                        padwidth = [(0, (len(org_feats[idx]) - len(enh)))] \
                            + [(0, 0)] * (enh.ndim - 1)
                        enh = np.pad(enh, padwidth, mode='constant')

                if args.enh_filetype in ('sound', 'sound.hdf5'):
                    enh_writer[name] = (args.fs, enh)
                else:
                    # Hint: To dump stft_signal, mask or etc,
                    # enh_filetype='hdf5' might be convenient.
                    enh_writer[name] = enh

            if num_images >= args.num_images and enh_writer is None:
                logging.info('Breaking the process.')
                break

