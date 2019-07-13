#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division
import pdb
import argparse
import logging
import math
import os

import editdistance

import chainer
import numpy as np
import six
import torch

from itertools import groupby

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import PreNet
from espnet.nets.pytorch_backend.rnn.encoders import Encoder

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self, stloss, stacc, asrloss, ctcloss, cer_ctc, asracc, ppl):
        reporter.report({'stloss': stloss}, self)
        reporter.report({'stacc': stacc}, self)
        reporter.report({'asrloss': asrloss}, self)
        reporter.report({'cer_ctc': cer_ctc}, self)
        reporter.report({"ctcloss": ctcloss}, self)
        reporter.report({'asracc': asracc}, self)
        reporter.report({'ppl': ppl}, self)




class E2E(ASRInterface, torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    :param E2E (torch.nn.Module) asr_model: pre-trained ASR model for encoder initialization
    :param E2E (torch.nn.Module) mt_model: pre-trained NMT model for decoder initialization

    """

    def __init__(self, idim, odim, args, asr_model=None, mt_model=None, st_model=None):
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = 1
        self.eos = 2

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # speech translation related
        self.replace_sos = args.replace_sos   #replace


        self.frontend = None
        self.prenet = PreNet(idim, 2, args.eunits, args.dropout_rate)
        self.dropout_pre = torch.nn.Dropout(p=args.dropout_rate)
        self.linear_pre = torch.nn.Linear(args.eunits*2, args.eunits)
        self.ctc = ctc_for(args, odim)
        # encoder
        self.embed_src = torch.nn.Embedding(odim, args.eunits, padding_idx=self.eos, _weight=self.ctc.ctc_lo.weight)
        self.dropemb = torch.nn.Dropout(p=args.dropout_rate)
        self.enc = Encoder('blstm', args.eunits, 3, args.eunits, args.eprojs, subsample, dropout=args.dropout_rate)
        # attention
        self.att = att_for(args)
        # decoder
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        # weight initialization
        self.init_like_chainer()

        # pre-training w/ ASR encoder and NMT decoder
        if asr_model is not None:
            param_dict = dict(asr_model.named_parameters())
            for n, p in self.named_parameters():
                # overwrite the encoder
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    if 'enc.enc' in n:
                        p.data = param_dict[n].data
                        logging.warning('Overwrite %s' % n)
        if mt_model is not None:
            param_dict = dict(mt_model.named_parameters())
            for n, p in self.named_parameters():
                # overwrite the decoder
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    if 'dec.' in n or 'att' in n:
                        p.data = param_dict[n].data
                        logging.warning('Overwrite %s' % n)
        if st_model is not None:
            param_dict = dict(st_model.named_parameters())
            for n, p in self.named_parameters():
                enc_n = n.replace('prenet', 'enc')
                if enc_n in param_dict.keys() and p.size() == param_dict[enc_n].size():
                    p.data = param_dict[enc_n].data
                    logging.warning('Overwrite %s' % n)
                enc_n = n.replace('enc.0', 'enc.1')
                enc_n = enc_n.replace('l3', 'l4')
                enc_n = enc_n.replace('l2', 'l3')
                enc_n = enc_n.replace('l1', 'l2')
                enc_n = enc_n.replace('l0', 'l1')
                if 'enc.enc' in enc_n and enc_n in param_dict.keys() and p.size() == param_dict[enc_n].size():
                    p.data = param_dict[enc_n].data
                    logging.warning('Overwrite %s' % n)
            self.dec.embed.weight[:10000, :] = st_model.dec.embed.weight
            self.dec.embed.weight[10000, :] = st_model.dec.embed.weight[1, :]
            self.dec.embed.weight[10001, :] = st_model.dec.embed.weight[1, :]
            self.dec.output.weight[:10000, :] = st_model.dec.output.weight
            self.dec.output.weight[10000, :] = st_model.dec.output.weight[1, :]
            self.dec.output.weight[10001, :] = st_model.dec.output.weight[1, :]
            self.dec.output.bias[:10000] = st_model.dec.output.bias
            self.dec.output.bias[10000] = st_model.dec.output.bias[1]
            self.dec.output.bias[10001] = st_model.dec.output.bias[1]
        # options for beam search
        if 'report_cer' in vars(args) and (args.report_cer or args.report_wer):
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank,
                          'tgt_lang': False}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.stloss = None
        self.asrloss = None
        self.stacc = None
        self.asracc = None
        self.ctcloss = None
        self.ppl = None

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """

        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() in (3, 4):
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad, task="st"):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loass value
        :rtype: torch.Tensor
        """
        # 0. prenet
        if task == "st" or task == "asr":
            hs_pad, hlens, _ = self.prenet(xs_pad, ilens)
            hs_pad = self.dropout_pre(hs_pad)
            hs_pad = self.linear_pre(hs_pad)
            hs_pad = self.dropout_pre(hs_pad)
        else:
            hs_pad = self.dropemb(self.embed_src(xs_pad))
            hlens = ilens

        # 1. Encoder
        if self.replace_sos:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
        else:
            tgt_lang_ids = None

        if task == "asr" and args.mtlalpha > 0:
            loss_ctc = self.ctc(hs_pad, hlens, ys_pad)
        else:
            loss_ctc = None

        cs_pad, clens, _ = self.enc(hs_pad, hlens)


        loss_att, acc, ppl = self.dec(cs_pad, clens, ys_pad, tgt_lang_ids=tgt_lang_ids)

        # 4. compute cer without beam search
        if task == "asr":
            cers = []

            y_hats = self.ctc.argmax(hs_pad).data
            for i, y in enumerate(y_hats):
                y_hat = [x[0] for x in groupby(y)]
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.blank, '')
                seq_true_text = "".join(seq_true).replace(self.space, ' ')

                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                if len(ref_chars) > 0:
                    cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

            cer_ctc = sum(cers) / len(cers) if cers else None
        else:
            cer_ctc = 0


        alpha = self.mtlalpha
        if task == "st":
            self.stloss = float(loss_att)
            self.stacc = acc
            loss = loss_att
        elif task == "mt" or task == "mt":
            self.ppl = ppl
            loss = loss_att
        else:
            loss = alpha * loss_ctc + (1 - alpha) * loss_att
            self.asrloss = float(loss)
            self.ctcloss = float(loss_ctc)
            self.asracc = acc

        loss_data = float(loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(self.stloss, self.stacc, self.asrloss, self.ctcloss, cer_ctc, self.asracc, self.ppl)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return loss

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E beam search

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(hs, ilens)
            hs, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs, hlens = hs, ilens

        # 1. encoder
        hs, hlens, _ = self.prenet(hs, hlens)
        hs = self.dropout_pre(hs)
        hs, _, _ = self.enc(hs, hlens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs)[0]
        else:
            lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()

        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        """E2E beam search

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(xs_pad, ilens)
            hs_pad, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs_pad)
        else:
            lpz = None

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad, hlens, lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def enhance(self, xs):
        """Forwarding only the frontend stage

        :param ndarray xs: input acoustic feature (T, C, F)
        """

        if self.frontend is None:
            raise RuntimeError('Frontend does\'t exist')
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
            else:
                hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            if self.replace_sos:
                tgt_lang_ids = ys_pad[:, 0:1]
                ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
            else:
                tgt_lang_ids = None
            hpad, hlens, _ = self.enc(hs_pad, hlens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad, tgt_lang_ids=tgt_lang_ids)

        return att_ws

    def subsample_frames(self, x):
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

