#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Nankai University (Chengyi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN Tandem Connectionist Encoding Network (pytorch)."""

from __future__ import division

import argparse
import logging
import math
import os
import pdb
from distutils.util import strtobool

import nltk

import chainer
import numpy as np
import six
import torch
from torch.nn import CrossEntropyLoss

from chainer import reporter
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy 
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.st_interface import STInterface
from pytorch_transformers import *


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss, acc):
        """Report at every step."""
        reporter.report({'loss': loss}, self)
        reporter.report({'acc': acc}, self)


class E2E(STInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param int mdim: dimension of intermediate outputs
    :param Namespace args: argument Namespace containing options
    :param E2E (ASRInterface) asr_model: pre-trained ASR model for encoder initialization
    :param E2E (MTInterface) mt_model: pre-trained NMT model for decoder initialization

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        return parser

    @property
    def attention_plot_class(self):
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=0, asr_model=None, mt_model=None):
        """Construct an E2E object."""
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.verbose = args.verbose
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.s2tdense = torch.nn.Linear(args.adim, 768)
        self.t2ddense = torch.nn.Linear(768, args.adim)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.pred = torch.nn.Linear(args.adim, odim)
        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.train()
        self.bert.train()
        # below means the last number becomes eos/os ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = -1
        self.subsample = [1]
        self.reporter = Reporter()

        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
                                               args.transformer_length_normalized_loss)
        self.reset_parameters(args)
        self.adim = args.adim
        self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)

        # multilingual related
        self.multilingual = getattr(args, "multilingual", False)
        self.replace_sos = args.replace_sos


        # initialize speech encoder from pretrained asr model
        if asr_model:
            param_dict = dict(asr_model.named_parameters())
            for n, p in self.named_parameters():
                asr_n = n.replace('senc', 'enc')
                if 'enc.enc' in asr_n or 'ctc' in asr_n:
                    if asr_n in param_dict.keys() and p.size() == param_dict[asr_n].size():
                        p.data = param_dict[asr_n].data
                        logging.warning('Overwrite %s from asr model' % n)

        # initialize text encoder and decoder from pretrained mt model
        if mt_model:
            param_dict = dict(mt_model.named_parameters())
            for n, p in self.named_parameters():
                mt_n = n.replace('tenc', 'enc')
                if 'dec' in mt_n or 'att' in mt_n:
                    if mt_n in param_dict.keys() and p.size() == param_dict[mt_n].size():
                        p.data = param_dict[mt_n].data
                        logging.warning('Overwrite %s from mt model' % n)

        # options for beam search
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = torch.zeros([1])

    def reset_parameters(self, args):
        # initialize parameters
        initialize(self, args.transformer_init)

    def target_language_biasing(self, xs_pad, ilens, ys_pad):
        """Prepend target language IDs to source sentences for multilingual NMT.

        These tags are prepended in source/target sentences as pre-processing.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: source text without language IDs
        :rtype: torch.Tensor
        :return: target text without language IDs
        :rtype: torch.Tensor
        :return: target language IDs
        :rtype: torch.Tensor (B, 1)
        """
        if self.multilingual:
            # remove language ID in the beggining
            tgt_lang_ids = ys_pad[:, 0].unsqueeze(1)
            xs_pad = xs_pad[:, 1:]  # remove source language IDs here
            ys_pad = ys_pad[:, 1:]

            # prepend target language ID to source sentences
            xs_pad = torch.cat([tgt_lang_ids, xs_pad], dim=1)
        return xs_pad, ys_pad
 
    def get_bert_mask(self, mask):
        bert_mask = mask.unsqueeze(1)
        bert_mask = bert_mask.to(dtype=next(self.parameters()).dtype)
        bert_mask = (1.0 - bert_mask) * -10000.0
        return bert_mask

    def speech_encoder_forward(self, xs_pad, ilens):
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        return hs_pad, hs_mask

    def decoder_forward(self, ys_pad, hs_pad, hs_mask):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        loss = self.criterion(pred_pad, ys_out_pad)
        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                          ignore_label=self.ignore_id)
        return loss, acc

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_asr=None):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        batch_size = xs_pad.size(0)
        # 1. forward encoder
        hs_pad, hs_mask = self.speech_encoder_forward(xs_pad, ilens)
        hs_len = hs_mask.view(batch_size, -1).sum(1)
        if ys_pad_asr is not None:
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad_asr)

        # bert encoder forward
        hs_pad = self.s2tdense(hs_pad)
        
        bert_mask = self.get_bert_mask(hs_mask)
        head_mask = [None] * self.bert.config.num_hidden_layers
        token_ids = torch.zeros([hs_pad.size(0), hs_pad.size(1)], dtype=torch.long, device=hs_pad.device)
        token_embeddings = self.bert.embeddings.token_type_embeddings(token_ids)
        hs_pad = hs_pad + token_embeddings

        # add position embedding
        enc_outputs = self.bert.encoder(hs_pad, bert_mask, head_mask = head_mask)
        hs_pad = self.t2ddense(enc_outputs[0])
        # decoder forward
        logits = self.pred(hs_pad)
        soft_logits = torch.softmax(logits)
        soft_logits = torch.mean(soft_logits, dim=1)
        
        pdb.set_trace()
        loss_st, acc = self.decoder_forward(ys_pad,  hs_pad, hs_mask)
                
        if ys_pad_asr is not None:
            y_lens = [len(y[y!=self.ignore_id]) for y in ys_pad_asr]
            ys_pad_asr = ys_pad_asr[:, :max(y_lens)]    #for data parallel
            txt_mask = (~make_pad_mask(y_lens)).to(ys_pad_asr.device).unsqueeze(-2)
            txt_bert_mask = self.get_bert_mask(txt_mask)
            txt_enc_outputs = self.bert(ys_pad_asr, txt_bert_mask, head_mask = head_mask)
            txt_hs_pad = self.t2ddense(txt_enc_outputs[0])
            loss_mt, _ = self.decoder_forward(ys_pad, txt_hs_pad, txt_mask)
        else:
            loss_mt = None
            
        # copyied from e2e_asr
        self.loss = loss
        loss_data = float(self.loss)
        self.acc[0] = acc
        self.reporter.report(loss_data, acc)
        #if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
        #    self.reporter.report(loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data)
        #else:
        #    logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.dec)

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 1. encoder
        hs, hlens, _ = self.senc(hs, ilens)
        hs, _, _ = self.tenc(hs, hlens)
        return hs.squeeze(0)

    def translate(self, x, trans_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace trans_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs = self.encode(x).unsqueeze(0)
        lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], lpz, trans_args, char_list, rnnlm)
        return y

    def translate_batch(self, xs, trans_args, char_list, rnnlm=None):
        """E2E beam search.

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace trans_args: argument Namespace containing options
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

        # 1. Encoder
        hs_pad, hlens, _ = self.senc(xs_pad, ilens)
        hs_pad, hlens, _ = self.tenc(hs_pad, hlens)
        lpz = None

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad, hlens, lpz, trans_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            # 1. Encoder
            if self.multilingual:
                tgt_lang_ids = ys_pad[:, 0:1]
                ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
            else:
                tgt_lang_ids = None
            hpad, hlens, _ = self.senc(xs_pad, ilens)
            hpad, hlens, _ = self.tenc(hpad, hlens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad, tgt_lang_ids=tgt_lang_ids)

        return att_ws

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen
