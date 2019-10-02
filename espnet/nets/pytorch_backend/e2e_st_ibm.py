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
from torch.nn import functional as F

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

    def __init__(self, idim, odim, args, asr_model=None, mt_model=None):
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
        self.xlm = XLMModel.from_pretrained('xlm-mlm-ende-1024')
        self.s2tdense = torch.nn.Linear(args.adim, 1024)
        self.t2ddense = torch.nn.Linear(1024, args.adim)
        #self.pred = torch.nn.Linear(args.adim, odim)
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
        # below means the last number becomes eos/os ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.pad_id = -1
        self.speech_id = 4
        self.txt_id = 5
        self.blank_id = 6
        self.subsample = [1]
        self.reporter = Reporter()
        """
        self.st_loss = args.st_loss
        self.mt_loss = args.mt_loss
        self.kd_loss = args.kd_loss
        self.ibm_loss = args.ibm_loss
        self.ctc_loss = args.ctc_loss
        """
        self.criterion = LabelSmoothingLoss(self.odim, self.pad_id, args.lsm_weight,
                                               False)
        self.reset_parameters(args)
        self.adim = args.adim
        #self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)

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
 
    def get_masks(self, slen, lengths):
        """"
        Generate hidden states mask, and optionally an attention mask
        """
        bs = lengths.size(0)
        assert lengths.max().item() <= slen
        alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
        mask = alen < lengths[:, None]
        assert mask.size() == (bs, slen)
        return mask

    def xlm_forward(self, x, xlens=None):
        bs = x.size(0)
        slen = x.size(1)
        if xlens is None:
            xlens = torch.ones((bs), device=x.device).fill_(slen).long()
        mask = self.get_masks(slen, xlens)
        langs = torch.ones((bs, slen), dtype=torch.long, device=x.device)
        x = x + self.xlm.lang_embeddings(langs)
        token_type_ids = torch.ones((bs, slen), dtype=torch.long, device=x.device).fill_(self.speech_id)
        x = x + self.xlm.embeddings(token_type_ids)
        x = self.xlm.layer_norm_emb(x)
        x = F.dropout(x, p=self.xlm.dropout, training=self.training)
        x *= mask.unsqueeze(-1).to(x.dtype)
    
        for i in range(self.xlm.n_layers):
            attn_outputs = self.xlm.attentions[i](x, mask, cache=None, head_mask=None)
            attn = attn_outputs[0]
            attn = F.dropout(attn, p=self.xlm.dropout, training=self.training)
            x = x + attn
            x = self.xlm.layer_norm1[i](x)

            x = x + self.xlm.ffns[i](x)
            x = self.xlm.layer_norm2[i](x)
            x *= mask.unsqueeze(-1).to(x.dtype)

        return x

    def speech_encoder_forward(self, xs_pad, ilens):
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        return hs_pad, hs_mask

    def decoder_forward(self, ys_pad, hs_pad, hs_mask):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.pad_id)
        ys_mask = target_mask(ys_in_pad, self.pad_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        loss = self.criterion(pred_pad, ys_out_pad)
        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                          ignore_label=self.pad_id)
        return loss, acc

    def ibm_loss(self, x, target, smoothing=0.1):
        """Compute the IBM loss"""
        batch_size = x.size(0)
        seq_len = target.size(1)
        size = x.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros([batch_size, seq_len, size], dtype=x.dtype, device=x.device)
            true_dist = true_dist.fill_(smoothing/(size-1))
            ignore = target == self.ignore_id
            true_length = torch.sum((1-ignore), dim=1)
            target = target.masked_fill(ignore, 0)
            true_dist = true_dist.view(-1, size)
            target = target.view(-1)
            true_dist.scatter_(1, target.unsqueeze(1), 1-smoothing)
            true_dist = true_dist.masked_fill(ignore.view(-1).unsqueeze(1), 0)
            true_dist = torch.sum(true_dist.view(batch_size, seq_len, size), dim=1)
            true_mean = true_dist / true_length.to(true_dist.dtype).unsqueeze(1)
            lfc = torch.nn.KLDivLoss(reduce=False)
            kl = lfc(torch.log(x), true_mean)
            loss = kl.sum() / batch_size
        return loss
            

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
        #if ys_pad_asr is not None:
        #    loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad_asr)

        # bert encoder forward
        hs_pad = self.s2tdense(hs_pad)
        enc_output = self.xlm_forward(hs_pad, hs_len)
 
        hs_pad = self.t2ddense(enc_output)
        # decoder forward
        #logits = self.pred(hs_pad)
        #soft_logits = torch.softmax(logits, dim=-1)
        #soft_logits = torch.mean(soft_logits, dim=1)

        #ibm_loss = self.ibm_loss(soft_logits, ys_pad)
        
        loss_st, acc = self.decoder_forward(ys_pad,  hs_pad, hs_mask)
        """        
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
        """    
        # copyied from e2e_asr
        self.loss = loss_st
        loss_data = float(self.loss)
        self.acc[0] = acc
        self.reporter.report(loss_data, acc)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_data,acc)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
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
        feat = torch.as_tensor(x).cuda().unsqueeze(0)
        enc_output, _ = self.encoder(feat, None)
        hs_pad = self.s2tdense(enc_output)
        hs_pad = self.xlm_forward(hs_pad)
        hs_pad = self.t2ddense(hs_pad)
        return hs_pad.squeeze(0)

    def recognize(self, feat, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace trans_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(feat).unsqueeze(0)
        h = enc_output.squeeze(0)

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}

        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1, device='cuda').unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).cuda().unsqueeze(0)
                # FIXME: jit does not match non-jit result
                local_att_scores = self.decoder.recognize(ys, ys_mask, enc_output)

                local_scores = local_att_scores

                local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            from espnet.nets.e2e_asr_common import end_detect
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(feat, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps
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
