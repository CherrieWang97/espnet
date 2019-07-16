#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division
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
from espnet.nets.pytorch_backend.rnn.multidec import Decoder
from espnet.nets.pytorch_backend.rnn.encoders import Encoder

CTC_LOSS_THRESHOLD = 10000



class Reporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self, stloss, stacc, asrloss, asracc, ppl):
        reporter.report({'stloss': stloss}, self)
        reporter.report({'stacc': stacc}, self)
        reporter.report({'asrloss': asrloss}, self)
        reporter.report({'asracc': asracc}, self)
        reporter.report({'ppl': ppl}, self)

class E2E(ASRInterface, torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs speech
    :param int odim: size of vocabulary
    :param Namespace args: argument Namespace containing options
    :param E2E (torch.nn.Module) asr_model: pre-trained ASR model for encoder initialization
    :param E2E (torch.nn.Module) mt_model: pre-trained NMT model for decoder initialization

    """

    def __init__(self, idim, vocab_size, args, asr_model=None, mt_model=None, st_model=None):
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.etype = args.etype   #speech encoder type
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.alpha = args.mtlalpha
        self.reporter = Reporter()

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = 1
        self.eos = 2

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)   #only use for speech encoder
        if args.etype.endswith("p") and not args.etype.startswith("vgg"):
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                 'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(vocab_size, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # speech translation related
        self.replace_sos = args.replace_sos

        self.frontend = None

        # encoder
        self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, None, args.dropout_rate)

        # attention
        self.srcatt = att_for(args)
        self.trgatt = att_for(args)
        # decoder
        self.srcdec = Decoder(args.eprojs, vocab_size, args.dtype, args.dlayers, args.dunits, self.sos, self.eos, self.srcatt,
                                args.verbose,
                                args.char_list, labeldist,
                                args.lsm_weight, args.sampling_probability, args.dropout_rate_decoder,
                                args.context_residual, args.replace_sos)
        if args.share_dict:
            embed = self.srcdec.embed
        else:
            embed = None
        self.linear = torch.nn.Linear(args.dunits * 2, args.eprojs) 
        self.trgdec = Decoder(args.eprojs, vocab_size, args.dtype, args.dlayers, args.dunits, self.sos, self.eos, self.trgatt,
                               args.verbose,
                                args.char_list, labeldist,
                                args.lsm_weight, args.sampling_probability, args.dropout_rate_decoder,
                                args.context_residual, args.replace_sos, embed=embed)

        # weight initialization
        self.init_like_chainer()
        if asr_model is not None:
            param_dict = dict(asr_model.named_parameters())
            for n, p in self.named_parameters():
                if 'enc.enc' in n and n in param_dict.keys() and p.size() == param_dict[n].size():
                    p.data = param_dict[n].data
                    logging.warning('Overwrite %s' % n)
                asr_n = n.lstrip('src')
                if asr_n in param_dict.keys() and p.size() == param_dict[asr_n].size():
                    if 'srcdec' in n or 'srcatt' in n:
                        p.data = param_dict[asr_n].data
                        logging.warning('Overwrite %s' % n)
           
        if mt_model is not None:
            param_dict = dict(mt_model.named_parameters())
            for n, p in self.named_parameters():
                mt_dec_n = n.lstrip('trg')
                if mt_dec_n in param_dict.keys() and p.size() == param_dict[mt_dec_n].size():
                    if 'trgdec' in n or 'trgatt.' in n:
                        p.data = param_dict[mt_dec_n].data
                        logging.warning('Overwrite %s' % n)
            #self.srcdec.embed.weight[:10000, :] = mt_model.dec.embed.weight
        if st_model is not None:
            param_dict = dict(st_model.named_parameters())
            for n, p in self.named_parameters():
                st_n = n.lstrip('s')
                if 'enc.enc' in n and st_n in param_dict.keys() and p.size() == param_dict[st_n].size():
                    p.data = param_dict[st_n].data
                    logging.warning('Overwrite %s' % n)
                st_n = n.lstrip('trg')
                if st_n in param_dict.keys() and p.size() == param_dict[st_n].size():
                    if 'trgdec' in n:
                        p.data = param_dict[st_n].data
                        logging.warning('Overwrite %s' % n)
                if st_n in param_dict.keys() and p.size() == param_dict[st_n].size():
                    p.data = param_dict[st_n].data
                    logging.warning('Overwrite %s' % n)

        # options for beam search
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.stloss = None
        self.asrloss = None
        self.stacc = None
        self.asracc = None
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
        self.srcdec.embed.weight.data.normal_(0, 1)
        self.trgdec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.srcdec.decoder)):
            set_forget_bias_to_one(self.srcdec.decoder[l].bias_ih)
        for l in six.moves.range(len(self.trgdec.decoder)):
            set_forget_bias_to_one(self.trgdec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad1, ys_pad2):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loass value
        :rtype: torch.Tensor
        """
        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        if self.replace_sos:
            src_lang_ids = ys_pad1[:, 0:1]
            ys_pad1 = ys_pad1[:, 1:]  # remove target language ID in the beggining
            tgt_lang_ids = ys_pad2[:, 0:1]
            ys_pad2 = ys_pad2[:, 1:]
        else:
            src_lang_ids = None
            tgt_lang_ids = None

        hs_pad, hlens, _ = self.enc(hs_pad, hlens)


        # 3. attention loss
        asrloss, asracc, zs, zlens = self.srcdec(hs_pad, hlens, ys_pad1, tgt_lang_ids=src_lang_ids)
        self.asracc = asracc
        self.asrloss = float(asrloss)
        zs = self.linear(zs)
        stloss, stacc, _, _ = self.trgdec(zs, zlens, ys_pad2, tgt_lang_ids=tgt_lang_ids)
        self.stacc = stacc
        self.stloss = float(stloss)
        loss = self.alpha * asrloss + (1 - self.alpha) * stloss
        if not math.isnan(self.stloss) and not math.isnan(self.asrloss):
            self.reporter.report(self.stloss, self.stacc, self.asrloss, self.asracc, self.ppl)
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
        hs, _, _ = self.enc(hs, hlens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs)[0]
        else:
            lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.trgdec.recognize_beam(hs[0], lpz, recog_args, char_list, rnnlm, strm_idx=1)

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
        y = self.trgdec.recognize_beam_batch(hs_pad, zlens, lpz, recog_args, char_list, rnnlm, strm_idx=1)

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
            att_ws = self.trgdec.calculate_all_attentions(hpad, hlens, ys_pad, tgt_lang_ids=tgt_lang_ids)

        return att_ws

    def subsample_frames(self, x):
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

