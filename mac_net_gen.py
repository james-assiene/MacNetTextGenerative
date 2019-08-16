#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:01:04 2019

@author: assiene
"""

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms

from pytorch_pretrained_bert import BertConfig

from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.utils import padded_3d
from parlai.core.logs import TensorboardLogger

from parlai.core.utils import (
    AttrDict,
    argsort,
    padded_tensor,
    warn_once,
    round_sigfigs,
    fp16_optimizer_wrapper,
)

from .MacNetwork import MacNetwork
from .bert_dictionary import BertDictionaryAgent

from tensorboardX import SummaryWriter

losses = []

class MacNetAgent(TorchRankerAgent):
    
    @staticmethod    
    def add_cmdline_args(argparser):
        TorchRankerAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group("MacNet Arguments")
        
        agent.add_argument("-dim", "--dimension", type=int, default=512, help="Dimension for all layers")
        agent.add_argument("-nrh", "--num-reasoning-hops", type=int, default=12, help="Number of reasoning hops")
            
        MacNetAgent.dictionary_class().add_cmdline_args(argparser)
        
        return agent
    
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        
        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)
            
         # default one does not average
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)
        torch.autograd.set_detect_anomaly(True)
        
        
        
    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent
        
            
    def vectorize(self, *args, **kwargs):
        """Override options in vectorize from parent."""
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        #kwargs['split_lines'] = True
        obs = super().vectorize(*args, **kwargs)
        
        if "supports" not in obs:
            return obs
        
        obs["support"] = '\n'.join(obs['supports'])
        obs["support_vec"] = self._vectorize_text(obs["support"], kwargs["add_start"], kwargs["add_end"], kwargs["text_truncate"])
        obs["query_vec"] = self._vectorize_text(obs["query"], kwargs["add_start"], kwargs["add_end"], kwargs["text_truncate"]) 
        
        return obs
    
    def batchify(self, obs_batch, sort=False):
        
        batch = super().batchify(obs_batch, sort)
        
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs, x_lens = None, None
        if any('support_vec' in ex for ex in exs):
            _xs = [ex.get('support_vec', self.EMPTY) for ex in exs]
            xs, x_lens = padded_tensor(
                _xs, self.NULL_IDX, self.use_cuda, fp16friendly=self.opt.get('fp16')
            )
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )
                
        qs, q_lens = None, None
        if any('query_vec' in ex for ex in exs):
            _qs = [ex.get('query_vec', self.EMPTY) for ex in exs]
            qs, q_lens = padded_tensor(
                _qs, self.NULL_IDX, self.use_cuda, fp16friendly=self.opt.get('fp16')
            )
            if sort:
                sort = False  # now we won't sort on labels
                qs, q_lens, valid_inds, exs = argsort(
                    q_lens, qs, q_lens, valid_inds, exs, descending=True
                )
                
        batch.query_vec = qs
        batch.query_lengths = q_lens
        batch.supports_vec = xs
        batch.supports_lengths = x_lens
        
        return batch
    
    
    def share(self):
        shared = super().share()
        shared['model'] = self.model
        shared['writer'] = self.writer
        return shared    
    
    def train_step(self, batch):
        
        self.model.init_hidden(len(batch.observations))
        out = super().train_step(batch)
        

        for mac_cell in self.model.mac_cells:
            for name_in, param_in in mac_cell.named_parameters():
                self.writer.add_histogram(name_in, param_in.clone().cpu().detach().data.numpy(), self.batch_iter)
                #self.writer.add_histogram(name_in + "_grad", param_in.grad.clone().cpu().data.numpy(), self.batch_iter)
        
        return out
        
    
    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """This function takes in a Batch object as well as a Tensor of
        candidate vectors. It must return a list of scores corresponding to
        the likelihood that the candidate vector at that index is the
        proper response. If `cand_encs` is not None (when we cache the
        encoding of the candidate vectors), you may use these instead of
        calling self.model on `cand_vecs`.
        """
        
        self.model.init_hidden(len(batch.observations))
        
        if self.on_text == False:
            image_preprocessing = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            images = torch.stack([image_preprocessing(img) for img in batch.image])
            contexts = images
            
        else:
            contexts = batch.supports_vec
               
        questions = batch.query_vec
        
        out = self.model(contexts, questions) # batch_size x hidden_size=768
        
        with torch.no_grad():
            batch_size, num_candidates, max_cand_len = cand_vecs.shape
            last_hidden_state, _ = self.model.input_unit.bert_model(cand_vecs.reshape(-1, max_cand_len)) #batch_size * num_candidates x max_cand_len x hidden_size=768
            label_candidates_encoded = last_hidden_state.reshape(batch_size, num_candidates, max_cand_len, 768).mean(dim=2) # batch_size x num_candidates x hidden_size
        
        
        scores = label_candidates_encoded.bmm(out.unsqueeze(2)).squeeze(2)
        
        return scores
        
    
    def build_model(self):
        """This function is required to build the model and assign to the
        object `self.model`.
        """
        
        self.vocab_size = 645575  
        self.n_labels = 100
        self.learning_rate = 1e-4
        self.batch_size = self.opt["batchsize"]
        self.num_reasoning_hops = self.opt["num_reasoning_hops"]
        self.max_seq_length = 512
        self.on_text = True
        self.batch_iter = 0
        
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MacNetwork(self.device, vocab_size=self.vocab_size, n_labels=self.n_labels, batch_size=self.batch_size, p=self.num_reasoning_hops)
        self.model.share_memory()
        self.writer = SummaryWriter()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model = self.model.to(self.device)
        
    def eval_step(self, batch):
        
        self.model.init_hidden(len(batch.observations))
        
        return super().eval_step(batch)
        
    
    
