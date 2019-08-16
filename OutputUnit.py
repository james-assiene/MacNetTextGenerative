#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:33:55 2019

@author: assiene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputUnit(nn.Module):
    
    def __init__(self, n_labels, d=512, score_candidates=True, hidden_size_candidate=768):
        
        super(OutputUnit, self).__init__()
        
        self.d = d
        
        self.linear1 = nn.Linear(2 * self.d, self.d)
        
        if score_candidates:
            self.linear2 = nn.Linear(self.d, hidden_size_candidate)
        else:
            self.linear2 = nn.Linear(self.d, n_labels)
        
    
    def forward(self, mp, q, label_candidates_encoded=None):
        
        #mp : batch x d
        #q : batch x d
        
        out = self.linear1(torch.cat([mp, q], dim=1))
        out = F.elu(out)
        out = self.linear2(out)
        
        return out