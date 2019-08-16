#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:05:09 2019

@author: assiene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MacCell(nn.Module):
    
    def __init__(self, device, d=512):
        
        super(MacCell, self).__init__()
        self.d = d # Dimension of control and memory states
        self.S = None # Number of words in the question
        self.p = None # Number of reasoning steps
        self.device = device
        
        self.q = None
        self.cws = None
        self.K = None
        self.M_past = None
        self.C_past = None
        
        self.control_qi = nn.Linear(1 * self.d, self.d) # 2 in original implementation (bi-lstm on the question). 1 because of BERT (mean on the sequence length axis)
        self.cqi_linear = nn.Linear(2 * self.d, self.d)
        self.cais_linear = nn.Linear(self.d, 1)
        
        self.memory_read_m = nn.Linear(self.d, self.d)
        self.memory_read_k = nn.Linear(self.d, self.d)
        self.memory_read_ik = nn.Linear(2 * self.d, self.d)
        self.memory_read_ci_i = nn.Linear(self.d, self.d)
        
        self.memory_write_ri_mi_1 = nn.Linear(2 * self.d, self.d)
        self.memory_write_ci_cj = nn.Linear(self.d, 1)
        self.memory_write_mi_sa = nn.Linear(self.d, self.d)
        self.memory_write_mi_info = nn.Linear(self.d, self.d, bias=True)
        self.memory_write_ci = nn.Linear(self.d, 1, bias=True)
        
        
    def control(self, ci_1, q, cws):
        #ci_1, qi : batch_size x d
        #cws : batch_size x S x d
        qi = self.control_qi(q)
        cqi = self.cqi_linear(torch.cat([ci_1, qi], dim=1).to(self.device)) # batch_size x d
        cqi_cws = cqi.unsqueeze(dim=1) * cws # batch x S x d
        cai = self.cais_linear(cqi_cws)# batch x S x 1
        cvis = F.softmax(cai, dim=1) # batch x S x 1
        ci = cvis * cws # batch x S x d
        ci = ci.sum(dim=1)
        
        return ci
    
    def read(self, mi_1, K, ci):
        #K : batch x H x W x d or batch_size x num_text_chunks x max_seq_len=512 x d
        #mi_1 : batch x d
        #ci :batch x d
        
        Ii = self.memory_read_m(mi_1).unsqueeze_(1).unsqueeze_(1) * self.memory_read_k(K) # batch x H x W x d
        Ii_prime = self.memory_read_ik(torch.cat([Ii, K], dim=3).to(self.device)) # batch x H x W x d
        
        rai = self.memory_read_ci_i(ci.unsqueeze(1).unsqueeze_(1) * Ii_prime) # batch x H x W x d
        
        rvi = F.softmax(rai, dim=1) # batch x H x W x d
        rvi = F.softmax(rvi, dim=2) # batch x H x W x d
        
        ri = (rvi * K).sum(dim=1).sum(dim=1) # batch x d
        
        return ri
        
    
    def write(self, ri, mi_1, C_past, M_past, ci):
        
        #ri : batch x d
        #mi_1 : batch x d
        #C : batch x p x d
        #M : batch x p x d
        #ci : batch x d
        
        mi_info = self.memory_write_ri_mi_1(torch.cat([ri, mi_1], dim=1).to(self.device))
        
        #C_past = C[:,:i,:] # batch x i x d
        #M_past = M[:,:i,:] # batch x i x d
        ci_cj = ci.unsqueeze(1) * C_past # batch x i x d
        saij = F.softmax(self.memory_write_ci_cj(ci_cj), dim=1) # batch x i x 1
        mi_sa = (saij * M_past).sum(dim=1) # bacth x d
        mi_prime = self.memory_write_mi_sa(mi_sa) + self.memory_write_mi_info(mi_info) # batch x d
        ci_prime = self.memory_write_ci(ci) # batch x 1
        
        m_weight = F.sigmoid(ci_prime)
        
        mi = m_weight * mi_1 + (1 - m_weight) * mi_prime
        
        return mi
    
    def forward(self, ci_1,  mi_1):
        ci = self.control(ci_1, self.q, self.cws)
        ri = self.read(mi_1, self.K, ci)
        mi = self.write(ri, mi_1, self.C_past, self.M_past, ci)
        
        return ci, mi