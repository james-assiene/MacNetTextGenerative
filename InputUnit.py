#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:33:55 2019

@author: assiene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

class InputUnit(nn.Module):
    
    def __init__(self, device, vocab_size, on_text=True, max_seq_len=512, batch_size=2, use_bert_encoder_for_question=True, d=512):
        
        super(InputUnit, self).__init__()
        self.d = d # Dimension of control and memory states
        self.S = None # Number of words in the question
        self.p = None # Number of reasoning steps
        self.vocab_size = vocab_size
        self.on_text = on_text
        self.max_seq_len = max_seq_len
        self.max_question_length = 0
        self.num_text_chunks = 35
        self.device = device
        self.CLS_index = 101
        self.SEP_index = 102
        
        self.use_bert_encoder_for_question = use_bert_encoder_for_question
        
        self.question_encoder = nn.LSTM(input_size=self.d, hidden_size=self.d, bidirectional=True)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d)
        
        if self.use_bert_encoder_for_question == False:
            self.cws_projection = nn.Linear(self.d * 2, self.d)
            
        else:
            self.cws_projection = nn.Linear(768, self.d)
        
        if self.on_text == False:
            self.build_img_encoder()
            
        else:
            self.build_text_encoder()
        
        
    
    def forward(self, context, question):                
        
        if self.use_bert_encoder_for_question == False:
            question = self.embedding_layer(question)
            cws, (q, _) = self.question_encoder(question.transpose(0,1))
            q = q.transpose(0,1) # batch x 2 x d
            q = q.reshape(q.shape[0], -1) # batch x 2d
            cws = cws.transpose(0,1)
            
        else:
            with torch.no_grad():
                question = F.pad(input=question, pad=(1,1)) # batch_size x max_seq_len
                question[:,0] = self.CLS_index
                question[:,-1] = self.SEP_index
                cws, _ = self.bert_model(question) #batch_size x max_question_length x hidden_size=768
        
        cws = self.cws_projection(cws) # batch x S x d
        q = cws.mean(dim=1) # batch_size x d
        label_candidates_encoded = None
        
        print("Encoding text...")
        K = context
        if self.on_text == False:
            K = self.resnet101(K)
            
        else:
            K, label_candidates_encoded = self.encode_text(K)
        
        K = self.context_encoder(K) #batch_size x num_text_chunks x max_seq_len=512 x d
        
        if self.on_text == False:
            K = K.transpose(1,2).transpose(2,3) # batch x h x w x d
            
        print("Done")
        
        return K, q, cws, label_candidates_encoded
    
    def build_img_encoder(self):
        self.resnet101 = models.resnet101(pretrained=True)
        modules = list(self.resnet101.children())[:-3]
        self.resnet101 = nn.Sequential(*modules)
        for p in self.resnet101.parameters():
            p.requires_grad = False
            
        self.context_encoder = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=self.d, kernel_size=3),
                nn.ELU(),
                nn.Conv2d(in_channels=self.d, out_channels=self.d, kernel_size=3),
                nn.ELU())
        
    def build_text_encoder(self):
        self.bert_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', pretrained_model_name_or_path="bert-base-uncased").to(self.device)
        self.context_encoder = nn.Linear(in_features=768, out_features=self.d) #768 : hidden_size of transformer
        
        
    def encode_text(self, contexts):
        
        self.bert_model.eval()
        
        contexts_splitted = list(contexts.split(self.max_seq_len - 2, 1)) # 2 for [CLS] and [SEP] ; list(num_text_chunks) x batch_size x 510
        last_context = torch.zeros((contexts.shape[0], self.max_seq_len - 2), dtype=torch.long).to(self.device) # batch_size x 510
        last_context[:,:contexts_splitted[-1].shape[1]] = contexts_splitted[-1]
        contexts_splitted[-1] = last_context
        context = torch.stack(contexts_splitted).transpose(0,1) # batch_size x num_text_chunk x 510
        context = F.pad(input=context, pad=(1,1)) # batch_size x num_text_chunks x max_seq_len
        context[:,:,0] = self.CLS_index
        context[:,:,-1] = self.SEP_index
        
        self.num_text_chunks = context.shape[1]
        
        with torch.no_grad():
            last_hidden_state, _ = self.bert_model(context.reshape(-1, self.max_seq_len)) #1 x sequence_length=512 x hidden_size=768
            encoded_text = last_hidden_state.reshape(-1, self.num_text_chunks, self.max_seq_len, 768) # batch_size x num_text_chunks x max_seq_len=512 x hidden_size=768
        
        
        return encoded_text, None
    