#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:05:33 2019

@author: assiene
"""

import torch
import torchtext
import torchtext.data as data

wikihop_train = data.TabularDataset(path='wikihop/train.json', format='json',
                                fields={"query": ("query", data.Field(sequential=False)),
                                        "answer": ("answer", data.Field(sequential=False)),
                                        "candidates": ("candidates", data.Field(sequential=False)),
                                        "supports": ("supports", data.Field(sequential=False))})

train_iter = data.BucketIterator(dataset=wikihop_train, batch_size=32)