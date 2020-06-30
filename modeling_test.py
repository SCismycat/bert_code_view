#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:30
# @Author  : Leslee

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import random
import re

import modeling
import six
import tensorflow as tf


class BertModelTest(tf.test.TestCase):

    class BertModelTester(object):

        def __init__(self,parent,batch_size=13,seq_length=7,is_training=True,
                     use_input_mask=True,use_token_type_ids=True,vocab_size=99,
                     hidden_size=32,num_hidden_layers=5, num_attention_heads=4,
                     intermediate_size=37,hidden_act="gelu",hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,max_position_embeddings=512,
                     type_vocab_size=16,initializer_range=0.02,scope=None):

            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.use_token_type_ids = use_token_type_ids
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.scope = scope

        def create_model(self):

            input_ids = BertModelTest.id_tensor([self.batch_size, self.seq_length], vocab_size=2)

            input_mask = None

            if self.use_input_mask:
                input_mask = BertModelTest.ids_tensor(
                    [self.batch_size,self.seq_length], vocab_size=2)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = BertModelTest.ids_tensor(
                    [self.batch_size, self.seq_length],vocab_size=2)

            config = modeling.BertConfig(
                vocab_size=self.vocab_size,hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,type_vocab_size=self.type_vocab_size,
                initializer_range=self.initializer_range)

            model = modeling.BertModel(
                config=config,is_training=self.is_training, input_ids=input_ids,
                input_mask=input_mask, token_type_ids=token_type_ids,scope=self.scope)

            outputs = {
                "embedding_output": model.get_embedding_output(),
                "sequence_output": model.get_sequence_output(),
                "pooled_output": model.get_pooled_output(),
                "all_encoder_layers": model.get_all_encoder_layers(),
            }
            return outputs

        def check_output(self,result):

            self.parent.assertAllEqual(
                result["embedding_output"].shape, [self.batch_size, self.seq_length,self.hidden_size]
            )

            self.parent.assertAllEqual(
                result["sequence_output"].shape, [self.batch_size, self.seq_length, self.hidden_size]
            )

            self.parent.assertAllEqual(
                result["pooled_output"].shape, [self.batch_size, self.hidden_size]
            )





























