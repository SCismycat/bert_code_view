#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 15:56
# @Author  : Leslee
# Transformer 架构

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf

class BertConfig(object):

    def __init__(self,
                 vocab_size,hidden_size=768,
                 num_hidden_layers=12,num_attention_heads=12,
                 intermediate_size=3072,hidden_act='gelu',
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512, type_vocab_size=16,
                 initializer_range=0.02):
        '''

        :param vocab_size:  词表的size
        :param hidden_size:  编码层和pooler层的隐藏层大小
        :param num_hidden_layers:  Transformer 编码层的隐藏层层数
        :param num_attention_heads: 在Transformer的encoder阶段，每个attention层的 注意力头的数量
        :param intermediate_size: 某些中间状态的size 如(feed-forward,前馈网络)的size
        :param hidden_act: 编码和pooler时候的非线性激活函数
        :param hidden_dropout_prob: 隐藏层的dropout
        :param attention_probs_dropout_prob: attention层的dropout
        :param max_position_embeddings: 最大位置长度
        :param type_vocab_size:
        :param initializer_range:The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
        '''
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = intermediate_size

    @classmethod
    def from_dict(cls,json_object):
        config = BertConfig(vocab_size=None)
        for (key,value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config
    # cls代表类本身的标识符，就像self一样
    @classmethod
    def from_json_file(cls,json_file):
        with tf.gfile.GFile(json_file,"r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        "实例化为Python dict"
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2,sort_keys=True) + '\n'

class BertModel(object):



    def __init__(self,
                 config,is_traning,
                 input_ids, input_mask=None,
                 token_type_ids=None,use_one_hot_embeddings=False,
                 scope=None):
        '''

        :param config:  模型配置文件， jsonfile
        :param is_traning:  是否训练，bool
        :param input_ids: 输入的token对应的ids，int32
        :param input_mask:  输入的mask的tokens， int32
        :param token_type_ids: 输入的句子类型id，int32
        :param use_one_hot_embeddings: 是否使用one-hot编码
        :param scope: 参数域
        '''
        config = copy.deepcopy(config)
        if not is_traning:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size,seq_length],dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # 对输入token进行embedding
                '''
                输入是,原来的输入token ids，词表size大小，embedding的size就是隐藏层的大小，初始化范围
                embedding的名称，是否使用ont-hot编码
                '''
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",use_one_hot_embeddings=use_one_hot_embeddings)
                # 增加位置embeddings 和 token类型(句子类型)的id embeddings，然后使用dropout和layer norm

                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,use_token_type=True,
                    token_type_ids=token_type_ids, token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings = True, position_embedding_name = "position_embeddings",
                    initializer_range = config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,dropout_prob = config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # 这一步将2D的输入:[batch_size,seq_length] 转成3D的[batch_size,seq_length,seq_length]
                # 然后用于计算 attention分数。
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)







##
def embedding_lookup(input_ids,vocab_size,
                     embedding_size=128,initializer_range=0.02,
                     word_embedding_name="word_embeddings",use_one_hot_embeddings=False):
    '''
    embedding层对输入的tokens ids进行embeddings;本函数假设输入维度是三维的，即[batch_size,seq_len,num_inputs]
    如果传入是2d的，就reshape成[batch_size,seq_len,1]
    :param input_ids: 输入的word 或tokens的ids，shape = [batch_size, seq_length]
    :param vocab_size: 词表的大小
    :param embedding_size: embedding的shape
    :param initializer_range: 初始化的范围
    :param word_embedding_name:
    :param use_one_hot_embeddings:
    :return:
        float Tensor: [batch_size,seq_length,embedding_size]
    '''
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids,axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer= create_initializer(initializer_range))
    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table,flat_input_ids)

    input_shape = get_shape_list(input_ids)
    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)

def embedding_postprocessor(input_tensor,use_token_type=False,
                            token_type_ids=None,token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    '''
    对词嵌入张量执行各种后处理
    :param input_tensor: [batch_size, seq_length, embedding_size]
    :param use_token_type: 是否增加token_ids embeddings
    :param token_type_ids: 如果增加，则需要传入token type的id列表。[batch_size, seq_len]
    :param token_type_vocab_size: token type的vocab size
    :param token_type_embedding_name:
    :param use_position_embeddings:
    :param position_embedding_name:
    :param initializer_range:
    :param max_position_embeddings:
    :param dropout_prob:
    :return:
    和input_tokens一样的维度
    '''
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor
    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
            name= token_type_embedding_name,
            shape=[token_type_vocab_size,width],
            initializer= create_initializer(initializer_range))

        flat_token_type_ids = tf.reshape(token_type_ids,[-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth= token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings
    # tf.control_dependencies()是用来控制计算流图的，给图中的某些节点指定计算的顺序。
    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape= [max_position_embeddings, width],
                initializer=create_initializer(initializer_range))
            # 这里的position embeddings参数是需要学习的参数，创建一个序列长度为 max_len的位置序列。
            # 但是一般情况下都会小于这个max_len，所以，用句子长度来切片
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())
            # 只有最后两个维度是相关的，即seq_length 和 width。所以，将shape改成(batch_size,seq_length,width)
            position_broadcast_shape = []
            for _ in range(num_dims -2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings
    output = layer_norm_and_dropout(output,dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor,to_mask):
    '''
    从一个2D的tensor 创建一个3D的attention mask
    :param from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    :param to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    :return:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    '''







def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output

def layer_norm(input_tensor,name=None):
    '''
    对输入进行layer norm
    :param input_tensor: 全部完成后的tensor
    :param name:
    :return:
    '''
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def layer_norm_and_dropout(input_tensor,dropout_prob,name=None):
    """
    对结果进行批归一化和dropout
    :param input_tensor:
    :param dropout_prob:
    :param name:
    :return:
    """
    output_tensor = layer_norm(input_tensor,name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor
















def get_shape_list(tensor, expected_rank=None,name=None):
    '''
    返回一个tensor的shape的List。
    :param tensor:  需要获取对象的tf.Tensor对象，
    :param expected_rank:
    :param name:
    :return:
    '''

    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert



def assert_rank(tensor, expected_rank, name=None):

    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims # 维度
    if  actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))






















