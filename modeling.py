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
                 config,is_training,
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
        if not is_training:
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

                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output, attention_mask=attention_mask,
                    hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act), hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            # pooler 将encoder的sequence tensor 从[batch_size, seq_len, hidden_size] 转换为'
            # [batch_size,hidden_size]. 对一些句子对分类任务比较友好。
            with tf.variable_scope("pooler"):
                #  取句子的第一个token的隐藏层状态作为语义表示
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor, config.hidden_size, activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        # 获取encoder的最后一个隐藏层。
        # 返回size： [batch_size, seq_length, hidden_size]
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table




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
    from_shape = get_shape_list(from_tensor,expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask,expected_rank=2)
    to_seq_length = to_shape[1]
    # tf.cast：改变tensor的数据类型
    to_mask = tf.cast(tf.reshape(to_mask, [batch_size,1,to_seq_length]), tf.float32)

    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length,1], dtype=tf.float32)

    mask = broadcast_ones * to_mask
    return mask





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


def gelu(x):

    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def get_activation(activation_string):
    # 将指定的一个string 转换为 对应的激活函数
    if not isinstance(activation_string,six.string_types):
        return activation_string
    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == 'relu':
        return tf.nn.relu
    elif act == 'gelu':
        return gelu
    elif act == 'tanh':
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):

    initialized_variable_names = {}
    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)



def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads =1,
                    size_pre_head=512,
                    query_act =None,
                    key_act = None,
                    value_act = None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    '''
    如果from_tensor和to_tensor是相同的，name就是self-attention. 每一个时间步，from_tensor都会和to_tensor进行关联计算，
    即计算 点积，得到一个向量。

    这里首先把 "from_tensor"复制为 query tensor，把"to_tensor"转换为 key tensor 和 value tensor。每个tensor都是
    [batch_size,seq_len, size_per_head].然后query和key进行点积归一化（这里用softmax作为概率结果），最后value将这些概率乘起来
    作为概率加权，然后返回结果。
    :param from_tensor: [batch_size, from_seq_length, from_width]
    :param to_tensor: [batch_size, to_seq_length, to_width]
    :param attention_mask:[batch_size, from_seq_length, to_seq_length],值是1或0，对为0的
    位置设置为-infinity,对为1的位置设置不变。
    :param num_attention_heads: 注意力头的数量
    :param size_pre_head: 每个注意力 head的size
    :param query_act: query的激活函数
    :param key_act: key的激活函数
    :param value_act: value的激活函数
    :param attention_probs_dropout_prob: attention概率的dropout概率
    :param initializer_range: 权重初始化范围
    :param do_return_2d_tensor: 如果为True，返回[batch_size * from_seq_length,num_attention_heads*size_per_head]
    如果是False，返回[batch_size, from_seq_length,num_attention_heads*size_per_heads]
    :param batch_size:如果输入是2D的，则batch_size是3D版本中的batch——size
    :param from_seq_length:from_tensor 的seq len
    :param to_seq_length:
    :return:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).
    '''
    def transpose_for_scores(input_tensor, batch_size,
                             num_attention_heads,seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size,seq_length,num_attention_heads,width])
        output_tensor = tf.transpose(output_tensor,[0,2,1,3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2 ,3])
    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # 名词解释
    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)
    # query_layer = [B*F,N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_pre_head,
        activation= query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))
    # key_layer = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d, num_attention_heads * size_pre_head,
        activation=key_act, name="key", kernel_initializer=create_initializer(initializer_range))

    # value_layer = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d, num_attention_heads * size_pre_head,
        activation=key_act, name="value", kernel_initializer=create_initializer(initializer_range))

    # query_layer = [B, N,F,H]
    query_layer = transpose_for_scores(query_layer,batch_size,num_attention_heads,
                                       from_seq_length,size_pre_head)
    # key_layer = [B,N,T,H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length,size_pre_head)
    # 将query 和 key进行点乘，获得注意力分数
    # attention_scores = [B,N,F,T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0/math.sqrt(float(size_pre_head))) # 乘以一个分数

    if attention_mask is not None:
        # attention_mask = [B,1,F, T]
        attention_mask = tf.expand_dims(attention_mask,axis=[1])
        # 创建一个函数，对于参与注意力计算的参数，位置时0.0，对于mask掉的position，取 -10000.0
        adder = (1.0 - tf.cast(attention_mask,tf.float32)) * -10000.0
        # 在softmax前将分数与adder相加，实际上等同于删除所有的masked。
        attention_scores += adder

    # 取softmax; [B,N,F,T]
    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # 处理value -->value_layer = [B,T,N,H]
    value_layer = tf.reshape(value_layer,
                             [batch_size,to_seq_length,num_attention_heads,size_pre_head])
    # 转置
    value_layer = tf.transpose(value_layer, [0,2,1,3])
    # 乘以上下文注意力权重
    # context_layer = [B,N,F,H]
    context_layer = tf.matmul(attention_probs,value_layer)

    # 再转置
    context_layer = tf.transpose(context_layer,[0,2,1,3])

    if do_return_2d_tensor:
        # context_layer = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer, [batch_size*from_seq_length, num_attention_heads*size_pre_head])
    else:
        # 拆成 [B,F,N*H]
        context_layer = tf.reshape(context_layer,
                                   [batch_size,from_seq_length,num_attention_heads*size_pre_head])
    return context_layer

def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob = 0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range = 0.02,
                      do_return_all_layers=False
                      ):
    '''

    :param input_tensor: tensor。 Shape=[batch_size, seq_length, hidden_size]
    :param attention_mask: [batch_size, seq_len, seq_len]，当该位置为1时，表名需要注意力关注，否则为0时不需要。
    :param hidden_size: Transformer的隐藏层尺寸
    :param num_hidden_layers: 隐藏层数量
    :param num_attention_heads: 注意力头的数量
    :param intermediate_size: 前馈神经网络的size
    :param intermediate_act_fn: 应用于前馈输出层或者中间层的非线性激活函数
    :param hidden_dropout_prob: 隐藏层的Dropout
    :param attention_probs_dropout_prob: attention的dropout概率
    :param initializer_range:初始化的概率范围
    :param do_return_all_layers: 决定是否所有层都返回结果，还是只在最后一层返回结果
    :return:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.
    '''


    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # Transformer需要在所有的层上进行残差和加和运算，所以输入必须跟隐藏层size相同。
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))
    # 这里reshape成2D，因为使用TPU运算的时候不方便来回切换shape，但是对GPU不存在这样的性能差
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor= layer_input, to_tensor=layer_input,attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads, size_pre_head=attention_head_size,
                        attention_probs_dropout_prob = attention_probs_dropout_prob,
                        initializer_range=initializer_range,do_return_2d_tensor=True,batch_size=batch_size,
                        from_seq_length=seq_length, to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # 将所有的sequences concatenate起来作为输出。
                    attention_output = tf.concat(attention_heads, axis=-1)

                # 线性投影层和残差网络
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,hidden_size,kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)
            # 该激活函数只应用于intermediate这个隐藏层
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,intermediate_size,
                    activation=intermediate_act_fn, kernel_initializer=create_initializer(initializer_range))
            # 向下投影回 ‘hidden_size’，并且添加残差
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output, hidden_size,kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output,hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_to_matrix(layer_output,input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_to_matrix(prev_output, input_shape)
        return final_output




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
        assert_rank(tensor,expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def reshape_to_matrix(input_tensor):
    # 将大于二维的 tensor 转为 2维的 Tensor
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor,[-1,width])
    return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])

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






















