#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 16:47
# @Author  : Leslee
from __future__ import absolute_import
from __future__ import division


import collections
import random
import tokenization
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# 输入文件
flags.DEFINE_string("input_file", "./sample_text.txt",
                    "Input raw text file (or comma-separated list of files).")

# 输出文件：tf.tfrecord 输出TensorFlow训练格式
flags.DEFINE_string(
    "output_file", "./sample.tfrecord",
    "Output TF example file (or comma-separated list of files).")

# 词表路径
flags.DEFINE_string("vocab_file", "./vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
# 是否大小写
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

# 猜测是 wordPiece是否随机遮盖整个词，还是遮盖词根。
# bert有两种mask方式，一种是全词mask，一种是mask掉词根。
flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

# 最大序列长度，长截少补
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

# 每条训练数据去mask的待预测token的最大数量
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

# 随机种子
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

# bert对文档多次随机，这里是对一篇文章随机10次，每次不同的位置mask
flags.DEFINE_integer("dupe_factor",10,
                     "Number of times to duplicate the input data (with different masks).")
# 一条训练数据有15%的概率产生mask tokens sentences。
# 这样每条训练数据随机产生 max_predictions_per_seq x masked_lm_prob数量的mask
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

# 产生小于max_seq_len的训练数据的概率。目的是缩小预训练和微调过程的差距。
flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")
# 训练数据的实例类
class TrainingInstance(object):

    def __init__(self,tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        '''
        :param tokens: 分词、分字后的每个token；
        :param segment_ids: 分词后的id
        :param masked_lm_positions:随机mask的位置信息
        :param masked_lm_labels:？？
        :param is_random_next:下一句是否是随机的
        '''
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
    # 用于可视化对象
    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

def write_instance_to_example_files(instances,tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    # 写入到训练实例中

    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_float_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(values=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

# 从源文件中创建训练实例
def create_trainging_instances(input_files, tokenizer, max_seq_length,
                               dupe_factor,short_seq_prob,masked_lm_prob,
                               max_predictions_per_seq,rng):
    '''
    :param input_files: 源文件txt； list类型
    :param tokenizer: 将word转化为pieces和其他处理的类
    :param max_seq_length: 最大句子长度
    :param dupe_factor: 文档多次重复随机产生训练集，随机次数
    :param short_seq_prob: 为了缩小预训练和微调过程的差距，以此概率产生小于max_seq_length的训练数据
    :param masked_lm_prob: 一条训练数据产生mask的概率，即每条训练数据随机产生max_predictions_per_seq×masked_lm_prob的mask
    :param max_predictions_per_seq: 每条数据的mask的最大数量
    :param rng: 随机数
    :return:
    '''
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]
    # 1. 每行必须是一个句子，不能是片段或段落；
    # 2. 每个document必须以空行作为结束符；
    for input_file in input_files:
        with tf.gfile.GFile(input_file,"r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # 空行作为文档的分隔符
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line) # token拆成单个的词
                if tokens:
                    all_documents[-1].append(tokens)# 构建成二维列表，第一列是文章，第二列是被拆开的tokens
    # 移除空的document
    all_documents = [x for x in all_documents if x] # 删除空行
    rng.shuffle(all_documents) # 随机打乱

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    # 每个数据循环dupe_factor次，每次用不同的概率mask
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                # 构造next sentence和mask data
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
            )

    rng.shuffle(instances)
    return instances


def create_instances_from_document(all_documents, document_index,max_seq_length,
                                   short_seq_prob,masked_lm_prob,
                                   max_predictions_per_seq, vocab_words, rng):
    '''
    :param all_documents: 所有的文档
    :param document_index: 文档的索引
    :param max_seq_length: 最大长度
    :param short_seq_prob: 句子序列小于最大长度的概率
    :param masked_lm_prob: 一条训练数据产生mask的概率，即每条训练数据随机产生max_predictions_per_seq×masked_lm_prob数量的mask
    :param max_predictions_per_seq: 每条训练数据被mask的最大数量
    :param vocab_words: 词表大小
    :param rng:
    :return:
    '''

    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        # 如果随机数小于这个概率值，就产生一个较短的训练长度
        target_seq_length = rng.randint(2, max_num_tokens)
    # 这里没采用拼接的方式拼接上下句，而是将句子分为句子A和句子B，这样会增加预测next sentence的难度增加模型鲁棒性；
    instances = []
    current_chunk = [] # 产生训练集的候选集
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i] # 第一个词
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # a_end决定了一个序列中有多少个词被放入句子A中
                a_end = 1
                if len(current_chunk) >= 2:
                    # a_end产生的index，永远不大于序列长度
                    a_end = rng.randint(1,len(current_chunk) -1)
                # 产生A句子
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # 随机生成下一句
                is_random_next = False
                if len(current_chunk) ==1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length = len(tokens_a)

                    for _ in range(10):
                        random_document_index = rng.randint(0,len(all_documents) -1) # 产生一个随机文档的索引
                        if random_document_index != document_index:
                            break
                    random_document = all_documents[random_document_index] # 随机产生一个文档
                    random_start = rng.randint(0,len(random_document) -1) # 随机生成一个开始索引
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j]) # 构造随机的next sentences
                        if len(tokens_b) >= target_b_length:
                            break
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    # 真实的next sentences
                    is_random_next = False
                    for j in range(a_end,len(current_chunk)):
                        tokens_b.extend(current_chunk[j]) # 正确的句子

                # 将一对序列截断为最大序列长度？？？
                truncate_seq_pair(tokens_a,tokens_b,max_num_tokens,rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = [] # 词序列
                segment_ids = [] # 句子编码，第一句是0，第二句是1
                tokens.append(["CLS"])
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0) # 第一句的每个词都是0？？？

                tokens.append("[SEP]") # 中间间隔符
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

                instance = TrainingInstance(
                    tokens = tokens,segment_ids=segment_ids,is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions, masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk =[]
            current_length = 0
        i += 1
    return instances



MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words,rng):
    '''
    :param tokens: 句子
    :param masked_lm_prob: masked语言的概率
    :param max_predictions_per_seq: 每个句子mask几个
    :param vocab_words: 词表大小
    :param rng:
    :return:
    '''
    # 对句子中的token进行mask
    cand_indexes = [] # 候选索引
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # 如果是全词掩盖，且 候选索引长度大于1 ，且token的开始是 ##
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and token.startwith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)
    # 需要mask的词是固定的
    num_to_predict = min(max_predictions_per_seq,
                         max(1,int(round(len(tokens) * masked_lm_prob))))# 如果超过了最大mask限制，就选更小的那个

    masked_lms = [] # 掩盖的词
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) > num_to_predict:
            break
        # 如果将全词mask加入到数组中，会超出最大预测数量，所以此时就选择跳过
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue

        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index) # 要被掩盖的词的索引

            masked_token = None
            # 这个词 80%的时候，代替为[mask]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10%的时候，保持原样
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token # 替换成对应的mask

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key= lambda x:x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index) #被mask掉的index
        masked_lm_labels.append(p.label) # 被mask掉的词
    # 最终结果返回(原来的tokens，被mask掉的词的位置，被mask掉的词)
    return (output_tokens, masked_lm_positions, masked_lm_labels)

# 将sentence A B截断为最大序列长度
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        # 希望有时候从前面阶段，有时候从后面截断，增加随机性且避免 biases

        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

# 主函数
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # FullTokenizer以vocab为词典，将词转换为对应的id。会将词进行全拆分，
    # 如果该词没出现，会拆开继续look table，比如，hello，vocab中找到了llo和he，会拆成he和##llo。
    # 相当于词表,相当于一种处理未登录词的方式
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern)) # 获取输入文件列表

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info(" %s",input_file)

    rng = random.Random(FLAGS.random_seed)

    instances = create_trainging_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)



if __name__ == '__main__':
    # flags.mark_flag_as_required("input_file","./sample_text.txt")
    # flags.mark_flag_as_required("output_file","./sample.tfrecord")
    # flags.mark_flag_as_required("vocab_file","./vocab.txt")
    # flags.mark_flag_as_required(FLAGS.input_file)
    # flags.mark_flag_as_required(FLAGS.output_file)
    # flags.mark_flag_as_required(FLAGS.vocab_file)
    tf.app.run()