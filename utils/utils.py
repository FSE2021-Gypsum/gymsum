from collections import defaultdict

import numpy as np
import torch

from utils.constants import BOS, EOS, PAD_INDEX

_GPUS_EXIST = True


def try_gpu(x, cuda=True):
    if x is None:
        return x
    """Try to put x on a GPU."""
    global _GPUS_EXIST
    if _GPUS_EXIST and cuda:
        try:
            return x.cuda()
        except (AssertionError, RuntimeError):
            print('No GPUs detected. Sticking with CPUs.')
            _GPUS_EXIST = False
    return x


def get_inputs(src, vocab):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """
    word_ids = word2id(src, vocab)
    padded_ids, masks = pad_sequence(word_ids)
    return try_gpu(torch.LongTensor(padded_ids))


def pad_sequence(src):
    """
    padding input函数，返回mask和padding_input

    Args:
        src: 输入数据
    Returns:
        padded_word
        masks
    """
    max_len = max(len(s) for s in src)
    batch_size = len(src)

    padded_word = []
    masks = []
    for i in range(max_len):
        padded_word.append([src[k][i] if len(src[k]) > i else PAD_INDEX for k in range(batch_size)])
        masks.append([1 if len(src[k]) > i else 0 for k in range(batch_size)])

    return padded_word, masks


def word2id(inputs, vocab):
    """

    Args:
        inputs:  输入数据
        vocab:  词典
    Returns:
        input_indices
    """
    if type(inputs[0]) == list:
        return [[vocab[w] for w in s] for s in inputs]
    else:
        return [vocab[w] for w in inputs]


def tensor_transform(linear, x):
    # X is a 3D tensor
    return linear(x.contiguous().view(-1, x.size(2))).view(x.size(0), x.size(1), -1)


def read_corpus(file_path, source):
    data = []
    for line in open(file_path, encoding='utf-8'):
        sent = line.strip().split(' ')
        sent = [w for w in sent]
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = [BOS] + sent + [EOS]
        data.append(sent)

    return data


def read_corpus_for_dsl(file_path, source):
    data = []
    lm_scores = []
    scores_path = file_path + '.score'
    for line, score in zip(open(file_path), open(scores_path)):
        sent = line.strip().split(' ')
        if source != 'tgt':
            lm_scores.append(float(score))
        data.append(sent)

    return data, lm_scores


def batch_slice(data, batch_size, sort=True):
    batched_data = []
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        batched_data.append((src_sents, tgt_sents))

    return batched_data


def batch_slice_for_dsl(data, batch_size, sort=True):
    batched_data = []
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]
        src_scores = [data[i * batch_size + b][2] for b in range(cur_batch_size)]
        tgt_scores = [data[i * batch_size + b][3] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]
            src_scores = [src_scores[src_id] for src_id in src_ids]
            tgt_scores = [tgt_scores[src_id] for src_id in src_ids]

        batched_data.append((src_sents, tgt_sents, src_scores, tgt_scores))

    return batched_data


def get_new_batch(batch_data):
    cur_batch_size = len(batch_data[0])
    src_sents, tgt_sents, src_scores, tgt_scores = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
    src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(tgt_sents[src_id]), reverse=True)
    src_sents = [src_sents[src_id] for src_id in src_ids]
    tgt_sents = [tgt_sents[src_id] for src_id in src_ids]
    src_scores = [src_scores[src_id] for src_id in src_ids]
    tgt_scores = [tgt_scores[src_id] for src_id in src_ids]

    batch_data = (src_sents, tgt_sents, src_scores, tgt_scores)
    return batch_data


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """
    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle:
            np.random.shuffle(tuples)
        batched_data.extend(batch_slice(tuples, batch_size))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch


def data_iter_for_dual(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle:
            np.random.shuffle(tuples)
        batched_data.extend(batch_slice_for_dsl(tuples, batch_size))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch
