import copy
import json

import torch

from c2nl.inputters.bert_vocabulary import Vocabulary
from itertools import chain


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 nodes=None,
                 node_types=None,
                 node_texts=None,
                 edges=None,
                 edge_types=None,
                 source_ast=None
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.source_ast = source_ast
        self.nodes = nodes
        self.node_types = node_types
        self.node_texts = node_texts
        self.edges = edges
        self.edge_types = edge_types


def read_examples(config, task):
    """Read examples from filename."""
    import os
    assert task in ['train', 'dev_remove', 'test_remove', 'dev', 'test']
    include_ast = config.include_ast
    base_dir = os.path.join(config.data_dir, config.lang, task)
    train_src = os.path.join(base_dir, config.src)
    train_tgt = os.path.join(base_dir, config.tgt)
    examples = []
    src_lines = open(train_src, encoding='utf-8').readlines()
    tgt_lines = open(train_tgt, encoding='utf-8').readlines()

    cur_idx = 0
    src_ast_length = None
    cur_json = None
    src_ast_lines = None
    source_ast = None
    code2seq_config = config.model.code2seq
    graph_config = config.model.graph
    include_graph = config.include_graph

    if include_ast:
        train_ast = os.path.join(code2seq_config.data.home, config.lang, code2seq_config.data[task])
        src_ast_lines = open(train_ast, encoding='utf-8').readlines()
        src_ast_length = len(src_ast_lines)
        cur_idx = 0
        cur_json = json.loads(src_ast_lines[cur_idx])

    graph_data = None
    nodes = None
    node_texts = None
    node_types = None
    edges = None
    edge_types = None

    if include_graph:
        graph_data = json.loads(open(os.path.join(base_dir, graph_config.data), 'r').read())

    for idx, (code, nl) in enumerate(zip(src_lines, tgt_lines)):
        code = code.replace('\n', ' ')
        code = ' '.join(code.strip().split(' '))
        nl = nl.replace('\n', ' ')
        nl = ' '.join(nl.strip().split(' '))

        if include_graph:
            graph = graph_data[str(idx)]
            nodes = graph['node_ids']
            node_texts = [a.lower() for a in graph['node_texts']]
            node_types = graph['node_types']
            edges = graph['edges']
            edge_types = graph['edge_types']

        if include_ast:
            if cur_idx < src_ast_length and cur_json['id'] == idx:
                source_ast = copy.deepcopy(cur_json['path'])
                cur_idx += 1
                if cur_idx < src_ast_length:
                    cur_json = json.loads(src_ast_lines[cur_idx])
                target_, *syntax_path = source_ast.split(' ')
                if len(syntax_path) == 0:
                    source_ast = None
            else:
                source_ast = None
        if not code2seq_config.remove or source_ast is not None:
            examples.append(
                Example(
                    idx=idx,
                    source=code.lower(),
                    target=nl.lower(),
                    nodes=nodes,
                    node_texts=node_texts,
                    node_types=node_types,
                    edges=edges,
                    edge_types=edge_types,
                    source_ast=source_ast
                )
            )

    print('task: %s, num_examples: %d' % (task, len(examples)))
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_tokens,
                 target_tokens,
                 source_texts,
                 target_texts,
                 source_mask,
                 target_mask,
                 source_len,
                 target_len,
                 ast_path,
                 target_tokens_for_ast,
                 node_ids,
                 node_lens,
                 node_token_ids,
                 node_token_lens,
                 total_token_len,
                 node_types,
                 edges,
                 edge_types):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.source_len = source_len
        self.target_len = target_len
        self.src_vocab = None
        self.ast_path = ast_path
        self.target_tokens_for_ast = target_tokens_for_ast
        self.node_ids = node_ids
        self.node_lens = node_lens
        self.node_token_ids = node_token_ids
        self.node_token_lens = node_token_lens
        self.total_token_len = total_token_len
        self.node_types = node_types
        self.edges = edges
        self.edge_types = edge_types

    def form_src_vocab(self, tokenizer) -> None:
        self.src_vocab = Vocabulary(tokenizer)
        assert self.src_vocab.remove(tokenizer.cls_token)
        assert self.src_vocab.remove(tokenizer.sep_token)
        self.src_vocab.add_tokens(self.source_tokens)


def convert_examples_to_features(examples, tokenizer, nmt_config, stage=None, logger=None):
    features = []
    c_model = nmt_config.model
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:c_model.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        source_len = len(source_ids)
        padding_length = c_model.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        source_texts = example.source
        target_texts = example.target

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:c_model.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = c_model.max_target_length - len(target_ids)
        target_len = len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        # TODO: code2seq mode
        ast_path = None
        target_tokens_for_ast = None
        if example.source_ast:
            target_tokens_for_ast, *ast_path = example.source_ast.split(' ')
            target_tokens_for_ast = target_tokens_for_ast.split('|')

        node_lens = None
        node_token_lens = None
        node_token_ids = None
        total_token_len = None
        if example.nodes:
            node_lens = len(example.nodes)
            node_texts = example.node_texts
            node_tokens = [tokenizer.tokenize(text) for text in node_texts]
            node_token_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in node_tokens]
            node_token_lens = [len(ids) for ids in node_token_ids]
            node_token_ids = list(chain.from_iterable(node_token_ids))
            total_token_len = sum(node_token_lens)

        if example_index < 5 and logger:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
        feature = InputFeatures(
            example.idx,
            source_ids,
            target_ids,
            source_tokens,
            target_tokens,
            source_texts,
            target_texts,
            source_mask,
            target_mask,
            source_len,
            target_len,
            ast_path,
            target_tokens_for_ast,
            node_ids=example.nodes,
            node_lens=node_lens,
            node_token_ids=node_token_ids,
            node_token_lens=node_token_lens,
            total_token_len=total_token_len,
            node_types=example.node_types,
            edges=example.edges,
            edge_types=example.edge_types)

        feature.form_src_vocab(tokenizer)
        features.append(feature)
    return features


# def get_ratio(dropout, bert_drop_radio, encoder_bert_mixup=True, training=True):
#     if dropout:
#         frand = float(uniform(0, 1))
#         if encoder_bert_mixup and training:
#             return [frand, 1 - frand]
#         if frand < bert_drop_radio and training:
#             return [1, 0]
#         elif frand > 1 - bert_drop_radio and training:
#             return [0, 1]
#         else:
#             return [0.5, 0.5]
#     else:
#         return [self.encoder_ratio, self.bert_ratio]


def sort_sequence(inputs, sequence_length, batch_first):
    # assume seq_lengths = [3, 5, 2]
    # 对序列长度进行排序(降序), sorted_seq_lengths = [5, 3, 2]
    # indices 为 [1, 0, 2], indices 的值可以这么用语言表述
    # 原来 batch 中在 0 位置的值, 现在在位置 1 上.
    # 原来 batch 中在 1 位置的值, 现在在位置 0 上.
    # 原来 batch 中在 2 位置的值, 现在在位置 2 上.
    sorted_seq_lengths, indices = torch.sort(sequence_length, descending=True)

    # 如果我们想要将计算的结果恢复排序前的顺序的话,
    # 只需要对 indices 再次排序(升序),会得到 [0, 1, 2],
    # desorted_indices 的结果就是 [1, 0, 2]
    # 使用 desorted_indices 对计算结果进行索引就可以了.
    _, desorted_indices = torch.sort(indices, descending=False)

    # 对原始序列进行排序
    if batch_first:
        inputs = inputs[indices]
    else:
        inputs = inputs[:, indices]

    return inputs, sorted_seq_lengths, desorted_indices
