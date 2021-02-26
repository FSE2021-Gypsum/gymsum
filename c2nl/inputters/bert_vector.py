import torch
import numpy as np
from code2seq.c2q_utils import sentence_to_ids, pad_seq
import random


def vectorize(ex, config, args):
    """Vectorize a single example."""
    vectorized_ex = dict()
    vectorized_ex['id'] = ex.example_id
    vectorized_ex['language'] = config.lang

    # code 信息
    vectorized_ex['code_ids'] = torch.LongTensor(ex.source_ids)
    vectorized_ex['code_mask'] = torch.LongTensor(ex.source_mask)
    vectorized_ex['code_tokens'] = ex.source_tokens
    vectorized_ex['code_lens'] = ex.source_len
    vectorized_ex['src_vocab'] = ex.src_vocab
    vectorized_ex['code_texts'] = ex.source_texts

    # summary信息
    vectorized_ex['summary_ids'] = None
    vectorized_ex['summary_mask'] = None
    vectorized_ex['summary_lens'] = None
    vectorized_ex['summary_texts'] = None
    vectorized_ex['summary_tokens'] = None
    vectorized_ex['target_seq'] = None

    # if not config.only_test:
    vectorized_ex['summary_ids'] = torch.LongTensor(torch.LongTensor(ex.target_ids))
    vectorized_ex['summary_mask'] = torch.LongTensor(ex.target_mask)
    vectorized_ex['summary_tokens'] = ex.target_tokens
    vectorized_ex['summary_lens'] = ex.target_len
    vectorized_ex['summary_texts'] = ex.target_texts
    # target is only used to compute loss during training
    vectorized_ex['target_seq'] = torch.LongTensor(torch.LongTensor(ex.target_ids))

    vectorized_ex['seq_target'] = None
    if args and 'vocab_target' in args and ex.target_tokens_for_ast:
        vectorized_ex['seq_target'] = \
            sentence_to_ids(args['vocab_target'], ex.target_tokens_for_ast)

    vectorized_ex['seq_start'] = None
    vectorized_ex['seq_end'] = None
    vectorized_ex['seq_node'] = None
    # assert len(ex.ast_path) > 0
    if ex.ast_path is not None:
        seq_start, seq_node, seq_end = [], [], []
        syntax_path = [s for s in ex.ast_path if s != '' and s != '\n']
        num_k = config.model.code2seq.hyper.num_k
        vectorized_ex['num_k'] = num_k
        # not sample here! leave sample in batch_handle!
        sampled_path_index = range(len(syntax_path))

        for j in sampled_path_index:
            terminal1, ast_path, terminal2 = syntax_path[j].split(',')

            terminal1 = sentence_to_ids(args['vocab_subtoken'], terminal1.split('|'))
            ast_path = sentence_to_ids(args['vocab_nodes'], ast_path.split('|'))
            terminal2 = sentence_to_ids(args['vocab_subtoken'], terminal2.split('|'))

            seq_start.append(terminal1)
            seq_end.append(terminal2)
            seq_node.append(ast_path)
        vectorized_ex['seq_start'] = seq_start
        vectorized_ex['seq_end'] = seq_end
        vectorized_ex['seq_node'] = seq_node

    if ex.node_ids is not None and config.include_graph:
        vectorized_ex['node_ids'] = ex.node_ids
        vectorized_ex['node_lens'] = ex.node_lens
        vectorized_ex['node_token_ids'] = ex.node_token_ids
        vectorized_ex['node_token_lens'] = ex.node_token_lens
        vectorized_ex['total_token_len'] = ex.total_token_len
        vectorized_ex['node_types'] = ex.node_types
        vectorized_ex['edges'] = ex.edges
        vectorized_ex['edge_types'] = ex.edge_types

    return vectorized_ex


def batch_handle(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]

    # Batch Code Representations
    max_code_len = np.max([ex['code_lens'] for ex in batch])
    code_ids = torch.cat([ex['code_ids'][:max_code_len].unsqueeze(0) for ex in batch], dim=0)
    code_mask = torch.cat([ex['code_mask'][:max_code_len].unsqueeze(0) for ex in batch], dim=0)
    code_lens = torch.LongTensor([ex['code_lens'] for ex in batch])

    source_maps = []
    src_vocabs = []
    for i in range(batch_size):
        context = batch[i]['code_tokens']
        vocab = batch[i]['src_vocab']
        src_vocabs.append(vocab)
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([vocab[w] for w in context])
        source_maps.append(src_map)

    no_graph = 'node_ids' not in batch[0].keys()
    if no_graph:
        nodes = None
        graph_node_lens = None
        node_token_ids = None
        node_token_lens = None
        node_types = None
        edges = None
        edges_attrs = None
    else:
        nodes = [ex['node_ids'] for ex in batch]
        graph_node_lens = [ex['node_lens'] for ex in batch]
        max_tokens_len = max([ex['total_token_len'] for ex in batch])
        node_token_ids = torch.LongTensor(
            [ex['node_token_ids'] + [0] * (max_tokens_len - ex['total_token_len']) for ex in batch])
        # node_token_ids_mask = torch.LongTensor(
        #     [[1] * ex['total_token_len'] + [0] * (max_tokens_len - ex['total_token_len']) for ex in batch])
        node_token_lens = [ex['node_token_lens'] for ex in batch]
        max_node_len = max(ex['node_lens'] for ex in batch)
        node_types = torch.LongTensor([ex['node_types'] + [0] * (max_node_len - ex['node_lens']) for ex in batch])
        edges = [ex['edges'] for ex in batch]
        edges_attrs = [ex['edge_types'] for ex in batch]

    # --------- Prepare Summary tensors ---------
    no_summary = batch[0]['summary_ids'] is None
    if no_summary:
        summary_lens = None
        summary_ids = None
        summary_mask = None
        target_seq = None
        alignments = None
    else:
        max_summary_len = np.max([ex['summary_lens'] for ex in batch])
        summary_ids = torch.cat([ex['summary_ids'][:max_summary_len].unsqueeze(0) for ex in batch], dim=0)
        summary_mask = torch.cat([ex['summary_mask'][:max_summary_len].unsqueeze(0) for ex in batch], dim=0)
        target_seq = torch.cat([ex['target_seq'][:max_summary_len].unsqueeze(0) for ex in batch], dim=0)
        summary_lens = torch.LongTensor([ex['summary_lens'] for ex in batch])

        alignments = []
        for i in range(batch_size):
            target = batch[i]['summary_tokens']
            align_mask = torch.LongTensor([src_vocabs[i][w] for w in target])
            alignments.append(align_mask)

    batch_starts, batch_nodes, batch_ends, batch_targets, start_lens, node_lens, end_lens, target_lens, \
    start_max_len, node_max_len, end_max_len, target_max_len, lengths_k, reverse_index = \
        None, None, None, None, None, None, None, None, \
        None, None, None, None, None, None

    if batch[0]['seq_start']:
        # TODO: prepare for code2seq
        # sample here
        num_k = batch[0]['num_k']
        seq_starts, seq_nodes, seq_ends = [], [], []
        for i in range(batch_size):
            _len = len(batch[i]['seq_start'])
            seq_start, seq_node, seq_end = [], [], []
            if _len > num_k:
                sampled_path_index = random.sample(range(_len), num_k)
            else:
                sampled_path_index = range(_len)
            for j in sampled_path_index:
                seq_start.append(batch[i]['seq_start'][j])
                seq_node.append(batch[i]['seq_node'][j])
                seq_end.append(batch[i]['seq_end'][j])
            seq_starts.append(seq_start)
            seq_nodes.append(seq_node)
            seq_ends.append(seq_end)
        # length_k : (batch_size, k)
        lengths_k = [len(ex) for ex in seq_nodes]
        # flattening (batch_size, k, l) to (batch_size * k, l)
        # this is useful to make torch.tensor
        seq_starts = [symbol for k in seq_starts for symbol in k]
        seq_nodes = [symbol for k in seq_nodes for symbol in k]
        seq_ends = [symbol for k in seq_ends for symbol in k]
        seq_targets = [ex['seq_target'] for ex in batch]
        # Padding
        start_lens = [len(s) for s in seq_starts]
        node_lens = [len(s) for s in seq_nodes]
        end_lens = [len(s) for s in seq_ends]
        target_lens = [len(s) for s in seq_targets]

        start_max_len = max(start_lens)
        node_max_len = max(node_lens)
        end_max_len = max(end_lens)
        target_max_len = max(target_lens)

        padded_starts = [pad_seq(s, start_max_len) for s in seq_starts]
        padded_nodes = [pad_seq(s, node_max_len) for s in seq_nodes]
        padded_ends = [pad_seq(s, end_max_len) for s in seq_ends]
        padded_targets = [pad_seq(s, target_max_len) for s in seq_targets]

        # index for split (batch_size * k, l) into (batch_size, k, l)
        reverse_index = range(len(node_lens))

        # sort for rnn
        seq_pairs = sorted(zip(node_lens, reverse_index, padded_nodes, padded_starts, padded_ends), key=lambda p: p[0],
                           reverse=True)
        node_lens, reverse_index, padded_nodes, padded_starts, padded_ends = zip(*seq_pairs)

        batch_starts = torch.tensor(padded_starts, dtype=torch.long)
        batch_ends = torch.tensor(padded_ends, dtype=torch.long)

        # transpose for rnn
        batch_nodes = torch.tensor(padded_nodes, dtype=torch.long).transpose(0, 1)
        batch_targets = torch.tensor(padded_targets, dtype=torch.long).transpose(0, 1)

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'code_ids': code_ids,
        'code_mask': code_mask,
        'code_lens': code_lens,
        'summary_ids': summary_ids,
        'summary_mask': summary_mask,
        'summary_lens': summary_lens,
        'target_seq': target_seq,
        'code_texts': [ex['code_texts'] for ex in batch],
        'summary_texts': [ex['summary_texts'] for ex in batch],
        'code_tokens': [ex['code_tokens'] for ex in batch],
        'summary_tokens': [ex['summary_tokens'] for ex in batch],
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'alignment': alignments,
        'batch_starts': batch_starts,
        'batch_nodes': batch_nodes,
        'batch_ends': batch_ends,
        'batch_targets': batch_targets,
        'start_lens': start_lens,
        'node_lens': node_lens,
        'end_lens': end_lens,
        'target_lens': target_lens,
        'max_start_len': start_max_len,
        'max_node_len': node_max_len,
        'max_end_len': end_max_len,
        'max_target_len': target_max_len,
        'seq_lens': lengths_k,
        'reverse_index': reverse_index,
        'nodes': nodes,
        'graph_node_lens': graph_node_lens,
        'node_token_ids': node_token_ids,
        'node_token_lens': node_token_lens,
        'node_types': node_types,
        'edges': edges,
        'edges_attrs': edges_attrs
    }
