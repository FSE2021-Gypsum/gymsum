import copy
import logging
import math

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils.configs import ConfigDict
from c2nl.models.transformer_bert import TransformerBert
from c2nl.utils.bert_copy_utils import collapse_copy_scores, replace_unknown, \
    make_src_map, align
from c2nl.utils.bert_misc import tens2sen
from utils.utils import try_gpu

logger = logging.getLogger(__name__)


class Code2NaturalLanguage(object):
    """High level model that handles initialization the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, config, code_bert, tokenizer, state_dict=None):
        # Book-keeping.
        c_model = config.model
        self.c_model = c_model
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.optimizer = None
        self.include_ast = config.include_ast
        self.include_graph = config.include_graph
        self.network = TransformerBert(self.c_model, code_bert, tokenizer, self.include_ast, self.include_graph)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.c_model.optimizer == 'sgd':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(parameters,
                                       self.c_model.learning_rate,
                                       momentum=self.c_model.momentum,
                                       weight_decay=self.c_model.weight_decay)

        elif self.c_model.optimizer == 'adam':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(parameters,
                                        self.c_model.learning_rate,
                                        weight_decay=self.c_model.weight_decay)

        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.c_model.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = try_gpu(v)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------
    # training progress
    def update(self, ex, tokenizer):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        source_map, alignment = None, None
        blank, fill = None, None

        # To enable copy attn, collect source map and alignment info
        if self.c_model.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex
            source_map = try_gpu(make_src_map(ex['src_map']))
            alignment = try_gpu(align(ex['alignment']))
            blank, fill = collapse_copy_scores(tokenizer, ex['src_vocab'])

        code_ids = try_gpu(ex['code_ids'])
        code_mask = try_gpu(ex['code_mask'])
        code_lens = try_gpu(ex['code_lens'])
        summary_ids = try_gpu(ex['summary_ids'])
        summary_mask = try_gpu(ex['summary_mask'])
        summary_lens = try_gpu(ex['summary_lens'])
        target_seq = try_gpu(ex['target_seq'])
        batch_starts = try_gpu(ex['batch_starts'])
        batch_nodes = try_gpu(ex['batch_nodes'])
        batch_ends = try_gpu(ex['batch_ends'])
        seq_lens = ex['seq_lens']
        reverse_index = ex['reverse_index']
        node_lens = ex['node_lens']

        nodes = ex['nodes']
        graph_node_lens = ex['graph_node_lens']
        node_token_ids = ex['node_token_ids']
        node_token_lens = ex['node_token_lens']
        node_types = ex['node_types']
        edges = ex['edges']
        edges_attrs = ex['edges_attrs']

        if any(l is None for l in ex['language']):
            ex_weights = None
        else:
            ex_weights = [1.0]
            ex_weights = try_gpu(torch.FloatTensor(ex_weights))

        # Run forward
        net_loss = self.network(code_ids=code_ids,
                                code_lens=code_lens,
                                summary_ids=summary_ids,
                                summary_lens=summary_lens,
                                target_seq=target_seq,
                                src_map=source_map,
                                alignment=alignment,
                                blank=blank,
                                fill=fill,
                                source_vocab=ex['src_vocab'],
                                code_mask=code_mask,
                                summary_mask=summary_mask,
                                example_weights=ex_weights,
                                batch_starts=batch_starts,
                                batch_nodes=batch_nodes,
                                batch_ends=batch_ends,
                                seq_lens=seq_lens,
                                reverse_index=reverse_index,
                                node_lens=node_lens,
                                nodes=nodes,
                                graph_node_lens=graph_node_lens,
                                node_token_ids=node_token_ids,
                                node_token_lens=node_token_lens,
                                node_types=node_types,
                                edges=edges,
                                edge_attrs=edges_attrs,
                                )

        loss = net_loss['ml_loss'].mean() if self.parallel \
            else net_loss['ml_loss']
        loss_per_token = net_loss['loss_per_token'].mean() if self.parallel \
            else net_loss['loss_per_token']
        ml_loss = loss.item()
        loss_per_token = loss_per_token.item()
        loss_per_token = 10 if loss_per_token > 10 else loss_per_token
        perplexity = math.exp(loss_per_token)

        loss.backward()

        clip_grad_norm_(self.network.parameters(), self.c_model.grad_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1
        return {
            'ml_loss': ml_loss,
            'perplexity': perplexity
        }

    # for RL
    def calculate(self, ex, tokenizer):
        """Forward a batch of examples; step the optimizer to update weights."""

        source_map, alignment = None, None
        blank, fill = None, None

        # To enable copy attn, collect source map and alignment info
        if self.c_model.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex
            source_map = try_gpu(make_src_map(ex['src_map']))
            alignment = try_gpu(align(ex['alignment']))
            blank, fill = collapse_copy_scores(tokenizer, ex['src_vocab'])

        code_ids = try_gpu(ex['code_ids'])
        code_mask = try_gpu(ex['code_mask'])
        code_lens = try_gpu(ex['code_lens'])
        summary_ids = try_gpu(ex['summary_ids'])
        summary_mask = try_gpu(ex['summary_mask'])
        summary_lens = try_gpu(ex['summary_lens'])
        target_seq = try_gpu(ex['target_seq'])
        batch_starts = try_gpu(ex['batch_starts'])
        batch_nodes = try_gpu(ex['batch_nodes'])
        batch_ends = try_gpu(ex['batch_ends'])
        seq_lens = ex['seq_lens']
        reverse_index = ex['reverse_index']
        node_lens = ex['node_lens']

        if any(l is None for l in ex['language']):
            ex_weights = None
        else:
            ex_weights = [self.config.dataset_weights[lang] for lang in ex['language']]
            ex_weights = try_gpu(torch.FloatTensor(ex_weights))

        # Run forward
        net_loss = self.network.calculate(code_ids=code_ids,
                                          code_lens=code_lens,
                                          summary_ids=summary_ids,
                                          summary_lens=summary_lens,
                                          target_seq=target_seq,
                                          src_map=source_map,
                                          alignment=alignment,
                                          blank=blank,
                                          fill=fill,
                                          source_vocab=ex['src_vocab'],
                                          code_mask=code_mask,
                                          summary_mask=summary_mask,
                                          example_weights=ex_weights,
                                          batch_starts=batch_starts,
                                          batch_nodes=batch_nodes,
                                          batch_ends=batch_ends,
                                          seq_lens=seq_lens,
                                          reverse_index=reverse_index,
                                          node_lens=node_lens
                                          )
        return net_loss

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, tokenizer, replace_unk=False):
        """Forward a batch of examples only to get predictions.
        Args:
            tokenizer
            ex: the batch examples
            replace_unk: replace `unk` tokens while generating predictions
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.network.eval()

        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.c_model.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex

            source_map = make_src_map(ex['src_map'])
            source_map = try_gpu(source_map)

            blank, fill = collapse_copy_scores(tokenizer, ex['src_vocab'])

        code_ids = try_gpu(ex['code_ids'])
        code_mask = try_gpu(ex['code_mask'])
        code_lens = try_gpu(ex['code_lens'])
        batch_starts = try_gpu(ex['batch_starts'])
        batch_nodes = try_gpu(ex['batch_nodes'])
        batch_ends = try_gpu(ex['batch_ends'])
        seq_lens = ex['seq_lens']
        reverse_index = ex['reverse_index']
        node_lens = ex['node_lens']

        nodes = ex['nodes']
        graph_node_lens = ex['graph_node_lens']
        node_token_ids = ex['node_token_ids']
        node_token_lens = ex['node_token_lens']
        node_types = ex['node_types']
        edges = ex['edges']
        edges_attrs = ex['edges_attrs']

        decoder_out = self.network(code_ids=code_ids,
                                   code_lens=code_lens,
                                   summary_ids=None,
                                   summary_lens=None,
                                   target_seq=None,
                                   src_map=source_map,
                                   alignment=alignment,
                                   blank=blank, fill=fill,
                                   source_vocab=ex['src_vocab'],
                                   code_mask=code_mask,
                                   batch_starts=batch_starts,
                                   batch_nodes=batch_nodes,
                                   batch_ends=batch_ends,
                                   seq_lens=seq_lens,
                                   reverse_index=reverse_index,
                                   node_lens=node_lens,
                                   nodes=nodes,
                                   graph_node_lens=graph_node_lens,
                                   node_token_ids=node_token_ids,
                                   node_token_lens=node_token_lens,
                                   node_types=node_types,
                                   edges=edges,
                                   edge_attrs=edges_attrs,
                                   )

        predictions = tens2sen(decoder_out['predictions'],
                               self.tokenizer,
                               ex['src_vocab'])
        if replace_unk:
            for i in range(len(predictions)):
                enc_dec_attn = decoder_out['attentions'][i]
                assert enc_dec_attn.dim() == 3
                enc_dec_attn = enc_dec_attn.mean(1)
                predictions[i] = replace_unknown(predictions[i],
                                                 enc_dec_attn,
                                                 src_raw=ex['code_tokens'][i],
                                                 unk_word=self.tokenizer.unk_token)

        last_pred = []
        for i in range(len(predictions)):
            ids = [tokenizer.convert_tokens_to_ids(w) for w in predictions[i].split(' ')]
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id)]

            if tokenizer.pad_token_id in ids:
                ids = ids[:ids.index(tokenizer.pad_token_id)]

            if len(ids) == 0:
                ids = [tokenizer.pad_token_id]

            last_pred.append(tokenizer.decode(ids, clean_up_tokenization_spaces=False))

        targets = [summary for summary in ex['summary_texts']]
        return last_pred, targets, decoder_out['copy_info']

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'args': self.config.to_vanilla_(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'args': self.config.to_vanilla_(),
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, code_bert, tokenizer, config):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        # args = ConfigDict(saved_params['args'])
        # TODO
        args = config
        state_dict = saved_params['state_dict']
        model = Code2NaturalLanguage(args, code_bert, tokenizer, state_dict)
        return model

    @staticmethod
    def load_checkpoint(filename, config, code_bert, tokenizer, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer = saved_params['optimizer']
        state_dict = saved_params['state_dict']
        model = Code2NaturalLanguage(config, code_bert, tokenizer)
        model.network.load_state_dict(state_dict)
        model.updates = updates
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = try_gpu(self.network)

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
