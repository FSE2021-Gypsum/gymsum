from abc import ABC

import copy
import torch
import torch.nn as nn
import torch.nn.functional as f

from bert_nmt.bert_utils import Beam
from bert_nmt.encoder_decoder import Decoder
from c2nl.modules.bert_copy_generator import CopyGenerator, CopyGeneratorCriterion
from c2nl.modules.global_attention import GlobalAttention
from code2seq.model import Encoder
from ggnn.model import Encoder as GraphEncoder
from utils.utils import try_gpu


class TransformerBert(nn.Module, ABC):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, config, code_bert, tokenizer, include_ast=False, include_graph=False):
        """"Constructor of the class."""
        super(TransformerBert, self).__init__()

        self.name = 'TransformerBert'
        self.config = config
        self.encoder = code_bert
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer
        # self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        # self.encoder = Encoder(config, config.emsize)
        self.decoder = Decoder(config, config.emsize, include_ext=include_graph or include_ast)
        self.layer_wise_attn = config.layer_wise_attn
        self.dense = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.generator = nn.Linear(self.decoder.input_size, tokenizer.vocab_size)
        # =================================================================
        # begin code2seq
        self.include_ast = include_ast
        c2q_config = config.code2seq.hyper
        self.c2q_config = c2q_config
        self.max_ast = c2q_config.num_k
        if include_ast:
            self.ast_encoder = Encoder(c2q_config.vocab_size_sub_token, c2q_config.vocab_size_nodes,
                                       c2q_config.token_size, config.emsize,
                                       bidirectional=c2q_config.bidirectional, num_layers=c2q_config.num_layers,
                                       rnn_dropout=c2q_config.rnn_dropout,
                                       embeddings_dropout=c2q_config.embeddings_dropout)
        # =================================================================

        # =================================================================
        # begin graph network
        self.include_graph = include_graph
        graph_config = config.graph
        self.graph_config = graph_config
        if include_graph:
            self.graph_encoder = GraphEncoder(graph_config, config.emsize)
        # =================================================================

        if config.share_decoder_embeddings:
            self.tie_weights()

        self._copy = config.copy_attn
        if self._copy:
            self.copy_attn = GlobalAttention(dim=self.decoder.input_size,
                                             attn_type=config.attn_type)
            self.copy_generator = CopyGenerator(self.decoder.input_size,
                                                tokenizer.pad_token_id,
                                                self.generator)
            self.criterion = CopyGeneratorCriterion(vocab_size=self.vocab_size,
                                                    force_copy=config.force_copy, tokenizer=tokenizer)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of wither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.generator,
                                   self.encoder.embeddings.word_embeddings)

    def _run_forward_ml(self,
                        code_ids,
                        code_lens,
                        summary_ids,
                        summary_lens,
                        target_seq,
                        src_map,
                        alignment,
                        **kwargs):

        # embed and encode the source sequence
        # code_emb = self.encoder.embeddings(code_word_rep)
        # source_mask 0->padding
        source_mask = kwargs['code_mask']
        target_mask = kwargs['summary_mask']

        # 'batch_starts': batch_starts,
        # 'batch_nodes': batch_nodes,
        # 'batch_ends': batch_ends,
        # 'batch_targets': batch_targets,
        # 'start_lens': start_lens,
        # 'node_lens': node_lens,
        # 'end_lens': end_lens,
        # 'target_lens': target_lens,
        # 'max_start_len': start_max_len,
        # 'max_node_len': node_max_len,
        # 'max_end_len': end_max_len,
        # 'max_target_len': target_max_len,
        # 'seq_lens': lengths_k,
        # 'reverse_index': reverse_index

        batch_starts = kwargs['batch_starts']
        batch_nodes = kwargs['batch_nodes']
        batch_ends = kwargs['batch_ends']
        seq_lens = kwargs['seq_lens']
        reverse_index = kwargs['reverse_index']
        node_lens = kwargs['node_lens']

        # Encoder
        ext_lens = None
        ext_out = None
        if self.include_ast:
            ext_out, ast_hidden = \
                self.ast_encoder(batch_starts, batch_nodes, batch_ends, seq_lens, reverse_index, node_lens)
            ext_lens = torch.LongTensor(seq_lens).to(ext_out.device)

        if self.include_graph:
            nodes = kwargs['nodes']
            graph_node_lens = kwargs['graph_node_lens']
            node_token_ids = kwargs['node_token_ids'].to(source_mask.device)
            node_token_lens = kwargs['node_token_lens']
            node_types = kwargs['node_types'].to(source_mask.device)
            edges = kwargs['edges']
            edges_attrs = kwargs['edge_attrs']
            ext_out = self.graph_encoder(nodes, graph_node_lens, node_token_ids,
                                         node_token_lens, node_types,
                                         edges, edges_attrs, self.encoder.embeddings)
            ext_lens = torch.LongTensor(graph_node_lens).to(ext_out.device)

        # B x seq_len x h
        outputs = self.encoder(code_ids, attention_mask=source_mask)
        memory_bank = outputs[0]
        # code_emb = self.encoder.embeddings(code_ids)
        # memory_bank, layer_wise_outputs = self.encoder(code_emb, code_lens)
        # memory_bank = memory_bank[-1]

        # embed and encode the target sequence
        summary_emb = self.encoder.embeddings(summary_ids)
        summary_pad_mask = (1 - target_mask).bool()  # 1 for padding
        layer_wise_dec_out, _ = self.decoder(memory_bank,
                                             code_lens,
                                             summary_pad_mask,
                                             summary_emb,
                                             ext_out,
                                             ext_lens)
        decoder_outputs = layer_wise_dec_out[-1]
        decoder_outputs = torch.tanh(self.dense(decoder_outputs))
        loss = dict()
        target = target_seq[:, 1:].contiguous()
        if self._copy:
            # copy_score: batch_size, tgt_len, src_len
            _, copy_score, _ = self.copy_attn(decoder_outputs,
                                              memory_bank,
                                              memory_lengths=code_lens,
                                              softmax_weights=False)

            # mask copy_attn weights here if needed
            # if source_mask is not None:
            #     mask = (1 - source_mask.unsqueeze(1)).bool()
            #     copy_score.data.masked_fill_(mask, -float('inf'))

            attn_copy = f.softmax(copy_score, dim=-1)
            scores = self.copy_generator(decoder_outputs, attn_copy, src_map)
            scores = scores[:, :-1, :].contiguous()
            ml_loss = self.criterion(scores, alignment[:, 1:].contiguous(), target)
        else:
            scores = self.generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
            scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`

            ml_loss = self.criterion(scores.view(-1, scores.size(2)),
                                     target=target.view(-1))

        ml_loss = ml_loss.view(*scores.size()[:-1])
        ml_loss = ml_loss.mul(target.ne(self.tokenizer.pad_token_id).float())
        ml_loss = ml_loss.sum(1) * kwargs['example_weights']
        loss['ml_loss'] = ml_loss.mean()
        loss['loss_per_token'] = ml_loss.div((summary_lens - 1).float()).mean()

        return loss

    def calculate(self,
                  code_ids,
                  code_lens,
                  summary_ids,
                  summary_lens,
                  target_seq,
                  src_map,
                  alignment,
                  **kwargs):

        # embed and encode the source sequence
        # code_emb = self.encoder.embeddings(code_word_rep)
        # source_mask 0->padding
        source_mask = kwargs['code_mask']
        target_mask = kwargs['summary_mask']

        # 'batch_starts': batch_starts,
        # 'batch_nodes': batch_nodes,
        # 'batch_ends': batch_ends,
        # 'batch_targets': batch_targets,
        # 'start_lens': start_lens,
        # 'node_lens': node_lens,
        # 'end_lens': end_lens,
        # 'target_lens': target_lens,
        # 'max_start_len': start_max_len,
        # 'max_node_len': node_max_len,
        # 'max_end_len': end_max_len,
        # 'max_target_len': target_max_len,
        # 'seq_lens': lengths_k,
        # 'reverse_index': reverse_index

        batch_starts = kwargs['batch_starts']
        batch_nodes = kwargs['batch_nodes']
        batch_ends = kwargs['batch_ends']
        seq_lens = kwargs['seq_lens']
        reverse_index = kwargs['reverse_index']
        node_lens = kwargs['node_lens']

        # Encoder
        ext_lens = None
        ext_out = None
        if self.include_ast:
            ext_out, ext_hidden = \
                self.ast_encoder(batch_starts, batch_nodes, batch_ends, seq_lens, reverse_index, node_lens)
            ext_lens = torch.LongTensor(seq_lens).to(ext_out.device)

        # B x seq_len x h
        outputs = self.encoder(code_ids, attention_mask=source_mask)
        memory_bank = outputs[0]
        # code_emb = self.encoder.embeddings(code_ids)
        # memory_bank, layer_wise_outputs = self.encoder(code_emb, code_lens)
        # memory_bank = memory_bank[-1]

        # embed and encode the target sequence
        summary_emb = self.encoder.embeddings(summary_ids)
        summary_pad_mask = (1 - target_mask).bool()  # 1 for padding
        layer_wise_dec_out, _ = self.decoder(memory_bank,
                                             code_lens,
                                             summary_pad_mask,
                                             summary_emb,
                                             ext_out,
                                             ext_lens)
        decoder_outputs = layer_wise_dec_out[-1]
        decoder_outputs = torch.tanh(self.dense(decoder_outputs))
        loss = dict()
        target = target_seq[:, 1:].contiguous()
        if self._copy:
            # copy_score: batch_size, tgt_len, src_len
            _, copy_score, _ = self.copy_attn(decoder_outputs,
                                              memory_bank,
                                              memory_lengths=code_lens,
                                              softmax_weights=False)

            # mask copy_attn weights here if needed
            # if source_mask is not None:
            #     mask = (1 - source_mask.unsqueeze(1)).bool()
            #     copy_score.data.masked_fill_(mask, -float('inf'))

            attn_copy = f.softmax(copy_score, dim=-1)
            scores = self.copy_generator(decoder_outputs, attn_copy, src_map)
            scores = scores[:, :-1, :].contiguous()
            _b_size, _len_, _vocab_size = scores.size()
            alignment = alignment[:, 1:].contiguous().view(-1, 1)
            target = target.view(-1, 1)
            scores = scores.view(-1, _vocab_size)
            loss = (scores.gather(1, alignment) + scores.gather(1, target)).view(_b_size, _len_)
        else:
            scores = self.generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
            scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`

        return torch.log(loss + 1e-20).sum(-1)

    def forward(self,
                code_ids,
                code_lens,
                summary_ids,
                summary_lens,
                target_seq,
                src_map,
                alignment,
                **kwargs):
        """
        Input:
            - code_ids: ``(batch_size, max_doc_len)``
            - code_lens: ``(batch_size)``
            - summary_ids: ``(batch_size, max_que_len)``
            - summary_lens: ``(batch_size)``
            - target_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.training:
            return self._run_forward_ml(code_ids,
                                        code_lens,
                                        summary_ids,
                                        summary_lens,
                                        target_seq,
                                        src_map,
                                        alignment,
                                        **kwargs)

        else:
            return self.decode(code_ids,
                               code_lens,
                               src_map,
                               alignment,
                               **kwargs)

    def con2src(self, t, src_vocabs):
        words = []
        for idx, w in enumerate(t):
            token_id = w[0].item()
            if token_id < self.vocab_size:
                # TODO check if correct
                words.append(self.tokenizer.convert_ids_to_tokens(token_id))
            else:
                token_id = token_id - self.vocab_size
                words.append(src_vocabs[idx][token_id])
        return words

    def __greedy_sequence(self,
                          params,
                          choice='greedy',
                          tgt_words=None):

        memory_bank = params['memory_bank']
        ext_outs = params['ext_outs']
        ext_lens = params['ext_lens']
        batch_size = memory_bank.size(0)

        if tgt_words is None:
            tgt_words = torch.LongTensor([self.tokenizer.cls_token_id]).to(memory_bank.device, non_blocking=True)
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1).long()  # B x 1

        dec_predictions = []
        copy_info = []
        attentions = []
        dec_log_prob = []
        acc_dec_outs = []

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]

        max_ext_len = None
        if self.include_ast or self.include_graph:
            max_ext_len = ext_outs[0].shape[1] \
                if isinstance(ext_outs, list) else ext_outs.shape[1]

        dec_states = self.decoder.init_decoder(params['src_len'], max_mem_len, ext_lens, max_ext_len)

        attn = {"coverage": None}

        enc_outputs = memory_bank

        # +1 for <EOS> token
        tgt_words_for_emb = copy.deepcopy(tgt_words).to(tgt_words.device, non_blocking=True)
        for idx in range(self.config.max_len + 1):
            tgt = self.encoder.embeddings(tgt_words_for_emb)[:, -1:, :]
            tgt_pad_mask = tgt_words.data.eq(self.tokenizer.pad_token_id)
            layer_wise_dec_out, attn = self.decoder.decode(tgt_pad_mask,
                                                           tgt,
                                                           enc_outputs,
                                                           dec_states,
                                                           step=idx,
                                                           layer_wise_coverage=attn['coverage'],
                                                           ext_outs=ext_outs)

            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            decoder_outputs = torch.tanh(self.dense(decoder_outputs))
            if self._copy:
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                  params['memory_bank'],
                                                  memory_lengths=params['src_len'],
                                                  softmax_weights=False)

                # mask copy_attn weights here if needed
                if params['src_mask'] is not None:
                    mask = params['src_mask'].byte().unsqueeze(1)  # Make it broadcastable.
                    mask = (1 - mask).bool()
                    copy_score.data.masked_fill_(mask, -float('inf'))
                attn_copy = f.softmax(copy_score, dim=-1)
                prediction = self.copy_generator(decoder_outputs,
                                                 attn_copy,
                                                 params['src_map'])
                prediction = prediction.squeeze(1)
                for b in range(prediction.size(0)):
                    if params['blank'][b]:
                        blank_b = try_gpu(torch.LongTensor(params['blank'][b]))
                        fill_b = try_gpu(torch.LongTensor(params['fill'][b]))
                        prediction[b].index_add_(0, fill_b,
                                                 prediction[b].index_select(0, blank_b))
                        prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = f.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_prob.append(log_prob.squeeze(1))
            dec_predictions.append(tgt.squeeze(1).clone())
            if "std" in attn:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attn["std"], dim=1)
                attentions.append(std_attn.squeeze(2))
            if self._copy:
                mask = tgt.gt(self.vocab_size - 1)
                copy_info.append(mask.float().squeeze(1))

            words = self.con2src(tgt, params['source_vocab'])
            words = [self.tokenizer.convert_tokens_to_ids(w) for w in words]
            words = torch.Tensor(words).type_as(tgt)
            # print([i for i in words])
            # print(tgt_words_for_emb.size())
            # print(tgt_words.size())
            tgt_words = words.unsqueeze(1)
            tgt_words_for_emb = torch.cat([tgt_words_for_emb, tgt_words], dim=1)
            # print([i for i in tgt_words_for_emb])
            # print([self.tokenizer.decode(i) for i in tgt_words_for_emb])
        return dec_predictions, attentions, copy_info, dec_log_prob

    def __generate_sequence(self,
                            params):
        beam_size = self.config.beam_size
        all_memory_bank = params['memory_bank']
        src_lens = params['src_len']
        ext_out = params['ext_outs']
        ext_lens = params['ext_lens']

        batch_size = all_memory_bank.size(0)
        dec_predictions = []
        acc_dec_outs = []
        all_attentions = []
        all_copy_info = []
        for bid in range(batch_size):
            beam = Beam(beam_size, self.tokenizer.cls_token_id, self.tokenizer.eos_token_id)
            tgt_words = beam.getCurrentState().to(all_memory_bank.device).long()  # 1 x 1

            memory_bank = all_memory_bank[bid:bid + 1].repeat(beam_size, 1, 1)  # 1*seq*emb
            ext_bank = ext_out[bid:bid + 1].repeat(beam_size, 1, 1)
            src_len = src_lens[bid:bid + 1].repeat(beam_size)
            ext_len = ext_lens[bid:bid + 1].repeat(beam_size)

            enc_outputs = memory_bank
            src_map = params['src_map'][bid:bid + 1].repeat(beam_size, 1, 1)
            blank = params['blank'][bid:bid + 1] * beam_size
            fill = params['fill'][bid:bid + 1] * beam_size
            source_vocab = params['source_vocab'][bid:bid + 1] * beam_size
            src_mask = None
            attentions = None
            copy_info = None

            if params['src_mask'] is not None:
                src_mask = params['src_mask'][bid:bid + 1].repeat(beam_size, 1)

            # +1 for <EOS> token
            tgt_words_for_emb = copy.deepcopy(tgt_words).to(tgt_words.device)

            for idx in range(self.config.max_len + 1):
                if beam.done():
                    break
                tgt = self.encoder.embeddings(tgt_words_for_emb)
                tgt_pad_mask = tgt_words_for_emb.data.eq(self.tokenizer.pad_token_id)
                layer_wise_dec_out, attn = self.decoder(enc_outputs,
                                                        src_len,
                                                        tgt_pad_mask,
                                                        tgt,
                                                        ext_bank,
                                                        ext_len)
                decoder_outputs = layer_wise_dec_out[-1]
                acc_dec_outs.append(decoder_outputs.squeeze(1))
                decoder_outputs = torch.tanh(self.dense(decoder_outputs))
                decoder_outputs = decoder_outputs[:, idx:idx + 1, :]
                if self._copy:
                    _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                      enc_outputs,
                                                      memory_lengths=src_len,
                                                      softmax_weights=False)

                    # mask copy_attn weights here if needed
                    if src_mask is not None:
                        mask = src_mask.byte().unsqueeze(1)  # Make it broadcastable.
                        mask = (1 - mask).bool()
                        copy_score.data.masked_fill_(mask, -float('inf'))
                    attn_copy = f.softmax(copy_score, dim=-1)
                    prediction = self.copy_generator(decoder_outputs,
                                                     attn_copy,
                                                     src_map)
                    prediction = prediction.squeeze(1)
                    for b in range(prediction.size(0)):
                        if blank[b]:
                            blank_b = try_gpu(torch.LongTensor(blank[b]))
                            fill_b = try_gpu(torch.LongTensor(fill[b]))
                            prediction[b].index_add_(0, fill_b,
                                                     prediction[b].index_select(0, blank_b))
                            prediction[b].index_fill_(0, blank_b, 1e-10)
                else:
                    prediction = self.generator(decoder_outputs.squeeze(1))
                    prediction = torch.log_softmax(prediction, dim=1)

                beam.advance(prediction.data)
                tgt = beam.getCurrentState()
                tgt_words_for_emb.data.copy_(tgt_words_for_emb.data.index_select(0, beam.getCurrentOrigin()))

                if "std" in attn:
                    # std_attn: batch_size x num_heads x 1 x src_len
                    std_attn = torch.stack(attn["std"], dim=1)
                    attentions = std_attn.squeeze(2)

                if self._copy and idx > 1:
                    mask = tgt_words_for_emb.gt(self.vocab_size - 1)
                    copy_info = mask.float().squeeze(1)

                words = self.con2src(tgt, source_vocab)
                words = [self.tokenizer.convert_tokens_to_ids(w) for w in words]
                words = torch.Tensor(words).type_as(tgt)
                # print([i for i in words])
                # print(tgt_words_for_emb.size())
                # print(tgt_words.size())
                tgt_words = words.unsqueeze(1)
                tgt_words_for_emb = torch.cat([tgt_words_for_emb, tgt_words], dim=1)
                # print([i for i in tgt_words_for_emb])
                # print([self.tokenizer.decode(i) for i in tgt_words_for_emb])
            # exit(0)
            dec_predictions.append(tgt_words_for_emb[0][1:])
            all_attentions.append(attentions[0].transpose(0, 1))
            all_copy_info.append(copy_info[0])
        return dec_predictions, all_attentions, all_copy_info, None

    def decode(self,
               code_ids,
               code_lens,
               src_map,
               alignment,
               **kwargs):
        # word_rep = self.encoder.embeddings(code_word_rep)
        source_mask = kwargs['code_mask'].to(code_ids.device)
        # target_mask = kwargs['summary_mask'].to(code_ids.device)
        # B x seq_len x h
        outputs = self.encoder(code_ids, attention_mask=source_mask)
        memory_bank = outputs[0]
        # memory_bank, layer_wise_outputs = self.encoder(code_word_rep)  # B x seq_len x h

        batch_starts = kwargs['batch_starts']
        batch_nodes = kwargs['batch_nodes']
        batch_ends = kwargs['batch_ends']
        seq_lens = kwargs['seq_lens']
        reverse_index = kwargs['reverse_index']
        node_lens = kwargs['node_lens']

        # Encoder
        ext_lens = None
        ext_out = None
        if self.include_ast:
            ext_out, ast_hidden = \
                self.ast_encoder(batch_starts, batch_nodes, batch_ends, seq_lens, reverse_index, node_lens)
            ext_lens = torch.LongTensor(seq_lens).to(ext_out.device)

        if self.include_graph:
            nodes = kwargs['nodes']
            graph_node_lens = kwargs['graph_node_lens']
            node_token_ids = kwargs['node_token_ids'].to(source_mask.device)
            node_token_lens = kwargs['node_token_lens']
            node_types = kwargs['node_types'].to(source_mask.device)
            edges = kwargs['edges']
            edges_attrs = kwargs['edge_attrs']

            ext_out = self.graph_encoder(nodes, graph_node_lens, node_token_ids,
                                         node_token_lens, node_types,
                                         edges, edges_attrs, self.encoder.embeddings)
            ext_lens = torch.LongTensor(graph_node_lens).to(ext_out.device)

        params = dict()
        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = None
        params['src_len'] = code_lens
        params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map
        params['src_mask'] = kwargs['code_mask']
        params['fill'] = kwargs['fill']
        params['blank'] = kwargs['blank']
        params['src_words'] = code_ids
        params['ext_outs'] = ext_out
        params['ext_lens'] = ext_lens

        # params['batch_starts'] = kwargs['batch_starts']
        # params['batch_nodes'] = kwargs['batch_nodes']
        # params['batch_ends'] = kwargs['batch_ends']
        # params['seq_lens'] = kwargs['seq_lens']
        # params['reverse_index'] = kwargs['reverse_index']
        # params['node_lens'] = kwargs['node_lens']

        generate_sequence = self.__greedy_sequence if self.config.beam_size <= 1 else self.__generate_sequence
        dec_predictions, attentions, copy_info, _ = generate_sequence(params)
        if self.config.beam_size <= 1:
            dec_predictions = torch.stack(dec_predictions, dim=1)
            copy_info = torch.stack(copy_info, dim=1) if copy_info else None
            # attentions: batch_size x tgt_len x num_heads x src_len
            attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_predictions,
            'copy_info': copy_info,
            'memory_bank': memory_bank,
            'attentions': attentions,
        }
