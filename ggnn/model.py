import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch_geometric.nn import GatedGraphConv

from code2seq.c2q_utils import *


class Encoder(torch.nn.Module):
    def __init__(self, config, out_size):
        super(Encoder, self).__init__()
        self.type_vocab_size = config.node_types
        self.max_node_len = config.max_node_len
        self.type_dim = config.type_embedding
        self.embedding_dim = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout
        self.type_embed = nn.Embedding(self.type_vocab_size, self.type_dim)
        self.embed_fusion = nn.Linear(self.type_dim + self.embedding_dim, self.embedding_dim)
        self.graph_layer = GatedGraphConv(self.hidden_size, self.num_layers)
        self.out = nn.Linear(self.hidden_size, out_size)
        self.embed_dropout = nn.Dropout(self.dropout_rate)

    def forward(self, nodes, graph_node_lens, node_token_ids,
                node_token_lens, node_types,
                edges, edges_attrs, bert_embedding):
        batch_size = len(nodes)
        type_embed = self.type_embed(node_types)
        text_embedding = bert_embedding.word_embeddings(node_token_ids)
        res = torch.zeros((batch_size, min(type_embed.size(1), self.max_node_len), self.embedding_dim)).to(type_embed.device)
        for i in range(batch_size):
            current_node_token_lens = node_token_lens[i]
            current_node_lens = graph_node_lens[i]
            current_node_type_embed = type_embed[i][:current_node_lens]
            current_text_embed = torch.cat([text_embedding[i][sum(current_node_token_lens[:j]):
                                                              sum(current_node_token_lens[:j + 1])].mean(
                dim=0).unsqueeze(0) for j, _ in enumerate(current_node_token_lens)], dim=0)
            current_embed = self.embed_fusion(torch.cat([current_node_type_embed, current_text_embed], dim=-1))
            graph_output = self.graph_layer(current_embed,
                                            torch.LongTensor(edges[i]).to(type_embed.device),
                                            torch.LongTensor(edges_attrs[i]).to(type_embed.device))

            res[i][:min(current_node_lens, self.max_node_len)] = graph_output[
                                                                 :min(current_node_lens, self.max_node_len)]

        return res


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, rnn_dropout):
        """
        hidden_size : decoder unit size,
        output_size : decoder output size,
        rnn_dropout : dropout ratio for rnn
        """

        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=rnn_dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, seqs, hidden, graph_attention, sentence_attention):
        emb = self.embedding(seqs)
        _, hidden = self.gru(emb, hidden)

        output = torch.cat((hidden, graph_attention, sentence_attention), 2)
        output = self.out(output)

        return output, hidden


class EncoderDecoder_with_Attention(nn.Module):
    """Combine Encoder and Decoder"""

    def __init__(self, vocab_len, embedding_dim, hidden_size, output_size, device,
                 num_layers=2, rnn_dropout=0.5):
        super(EncoderDecoder_with_Attention, self).__init__()
        self.encoder = Encoder(vocab_len, embedding_dim, hidden_size, num_layers, rnn_dropout, device)
        self.decoder = Decoder(hidden_size, output_size, rnn_dropout)

        self.W_a = torch.rand((hidden_size, hidden_size), dtype=torch.float, device=device, requires_grad=True)

        self.device = device
        nn.init.xavier_uniform_(self.W_a)

    def forward(self, sentence_input, graph_input, edge_index, terget_max_length):
        # Encoder
        sentence_out, graph_out, encoder_hidden = \
            self.encoder(sentence_input, graph_input, edge_index)

        _batch_size = len(sentence_input)
        decoder_hidden = encoder_hidden

        # make initial input for decoder
        decoder_input = torch.tensor([BOS] * _batch_size, dtype=torch.long, device=self.device)
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)

        # output holder
        decoder_outputs = torch.zeros(terget_max_length, _batch_size, self.decoder.output_size, device=self.device)

        # print('=' * 20)
        for t in range(terget_max_length):
            # ct
            sentence_ct = self.attention(sentence_out, decoder_hidden)
            graph_ct = self.attention(graph_out, decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, graph_ct, sentence_ct)

            # print(decoder_output.max(-1)[1])

            decoder_outputs[t] = decoder_output

            decoder_input = decoder_output.max(-1)[1]

        return decoder_outputs

    def attention(self, encoder_output_bag, hidden):
        """
        encoder_output_bag : (batch, k, hidden_size) bag of embedded ast path
        hidden : (1 , batch, hidden_size):
        """

        ha = einsum('ijk,kt->ijt', encoder_output_bag, self.W_a)
        hd = hidden.transpose(0, 1)
        at = F.softmax(einsum('ijk,itk->ijt', ha, hd), dim=1)
        ct = torch.sum(torch.mul(encoder_output_bag, at.expand_as(encoder_output_bag)), dim=1).transpose(0, 1)
        return ct
