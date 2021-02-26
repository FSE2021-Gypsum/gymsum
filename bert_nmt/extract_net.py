import torch
from torch import nn
from utils.utils import try_gpu
import bert_nmt.pytorch_util as ptu
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
import logging
from utils.configs import ConfigDict
import copy
import pdb
from bert_nmt.data_utils import sort_sequence

logger = logging.getLogger(__name__)


class ExtractNet(nn.Module):
    def __init__(self, config, tokenizer):
        super(ExtractNet, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.setting = config.extract
        self.optimizer = None

        self.token_embedding = nn.Embedding(self.vocab_size, self.setting.embed_size,
                                            padding_idx=tokenizer.pad_token_id)

        self.rnn = nn.LSTM(
            input_size=self.setting.embed_size,
            hidden_size=self.setting.hidden_size,
            bidirectional=self.setting.bid,
            batch_first=True,
            num_layers=self.setting.n_layers,
            dropout=self.setting.dropout
        )

        dims = self.setting.dim.to_vanilla_()
        mlp = []
        input_dim = self.setting.hidden_size * 2

        for hidden_size in dims:
            mlp.append(nn.Dropout(self.setting.dropout))
            mlp.append(nn.Linear(input_dim, hidden_size))
            mlp.append(nn.ReLU())
            input_dim = hidden_size
        mlp.append(nn.Linear(input_dim, 2))

        self.generation = nn.Sequential(*mlp)
        self.updates = 0

    def forward(self, ex, generate_model):
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.train()

        code_ids = try_gpu(ex['code_ids'])
        code_lens = try_gpu(ex['code_lens'])
        code_mask = try_gpu(ex['code_mask'])

        batch_size, seq_len = code_ids.size()

        code_embed = self.token_embedding(code_ids)
        inputs, sorted_seq_lengths, desorted_indices = sort_sequence(code_embed, code_lens,
                                                                     batch_first=True)
        pad_code_embed = pack_padded_sequence(code_embed, ptu.to_numpy(sorted_seq_lengths), batch_first=True)
        outs, hidden = self.rnn(pad_code_embed)
        outs, _ = pad_packed_sequence(outs, batch_first=True, total_length=seq_len)
        outs = outs[desorted_indices]

        x_pred = torch.softmax(self.generation(outs), dim=-1)
        for i, code_len in enumerate(code_lens):
            code_mask[i, code_len - 1] = 0
            code_mask[i, 0] = 0
        code_mask = ~(code_mask.bool().unsqueeze(-1))
        x_pred = x_pred.masked_fill(code_mask, 1)
        pred_prob, pred_action = x_pred.max(-1)
        log_prob_sum = torch.log(pred_prob + 1e-20).sum(-1)
        x_hats = []

        for i, code_len in enumerate(code_lens):
            x_hat = []
            for j in range(0, code_len - 1):
                if not pred_action[i][j]:
                    x_hat.append(int(ptu.to_numpy(code_ids[i][j])))
            x_hat.append(self.tokenizer.eos_token_id)
            x_hat += [self.tokenizer.pad_token_id] * (seq_len - len(x_hat))
            x_hats.append(x_hat)

        x_hats = ptu.from_numpy(np.array(x_hats)).to(code_ids.device)
        code_mask = (x_hats != self.tokenizer.pad_token_id)
        code_lens = code_mask.sum(dim=-1)

        ex['code_ids'] = x_hats.long()
        ex['code_mask'] = code_mask
        ex['code_lens'] = code_lens
        print('\n')
        print(self.tokenizer.decode(list(ptu.to_numpy(code_ids[0]))))
        print(self.tokenizer.decode(list(ptu.to_numpy(x_hats[0]))))

        # print(pred_action)
        # print(pred_prob)

        generate_model.network.eval()
        with torch.no_grad():
            log_prob = generate_model.calculate(ex, self.tokenizer)

        loss = log_prob_sum.mul(log_prob).sum() / batch_size

        ml_loss = loss.item()

        loss.backward()

        clip_grad_norm_(self.parameters(), self.setting.grad_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1

        return {
            'ml_loss': ml_loss,
            'perplexity': 0.0
        }

    def pred(self, ex, generate_model):
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.eval()

        code_ids = try_gpu(ex['code_ids'])
        code_lens = try_gpu(ex['code_lens'])
        code_mask = try_gpu(ex['code_mask'])

        batch_size, seq_len = code_ids.size()

        code_embed = self.token_embedding(code_ids)
        inputs, sorted_seq_lengths, desorted_indices = sort_sequence(code_embed, code_lens,
                                                                     batch_first=True)
        pad_code_embed = pack_padded_sequence(code_embed, ptu.to_numpy(sorted_seq_lengths), batch_first=True)
        outs, hidden = self.rnn(pad_code_embed)
        outs, _ = pad_packed_sequence(outs, batch_first=True, total_length=seq_len)
        outs = outs[desorted_indices]
        x_pred = torch.softmax(self.generation(outs), dim=-1)
        for i, code_len in enumerate(code_lens):
            code_mask[i, code_len - 1] = 0
            code_mask[i, 0] = 0
        code_mask = ~(code_mask.bool().unsqueeze(-1))
        x_pred = x_pred.masked_fill(code_mask, 1)
        pred_prob, pred_action = x_pred.max(-1)
        log_prob_sum = pred_prob.log().sum(-1)
        x_hats = []

        for i, code_len in enumerate(code_lens):
            x_hat = [self.tokenizer.bos_token_id]
            for j in range(1, code_len - 1):
                if not pred_action[i][j]:
                    x_hat.append(int(ptu.to_numpy(code_ids[i][j])))
            x_hat.append(self.tokenizer.eos_token_id)
            x_hat += [self.tokenizer.pad_token_id] * (seq_len - len(x_hat))
            x_hats.append(x_hat)

        x_hats = ptu.from_numpy(np.array(x_hats)).to(code_ids.device)
        code_mask = (x_hats != self.tokenizer.pad_token_id)
        code_lens = code_mask.sum(dim=-1)

        ex['code_ids'] = x_hats.long()
        ex['code_mask'] = code_mask
        ex['code_lens'] = code_lens

        generate_model.network.eval()
        return generate_model.predict(ex, self.tokenizer, replace_unk=True)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.setting.optimizer == 'sgd':
            parameters = [p for p in self.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(parameters,
                                       self.setting.learning_rate,
                                       momentum=self.setting.momentum,
                                       weight_decay=self.setting.weight_decay)

        elif self.setting.optimizer == 'adam':
            parameters = [p for p in self.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(parameters,
                                        self.setting.learning_rate,
                                        weight_decay=self.setting.weight_decay)

        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.setting.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = try_gpu(v)

    def checkpoint(self, filename, epoch):

        params = {
            'state_dict': self.state_dict(),
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
    def load(filename, tokenizer):
        logger.info('Loading extract model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        args = ConfigDict(saved_params['args'])
        return ExtractNet(args, tokenizer)

    @staticmethod
    def load_checkpoint(filename, config, tokenizer, use_gpu=True):
        logger.info('Loading extract model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer = saved_params['optimizer']
        state_dict = saved_params['state_dict']
        model = ExtractNet(config, tokenizer)
        model.load_state_dict(state_dict)
        model.updates = updates
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    def save(self, filename):
        state_dict = copy.copy(self.state_dict())
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
