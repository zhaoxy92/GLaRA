import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import *


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

START_TAG: str = '<START>'
STOP_TAG: str = '<STOP>'
    
    
import string

chars = string.ascii_letters + string.digits + string.punctuation.strip()

char2idx = {"<pad>":0}
for c in chars:
    char2idx[c]  = len(char2idx)
idx2char = {v:k for k,v in char2idx.items()}


class BertTagger(torch.nn.Module):
    def __init__(self, bert_model_name, tag2idx, num_hidden_layers=2, max_sent_length=100, use_crf=True):
        super(BertTagger, self).__init__()
    
        if bert_model_name=='allenai/scibert_scivocab_uncased':
            self.bert = AutoModel.from_pretrained(bert_model_name)
        else:
            self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=False, num_hidden_layers=num_hidden_layers, num_attention_heads=12)
        
        self.char_emb_dim = 50
        self.char_emb = torch.nn.Embedding(len(char2idx), self.char_emb_dim)
        self.conv1_padding = 0
        self.dilation = 1
        self.kernel_size = (3,1)
        self.stride = 1
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=self.kernel_size,
                            stride=self.stride, dilation=self.dilation,
                            bias=True, padding=self.conv1_padding, padding_mode='rand'),
            torch.nn.ReLU(),
            
        )
        self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv1_out_dim = (self.char_emb_dim+2*self.conv1_padding-self.dilation*(self.kernel_size[0]-1))//3
        print("char embedding dim: ", self.conv1_out_dim)
        self.linear = torch.nn.Linear(256*2, len(tag2idx))
        self.dropout = torch.nn.Dropout(0.1)
        self.rnn = torch.nn.LSTM(768+self.conv1_out_dim, 256, batch_first=True, bidirectional=True, num_layers=1)
        self.dropout2 = torch.nn.Dropout(0.1)


#         self.rnn = torch.nn.LSTM(768, 128, batch_first=True, bidirectional=True, num_layers=1)
#         self.linear = torch.nn.Linear(128*2, len(tag2idx))
#         self.dropout = torch.nn.Dropout(0.1)
#         self.dropout2 = torch.nn.Dropout(0.1)

    
        if use_crf:
            self.crf = CRF(len(tag2idx), tag_dict=tag2idx)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input, attention_mask, tokens_batch):
        self.zero_grad()
        out = self.bert(input, attention_mask=attention_mask)
        state = out[0]   
        state = self.dropout(state)
        
        ############# char embedding ########
        batch_size, batch_sent_length = state.shape[0], state.shape[1]
        batch_char_embedding = torch.zeros(batch_size, batch_sent_length, self.conv1_out_dim).to(device)
        for i, tokens in enumerate(tokens_batch):
            sent_length = len(tokens)
            max_char_length = max(len(tk) if not tk.startswith('##') else len(tk)-2 for tk in tokens)
            
            onehot = [[char2idx['<pad>'] for _ in range(max_char_length)] for _ in range(sent_length)]
            for tid, tk in enumerate(tokens):
                if tk.startswith('##'):
                    for cid in range(len(tk[2:])):
                        onehot[tid][cid] = char2idx[tk[2+cid]]
                else:
                    for cid in range(len(tk)):
                        onehot[tid][cid] = char2idx[tk[cid]]
                        
            onehot = torch.LongTensor(onehot).to(device)
            
            seq_char_embedding = self.char_emb(onehot)
            conv_seq = self.conv1(seq_char_embedding.unsqueeze(1))
            conv_seq = torch.mean(conv_seq, dim=1)
            pool_seq = F.max_pool1d(conv_seq, kernel_size=3)
            emb_seq = torch.max(pool_seq, dim=1)[0]
                        
            batch_char_embedding[i, :sent_length] = emb_seq
        
        state = torch.cat((state, batch_char_embedding), dim=2)
        #####################################
        
        
        rnn_output, (final_hidden, _) = self.rnn(state)
        rnn_output = self.dropout2(rnn_output)
        logit = self.linear(rnn_output)
        
        return logit

    
    
START_TAG: str = '<START>'
STOP_TAG: str = '<STOP>'


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list, type_=torch.FloatTensor):
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = type_(*shape)
    template.fill_(0)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, :lens_[i]] = tensor

    return template, lens_

class CRF(torch.nn.Module):
    def __init__(self, tagset_size, tag_dict):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.tag_dict = tag_dict
        self.transitions = torch.nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[self.tag_dict[START_TAG], :] = -10000.
        self.transitions.data[:, self.tag_dict[STOP_TAG]] = -10000.

        if torch.cuda.is_available():
            self.cuda()

    def neg_log_likelihood(self, rnn_out, tags, tags_prob, lengths):

        if torch.cuda.is_available():
            tags, _ = pad_tensors(tags, torch.cuda.LongTensor)
        else:
            tags, _ = pad_tensors(tags, torch.LongTensor)

        forward_score = self._forward_alg(rnn_out[:len(tags), :, :], lengths)
        gold_score = self._score_sentence(rnn_out[:len(tags), :, :], tags, tags_prob, lengths)

        score = torch.abs(forward_score - gold_score)

        return score.mean()

    def _forward_alg(self, feats, lens_):
        init_alphas = torch.Tensor(self.tagset_size).fill_(-10000.)
        init_alphas[self.tag_dict[START_TAG]] = 0.
        forward_var = torch.FloatTensor(feats.shape[0], feats.shape[1] + 1, feats.shape[2]).fill_(0)

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
        if torch.cuda.is_available():
            forward_var = forward_var.cuda()

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]
            tag_var = \
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2]) + \
                transitions + \
                forward_var[:, i, :][:, :, None].repeat(1, 1, transitions.shape[2]).transpose(2, 1)

            max_tag_var, _ = torch.max(tag_var, dim=2)
            tag_var = tag_var - max_tag_var[:, :, None].repeat(1, 1, transitions.shape[2])

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        terminal_var = forward_var + \
                       self.transitions[self.tag_dict[STOP_TAG]][None, :].repeat(forward_var.shape[0],
                                                                                                  1)

        alpha = log_sum_exp_batch(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, tags_prob, lens_):
                
        start = torch.LongTensor([self.tag_dict[START_TAG]])
        start = start[None, :].repeat(tags.shape[0], 1)
        stop = torch.LongTensor([self.tag_dict[STOP_TAG]])
        stop = stop[None, :].repeat(tags.shape[0], 1)
        if torch.cuda.is_available():
            start = start.cuda()
            stop = stop.cuda()

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]:] = self.tag_dict[STOP_TAG]

        score = torch.FloatTensor(feats.shape[0])

        if torch.cuda.is_available():
            score = score.cuda()


        start_prob, end_prob = torch.Tensor([1.0]), torch.Tensor([1.0])
        if torch.cuda.is_available():
            start_prob = start_prob.cuda()
            end_prob = end_prob.cuda()


        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i]))
            if torch.cuda.is_available():
                r = r.cuda()
            
            if tags_prob:
                feats_prob = feats[i, r, tags[i, :lens_[i]]] * tags_prob[i][r, tags[i, :lens_[i]]]

                pad_start_tags_prob = torch.cat((start_prob, tags_prob[i][r, tags[i, :lens_[i]]]))
                pad_end_tags_prob = torch.cat((tags_prob[i][r, tags[i, :lens_[i]]], end_prob))

                score[i] = \
                    torch.sum(
                        self.transitions[pad_stop_tags[i, :lens_[i] + 1], pad_start_tags[i, :lens_[i] + 1]] * pad_start_tags_prob * pad_end_tags_prob
                    ) + torch.sum(feats[i,:lens_[i],:] * tags_prob[i])
#                     torch.sum(feats_prob)
                    
            else:
                score[i] = \
                torch.sum(
                    self.transitions[pad_stop_tags[i, :lens_[i] + 1], pad_start_tags[i, :lens_[i] + 1]]
                ) + \
                torch.sum(feats[i, r, tags[i, :lens_[i]]])

        return score

    def viterbi_decode(self, feats):
        backpointers, backscores = [], []

        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.0)
        init_vvars[0][self.tag_dict[START_TAG]] = 0
        forward_var = init_vvars
        if torch.cuda.is_available():
            forward_var = forward_var.cuda()

        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            # bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            # next_tag_var = next_tag_var.data.cpus().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_dict[STOP_TAG]]
        terminal_var.data[self.tag_dict[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_dict[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))

        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())

        start = best_path.pop()
        assert start == self.tag_dict[START_TAG]
        best_path.reverse()
        return best_scores, best_path