import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tagger_models import *

from transformers import *

from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import precision_recall_fscore_support

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    


class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting probabilistic (float) targets

    Args:
        weight: a tensor of relative weights to assign to each class.
            the kwarg name 'weight' is used to match CrossEntropyLoss
        reduction: how to combine the elementwise losses
            'none': return an unreduced list of elementwise losses
            'mean': return the mean loss per elements
            'sum': return the sum of the elementwise losses

    Accepts:
        input: An [n, k] float tensor of prediction logits (not probabilities)
        target: An [n, k] float tensor of target probabilities
    """

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        # Register as buffer is standard way to make sure gets moved /
        # converted with the Module, without making it a Parameter
        if weight is None:
            self.weight = None
        else:
            # Note: Sets the attribute self.weight as well
            if torch.cuda.is_available():
                self.register_buffer("weight", torch.FloatTensor(weight).cuda())
            else:
                self.register_buffer("weight", torch.FloatTensor(weight))
        self.reduction = reduction

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input, target):
        n, k = input.shape
        # Note that t.new_zeros, t.new_full put tensor on same device as t
        cum_losses = input.new_zeros(n)
        for y in range(k):
            cls_idx = input.new_full((n,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, cls_idx, reduction="none")
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            if torch.cuda.is_available():
                target = target.cuda()
            cum_losses += target[:, y].float() * y_loss
        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")


def load_data(fname, use_soft_tag=False):
    data, sent = [], []
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line=='':
                temp = line.split('\t')
                if use_soft_tag:
                    sent.append((temp[0], [float(x) for x in temp[-1].split(' ')[:2]]))
                else:
                    if temp[-1]=='B':
                        temp[-1] = 'I'
                    sent.append((temp[0], temp[-1]))
            else:
                data.append(sent)
                sent = []
        
        if len(sent)>0:
            data.append(sent)
    return data

def score_sequence_span_level(predicted_labels, gold_labels):
    if len(predicted_labels) != len(gold_labels):
        raise ValueError("Lengths of predicted_labels and gold_labels must match")

    tp, fp, fn = 0, 0, 0
    # Collects predicted and correct spans for the instance
    predicted_spans, correct_spans = set(), set()
    data = ((predicted_labels, predicted_spans), (gold_labels, correct_spans))
    for labels, spans in data:
        start = None
        tag = None
        for i in range(len(labels)):
            if labels[i][0] == 'I':
                # Two separate conditional statements so that 'I' is always
                # recognized as a valid label
                if start is None:
                    start = i
                    tag = labels[i]
                # Also checks if label has switched to new type
                elif tag != labels[i]:
                    spans.add((start, i, tag))
                    start = i
                    tag = labels[i]
            elif labels[i][0] == 'O' or labels[i] == 'ABS':
                if start is not None:
                    spans.add((start, i, tag))
                start = None
                tag = None
            elif labels[i][0] == 'B':
                if start is not None:
                    spans.add((start, i, tag))
                start = i
                tag = labels[i]
            else:
                raise ValueError("Unrecognized label: %s" % labels[i] )

        # Closes span if still active
        if start is not None:
            spans.add((start, len(labels), tag))

    # Compares predicted spans with correct spans
    for span in correct_spans:
        if span in predicted_spans:
            tp += 1
            predicted_spans.remove(span)
        else:
            fn += 1
    fp += len(predicted_spans)

    return tp, fp, fn

def make_batch_featureX(batch_sentences, is_train_data=True):
    texts = []
    for sentence in batch_sentences:
        texts.append(' '.join([tk[0] for tk in sentence]))
    s_encoded = tokenizer.batch_encode_plus(texts, add_special_tokens=False)

    ids_list = []
    attn_mask_list = []
    tag_list = []
    token_start_idx_list = []
    tokens_list = []
    for i, sent in enumerate(batch_sentences):
        ids = s_encoded['input_ids'][i]
        attn = s_encoded['attention_mask'][i]
        ids_list.append(ids)
        attn_mask_list.append(attn)
        
        bert_tokens = tokenizer.convert_ids_to_tokens(ids)
        tokens_list.append(bert_tokens)
        
        token_start_idx = []
        for j in range(len(bert_tokens)):
            if not bert_tokens[j].startswith('##'):
                token_start_idx.append(j)
        token_start_idx_list.append(token_start_idx)
        
        tags = [item[1] for item in sent]
        if is_train_data and use_crf:
            tags = [item[1] + [0.0, 0.0] for item in sent] # add for start and stop tag
        
        bert_tags = []
        j, k = 0, 0
        while j<len(bert_tokens):
            if not bert_tokens[j].startswith('##'):
                bert_tags.append(tags[k])
                k+=1
            else:
                bert_tags.append(tags[k-1])
            j+=1
        
        if is_train_data:
            tag_idx = [t for t in bert_tags]
        else:
            tag_idx = [tag2idx[t] for t in bert_tags]
        tag_idx_tensor = torch.Tensor(tag_idx).to(device)
        tag_list.append(tag_idx_tensor)
    
    max_len = max(len(ids) for ids in ids_list)
    lengths = [len(ids) for ids in ids_list]
    for i in range(len(batch_sentences)):
        cur_len = len(ids_list[i])
        ids_list[i].extend([0 for _ in range(max_len-cur_len)])
        attn_mask_list[i].extend([0 for _ in range(max_len-cur_len)])
    s_tensor = torch.LongTensor(ids_list).to(device)
    attn_mask_list = torch.Tensor(attn_mask_list).to(device)
    
    return tokens_list, s_tensor, attn_mask_list, tag_list, token_start_idx_list, lengths
        
    
def train(batch, use_crf=True):
    optimizer.zero_grad()
    tokens_batch, feature_batch, attn_mask_batch, tag_batch_prob, token_start_idx_list, lengths = make_batch_featureX(batch, is_train_data=True)
    out = tagger(feature_batch, attn_mask_batch, tokens_batch)
    loss_batch = 0.0
    
    if use_crf:
        tag_batch_crf = []
        for tag_tensor in tag_batch_prob:
            _, idx = torch.max(tag_tensor, dim=1)
            tag_batch_crf.append(idx)
        loss_batch = tagger.crf.neg_log_likelihood(out, tag_batch_crf, tag_batch_prob, lengths)
    else:
        for i in range(len(lengths)):
            sent_length = lengths[i]
            sent_feature = out[i, :sent_length, :]
            tag_tensor_prob = tag_batch_prob[i]
            token_positions = token_start_idx_list[i]
            loss_batch+=soft_cross_entropy(sent_feature[token_positions, :], tag_tensor_prob[token_positions, :])
    
    cur_loss = loss_batch.item()
    loss_batch.backward()
    torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return cur_loss


def evaluate(sentences, use_crf=True):
    all_confidences = []
    all_pred_tags = []
    all_gold_tags = []
    
    with torch.no_grad():
        batches = [sentences[x:x + batch_size] for x in range(0, len(sentences), batch_size)]
        for batch_no, batch in enumerate(batches):
            tokens_batch,feature_batch, attn_mask_batch, tag_batch, token_start_idx_list, lengths = make_batch_featureX(batch, is_train_data=False)
            
            out = tagger(feature_batch, attn_mask_batch, tokens_batch)
            if not use_crf:
                softmax = F.softmax(out, dim=2)
                for i in range(len(batch)):
                    sent_length = lengths[i]
                    sent_feature = out[i,:sent_length, :]
                    tag_tensor = tag_batch[i]
                    pred_tag, confidences = [], []
                    for j in range(sent_length):
                        _, idx = torch.max(softmax[i, j, :], 0)
                        pred_tag.append(idx.item())
                        confidences.append(softmax[i, j, idx.item()])
                   
                    token_positions = token_start_idx_list[i]
                    recovered_pred_tag = [idx2tag[pred_tag[int(ix)]] for ix in token_positions]
                    recovered_gold_tag = [idx2tag[tag_tensor.tolist()[ix]] for ix in token_positions]
                    recovered_confidences = [confidences[ix] for ix in token_positions]
                    
                    all_pred_tags.append(recovered_pred_tag)
                    all_gold_tags.append(recovered_gold_tag)
                    all_confidences.append(recovered_confidences)
            else:
                
                for i in range(len(batch)):
                    sent_length = lengths[i]
                    sent_feature = out[i,:sent_length, :]
                    tag_tensor = tag_batch[i][:sent_length]
                    confidences, pred_tag = tagger.crf.viterbi_decode(sent_feature)

                    token_positions = token_start_idx_list[i]
                    recovered_pred_tag = [idx2tag[int(pred_tag[ix])] for ix in token_positions]   
                    recovered_gold_tag = [idx2tag[tag_tensor.tolist()[int(ix)]] for ix in token_positions]
                    recovered_confidences = [confidences[ix] for ix in token_positions]
                                        
                    all_pred_tags.append(recovered_pred_tag)
                    all_gold_tags.append(recovered_gold_tag)
                    all_confidences.append(recovered_confidences)
    
    return all_confidences, all_pred_tags, all_gold_tags

max_sent_length = 128
dataset = 'NCBI'
train_data = load_data('../datasets/{}/train-bert-{}.txt'.format(dataset, max_sent_length), use_soft_tag=True)
dev_data = load_data('../datasets/{}/dev-bert-{}.txt'.format(dataset, max_sent_length), use_soft_tag=False)
test_data = load_data('../datasets/{}/test-bert-{}.txt'.format(dataset, max_sent_length), use_soft_tag=False)

use_crf = True

if use_crf:
    idx2tag  = {0:"O", 1:'I', 2:'<START>', 3:"<STOP>"} # for make_batch_featureX
else:
    idx2tag  = {0:"O", 1:'I'} # for make_batch_featureX (no CRF)

tag2idx = {v:k for k,v in idx2tag.items()}


# bert_model_name = 'bert-base-uncased'
bert_model_name = 'allenai/scibert_scivocab_uncased'


tagger = BertTagger(
    bert_model_name, tag2idx=tag2idx, max_sent_length=max_sent_length, num_hidden_layers=12, use_crf=use_crf
).to(device)


if bert_model_name=='allenai/scibert_scivocab_uncased':
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
else:
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

batch_size = 8
lr = 1e-4
n_epoch = 30

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in tagger.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.9,
    },
    {"params": [p for n, p in tagger.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=50, num_training_steps=n_epoch*len(train_data)//batch_size
)

# optimizer = torch.optim.Adam(tagger.parameters(), lr=lr)
# patience = 3
# scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, mode='max', verbose=True)

param_size = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
print('model size (trainable parameters): {}'.format(param_size))

soft_cross_entropy = SoftCrossEntropyLoss()

prev_lr = lr
best_f1_dev, best_epoch_dev = 0.0, 0
best_f1_test, best_epoch_test = 0.0, 0
best_f1_test_strict, best_epoch_test_strict = 0.0, 0

best_state_dict = None
for epoch in range(n_epoch):
    
    random.shuffle(train_data)
    batches = [train_data[x:x + batch_size] for x in range(0, len(train_data), batch_size)]
    tagger.train()
    
    current_loss, seen_sentences, modulo = 0.0, 0, max(1, int(len(batches) / 10))
    for batch_no, sent_batch in enumerate(batches):
        loss_batch = train(sent_batch, use_crf=use_crf)
        current_loss += (loss_batch)
        seen_sentences += len(sent_batch)
        if batch_no % modulo == 0:
            print(
                "epoch {0} - iter {1}/{2} - lr {3} - loss {4:.6f}".format(
                    epoch + 1, batch_no, len(batches), lr, current_loss / seen_sentences
                )
            )
            iteration = epoch * len(batches) + batch_no 
    current_loss /= len(train_data)
    
    
    print("-------------------------- (DEV) -----------------------------")
    tagger.eval()
    all_confidences, all_pred_tags, all_gold_tags = evaluate(dev_data, use_crf=use_crf)
    
    tp_dev, fp_dev, fn_dev = 0, 0, 0
    for i in range(len(all_pred_tags)):
        
        tp, fp, fn = score_sequence_span_level(all_pred_tags[i], all_gold_tags[i])
        tp_dev += tp
        fp_dev += fp
        fn_dev += fn
    
    print("TP: {}, FP: {}, FN: {}".format(tp_dev, fp_dev, fn_dev))
    if tp_dev+fp_dev>0 and tp_dev+fn_dev>0:
        prec = tp_dev/(tp_dev+fp_dev)
        recall = tp_dev/(tp_dev+fn_dev)
        cur_f1_dev = 2*prec*recall/(prec+recall)
        if cur_f1_dev>best_f1_dev:
            best_f1_dev = cur_f1_dev
            best_epoch_dev = epoch
            
            with torch.no_grad():
                best_state_dict = tagger.state_dict()
                
        print("Current (Dev): P: {}, R: {}, F1: {}".format(prec, recall, cur_f1_dev))
        print("Best (Dev): F1: {} on epoch {}".format(best_f1_dev, best_epoch_dev))
    else:
        print("Current (Dev): P: {}, R: {}, F1: {}".format(None, None, None))
        print("Best (Dev): F1: {} on epoch {}".format(None, None))
    
    
    print("-------------------------- (TEST) -----------------------------")
    tagger.eval()
    all_confidences, all_pred_tags, all_gold_tags = evaluate(test_data, use_crf=use_crf)
    tp_test, fp_test, fn_test = 0, 0, 0
    for i in range(len(all_pred_tags)):
        tp, fp, fn = score_sequence_span_level(all_pred_tags[i], all_gold_tags[i])
        tp_test += tp
        fp_test += fp
        fn_test += fn
    print("TP: {}, FP: {}, FN: {}".format(tp_test, fp_test, fn_test))
    if tp_test+fp_test>0 and tp_test+fn_test>0:
        prec = tp_test/(tp_test+fp_test)
        recall = tp_test/(tp_test+fn_test)
        cur_f1_test = 2*prec*recall/(prec+recall)
        if cur_f1_test>best_f1_test:
            best_f1_test = cur_f1_test
            best_epoch_test = epoch
        print("Current (Test): P: {}, R: {}, F1: {}".format(prec, recall, cur_f1_test))
        print("Best (Test): F1: {} on epoch {}".format(best_f1_test, best_epoch_test))
    else:
        print("Current (Test): P: {}, R: {}, F1: {}".format(None, None, None))
        print("Best (Test): F1: {} on epoch {}".format(None, None))
    
#     scheduler.step(best_f1_dev)
#     try:
#         bad_epochs = scheduler.num_bad_epochs
#     except:
#         bad_epochs = 0
#         for group in optimizer.param_groups:
#             lr = group["lr"]
#         if lr != prev_lr:
#             bad_epochs = patience + 1


print('training finished. ')
print('loading best model to evaluate test data.')
tagger.load_state_dict(best_state_dict)
scores, all_pred_tags, all_gold_tags = evaluate(test_data, use_crf=use_crf)
tp_test, fp_test, fn_test = 0, 0, 0
for i in range(len(all_pred_tags)):
    tp, fp, fn = score_sequence_span_level(all_pred_tags[i], all_gold_tags[i])
    tp_test += tp
    fp_test += fp
    fn_test += fn
print("TP: {}, FP: {}, FN: {}".format(tp_test, fp_test, fn_test))
if tp_test+fp_test>0 and tp_test+fn_test>0:
    prec = tp_test/(tp_test+fp_test)
    recall = tp_test/(tp_test+fn_test)
    f1 = 2*prec*recall/(prec+recall)
    print("Final (Test): P: {}, R: {}, F1: {}".format(prec, recall, f1))
else:
    print("Final (Test): P: {}, R: {}, F1: {}".format(None, None, None))