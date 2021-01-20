import pickle
from transformers import *

dataset = 'LaptopReview'
data_name = 'train'  # train, dev, test


train_file = 'train_data_link_hmm.p'
dev_file = 'dev_data.p'
test_file = 'test_data.p'

with open('output-gen/{}/{}'.format(dataset,train_file), 'rb') as f:
    train_data = pickle.load(f)
    print(len(train_data))
    
with open('output-gen/{}/{}'.format(dataset,dev_file), 'rb') as f:
    dev_data = pickle.load(f)
    print(len(dev_data))
    
with open('output-gen/{}/{}'.format(dataset,test_file), 'rb') as f:
    test_data = pickle.load(f)
    print(len(test_data))
    

doing_train_data = None
data = None

if data_name =='train':
    doing_train_data = True
    data = train_data
else:
    doing_train_data = False
    if data_name=='dev':
        data = dev_data
    else:
        data = test_data

bert_model_name = 'bert-base-uncased'
# bert_model_name = 'allenai/scibert_scivocab_uncased'

if bert_model_name=='allenai/scibert_scivocab_uncased':
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
else:
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

max_sent_len = 32
output_file = None
if data_name=='train':
    output_file = 'train-bert-{}.txt'.format(max_sent_len)    
elif data_name=='dev':
    output_file = 'dev-bert-{}.txt'.format(max_sent_len)   
else:
    output_file = 'test-bert-{}.txt'.format(max_sent_len)   

tokens_all, tags_all, tags_prob_all= [], [], []
for sent in data:
    tmp_tokens = [tk.text for tk in sent['tokens']]
    tmp_tags = sent['tags'].labels
    tmp_tags_prob = None
    if doing_train_data:
        pad_lengths = sent['unary_marginals'].get_padding_lengths()
        tmp_tags_prob = sent['unary_marginals'].as_tensor(pad_lengths).tolist()
        
    
    tokens, tags, tags_prob = [], [], []
    for ix, tk in enumerate(tmp_tokens):
        if len(tk.strip())>0:
            tokens.append(tk)
            tags.append(tmp_tags[ix])
            if doing_train_data:
                tags_prob.append(tmp_tags_prob[ix])
    
    length = len(tokens)
    start = 0
    for ix, tk in enumerate(tokens):
        if tk=='.' and 5<=ix<length-5 and all([tg=='O' for tg in tags[ix-5:ix+5]]):
            tokens_all.append(tokens[start:ix+1])
            tags_all.append(tags[start:ix+1])
            if doing_train_data:
                tags_prob_all.append(tags_prob[start:ix+1])
            
            start = ix+1
        elif ix==length-1:
            tokens_all.append(tokens[start:ix+1])
            tags_all.append(tags[start:ix+1])
            if doing_train_data:
                tags_prob_all.append(tags_prob[start:ix+1])
##################################################################################

def create_mapping(tokens, bert_tokens):
    assert len(tokens) <= len(bert_tokens)
    bert2normal = {}
    
    j = 0  # track current positions in tokens
    cur = '' # track current word covered in tokens[j]
    for i in range(len(bert_tokens)):
        tk = bert_tokens[i]
        if len(cur+tk.lower()) < len(tokens[j]):
            bert2normal[i] = j
            cur+=tk.lower()
        elif len(cur+tk.lower())==len(tokens[j]):
            bert2normal[i] = j
            cur  = ''
            j+=1
    return bert2normal


import copy

cnt = 0
with open('datasets/{}/{}'.format(dataset, output_file), 'w') as fw:
    batch_size = 1
    batches = []
    batch = []
    
    if doing_train_data:
        for tokens, tags, tags_prob in zip(tokens_all, tags_all, tags_prob_all):
            if len(batch)<batch_size:
                batch.append((tokens, tags, tags_prob))
            else:
                batches.append(batch)
                batch = []
        if len(batch)>0:
            batches.append(batch)
    else:
        for tokens, tags in zip(tokens_all, tags_all):
            if len(batch)<batch_size:
                batch.append((tokens, tags))
            else:
                batches.append(batch)
                batch = [(tokens, tags)]
        if len(batch)>0:
            batches.append(batch)    
    
    cnt = 0
    cached_bert_tokens, cached_bert_tags, cached_bert_tags_prob = [], [], []
    for batch in batches:
        tokens_batch = []
        tags_batch = []
        tags_prob_batch = []
        text_batch = []
        
        for sent in batch:
            tokens, tags, tags_prob = None, None, None
            if doing_train_data:
                tokens, tags, tags_prob = sent
            else:
                tokens, tags = sent
            text = ' '.join(tokens)
            
            tokens_batch.append(tokens)
            text_batch.append(text)
            tags_batch.append(tags)
            if doing_train_data:
                tags_prob_batch.append(tags_prob)
       
        encoded_batch = tokenizer.batch_encode_plus(text_batch, add_special_tokens=False)
        for i in range(len(batch)):
            
            ids = encoded_batch['input_ids'][i]
            tokenized_tokens = tokenizer.convert_ids_to_tokens(ids)

            bert_tokens = []
            cur_token = tokenized_tokens[0] 
            if len(tokenized_tokens)>1:
                for token in tokenized_tokens[1:]:  
                    if token.startswith('##'):
                        cur_token += token[2:]
                    else:
                        bert_tokens.append(cur_token)
                        cur_token=token
            bert_tokens.append(cur_token)
            
            tokens = tokens_batch[i]
            tags = tags_batch[i]
            if doing_train_data:
                tags_prob = tags_prob_batch[i]
            
            bertIdx2idx = create_mapping(tokens, bert_tokens)
            bert_tags = []
            bert_tags_prob = []
            for idx in range(len(bert_tokens)):
                
                if tags[bertIdx2idx[idx]]=='B' and bert_tags[-1]=='B':
                    bert_tags.append('I')
                else:
                    bert_tags.append(tags[bertIdx2idx[idx]])
                    
                if doing_train_data:
                    bert_tags_prob.append(tags_prob[bertIdx2idx[idx]])
            
            if len(cached_bert_tokens) + len(bert_tokens) < max_sent_len:
                cached_bert_tokens.extend(bert_tokens)
                cached_bert_tags.extend(bert_tags)
                cached_bert_tags_prob.extend(bert_tags_prob)
            else:
                if doing_train_data:
                    for w, t, t_prob in zip(cached_bert_tokens, cached_bert_tags, cached_bert_tags_prob):
                        fw.writelines(w + '\t' + t + '\t' + ' '.join([str(x) for x in t_prob]) + '\n')
                else:
                    for w, t in zip(cached_bert_tokens, cached_bert_tags):
                        fw.writelines(w + '\t' + t + '\n')
                fw.writelines('\n')
        
                
                cached_bert_tokens = copy.deepcopy(bert_tokens)
                cached_bert_tags = copy.deepcopy(bert_tags)
                if doing_train_data:
                    cached_bert_tags_prob = copy.deepcopy(bert_tags_prob)
      
    if doing_train_data:
        for w, t, t_prob in zip(cached_bert_tokens, cached_bert_tags, cached_bert_tags_prob):
            fw.writelines(w + '\t' + t + '\t' + ' '.join([str(x) for x in t_prob]) + '\n')
    else:
        for w, t in zip(cached_bert_tokens, cached_bert_tags):
            fw.writelines(w + '\t' + t + '\n')