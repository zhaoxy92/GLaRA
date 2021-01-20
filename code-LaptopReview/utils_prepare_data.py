from collections import Counter
from pre_defined_variables import get_exception_list


def get_sentence_embedding(batch):
    sentences = []
    for sent in batch:
        sentences.append([tk.text for tk in sent['tokens']])
    character_ids = batch_to_ids(sentences).to(device)
    with torch.no_grad():
        embeddings = elmo(character_ids)
    return embeddings['elmo_representations'][0]

def process_candidate_embedding(emb, allennlp_concat):
    if allennlp_concat:
        return emb
    return torch.mean(emb, dim=0)

def compute_cosine_sim(soruce_cand2emb, target_cand2emb, allennlp_concat, sim_func):
    sim = {}
    for k1 in soruce_cand2emb:
        sim[k1] = []
        e1 = process_candidate_embedding(soruce_cand2emb[k1], allennlp_concat)
        for k2 in target_cand2emb:
            e2 = process_candidate_embedding(target_cand2emb[k2], allennlp_concat)
            sim[k1].append((k2, sim_func(e1, e2).item()))
        sim[k1] = sorted(sim[k1], key=lambda x: x[1], reverse=True)
    return sim


def collect_POS(sentence, label):
    pos = Counter()    
    tokens, tags = sentence['tokens'], sentence['tags']
    left, right = 0, 0
    while left<len(tokens):
        while left<len(tokens) and tags[left]!=label:
            left+=1
        if left<len(tokens) and tags[left]==label:
            right = left+1
            while right<len(tokens) and tags[right]==label:
                right+=1
            cand = ' '.join([tk.pos_ for tk in tokens[left:right]])
            pos[cand]+=1  
            left = right+1
    return pos

def find_NP_forward(sentence, start):
    tokens, tags = sentence['tokens'], sentence['tags']
    left = start
    right = left+1
    while right<len(tokens) and tokens[right].pos_=='ADJ':
        right+=1
    while right<len(tokens) and tokens[right].pos_=='NOUN':
        right+=1
    return ' '.join([tk.lemma_ for tk in tokens[left:right]])

def find_NP_backward(sentence, start):
    tokens, tags = sentence['tokens'], sentence['tags']
    right = start
    left = right-1
    while left>=0 and tokens[left].pos_=='NOUN':
        left-=1
    while left>=0 and tokens[left].pos_=='ADJ':
        left-=1
    return ' '.join([tk.lemma_ for tk in tokens[left+1:right+1]])

def match_NP_pattern(sentence, start, pattern):
    tokens, tags = sentence['tokens'], sentence['tags']
    pos_list = pattern.split()
    if start>len(tokens)-len(pos_list):
        return False    
    if pattern==' '.join([tk.pos_ for tk in tokens[start:start+len(pos_list)]]):
        return True
    return False

def collect_SurfaceForm_candidates(data, freq_pos_set):
    phrase_cnt = Counter()
    for sent in data:
        tokens = sent['tokens']
        for i in range(len(tokens)):
            if tokens[i].pos_=='NOUN' and (i==len(tokens)-1 or not tokens[i+1].pos_=='NOUN'):
                left = i-1
                while left>0 and tokens[left].pos_ in ['NOUN']:
                    left-=1
                cand = ' '.join([tk.lemma_ for tk in tokens[left+1:i+1]])
                phrase_cnt[cand]+=1

                while left>=0 and tokens[left].pos_ in ['ADJ',]:
                    cand = tokens[left].lemma_ + ' ' + cand
                    phrase_cnt[cand]+=1
                    left-=1
                    
        # collect based on top POS pattern in dev set.
        for i in range(len(tokens)):
            for pattern in freq_pos_set:
                if match_NP_pattern(sent, i, pattern):
                    cand = ' '.join([tk.lemma_ for tk in tokens[i:i+len(pattern.split())]])
                    phrase_cnt[cand]+=1
        
        for i in range(len(tokens)-1):
            if tokens[i].pos_=='PROPN':
                cand = tokens[i].lemma_.lower()
                phrase_cnt[cand]+=1

            if tokens[i].pos_=='PROPN' and tokens[i+1].pos_=='NOUN':
                cand = ' '.join([tk.lemma_ for tk in tokens[i:i+2]]).lower()
                phrase_cnt[cand]+=1
            if tokens[i].pos_=='PROPN' and tokens[i+1].pos_=='PROPN':
                cand = ' '.join([tk.lemma_ for tk in tokens[i:i+2]]).lower()
                phrase_cnt[cand]+=1

            if tokens[i].pos_=='VERB':
                cand = tokens[i].lemma_.lower()
                phrase_cnt[cand]+=1
                
    return phrase_cnt


def collect_suffix_candidates(data, len_list):
    suffix_dict = Counter() 
    for sent in data:
        for tk in sent['tokens']:
            lemma = tk.lemma_
            if len(lemma)>3 and tk.pos_=='NOUN' and \
                    not lemma in [
                        'very', 'every', 'answer', 'between', 'music', 'mac', 'pc','computer','laptop','phone',
                        'iphone','imac','apple'
                    ] and not tk.text.lower().endswith(('tion', 'ing')) and not lemma.endswith(('tion','ing')) and\
                    not lemma.endswith(('ed', 'er','ist','est')) and not ('.' in lemma or '-' in lemma):
                for suffix_len in len_list:
                    suffix_dict[lemma[-suffix_len:]] +=1
    return suffix_dict 

def collect_prefix_candidates(data, len_list):
    prefix_dict = Counter() 
    for sent in data:
        for tk in sent['tokens']:
            lemma = tk.lemma_
            if len(lemma)>3 and tk.pos_=='NOUN' and \
                    not lemma.lower() in [ 'very', 'every', 'answer', 'between', 'music', 'mac', 'pc','computer',
                    'laptop','phone','iphone','imac','apple', 'device', 'equipment']:
                for prefix_len in len_list:
                    prefix_dict[lemma[:prefix_len]] +=1
    return prefix_dict 


def collect_inclusive_postNgram_candidates(candidate_set, ngram_list):
    ngram_dict = Counter()
    for ngram in ngram_list:
        for item in candidate_set:
            if len(item.split())>ngram and not item.startswith(('for','to','a','the','cut','quality')):
                ngram_dict[' '.join(item.split()[-ngram:])] +=1
    return ngram_dict


def collect_inclusive_preNgram_candidates(candidate_set, ngram_list, exceptions=[]):
    ngram_dict = Counter()
    for ngram in ngram_list:
        for item in candidate_set:
            if len(item.split())>ngram:
                ngram_dict[' '.join(item.split()[:ngram])] +=1
    return ngram_dict



def collect_exclusive_preNgram_candidates(data, ngram_list=[1,2,3]):
    prengram_dict = Counter()
    for sent in data:
        tokens = sent['tokens']
        i = 1
        while i<len(tokens):
            if tokens[i].pos_ in [ 'NOUN'] and tokens[i-1].pos_ not in ['NOUN'] and \
                    not tokens[i].lemma_ in ['pc', 'computer', 'laptop', 'version', 'music', 'mac', 'phone']:
                right= i
                for ngram in ngram_list:
                    left = max(0, i-ngram)
                    cand = ' '.join([tk.lemma_ for tk in tokens[left:right]]).lower()
                    prengram_dict[cand]+=1
            i+=1
    return prengram_dict



from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
from torch.autograd import Variable
from allennlp.modules.span_extractors import EndpointSpanExtractor

    

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 1, dropout=0).to(device)


def extract_candidate_embeddings(data, lf, batch_size=32):
    extractor = EndpointSpanExtractor(input_dim=1024, combination="x,y")
    candidate_cnt = Counter()
    cand2emb = {}
    data_batches = [data[x:x + batch_size] for x in range(0, len(data), batch_size)]
    for batch_ix, batch in enumerate(data_batches):
        print("Current batch: {}, total batch: {}".format(batch_ix, len(data_batches)))
        batch_embeddings = get_sentence_embedding(batch)
        for sid, sent in enumerate(batch):
            tokens = sent['tokens']
            spans_dict = {}
            _, lf_spans = lf.apply_instance(sent)
            for k in lf_spans:                
                if k in spans_dict:
                    spans_dict[k].extend(lf_spans[k])
                else:
                    spans_dict[k] = lf_spans[k] 
                spans_dict[k] = list(set(spans_dict[k]))
            spans = [sp for k in spans_dict for sp in spans_dict[k]]
            spans = list(set(spans))
            
            if spans_dict:
                inclusive_spans_dict = {}
                for k in spans_dict:
                    inclusive_spans_dict[k] = []
                    for x, y in spans_dict[k]:
                        inclusive_spans_dict[k].append((x,y-1))
                
                inclusive_spans = [sp for k in inclusive_spans_dict for sp in inclusive_spans_dict[k]]
                inclusive_spans_withKEY = [(sp, k) for k in inclusive_spans_dict for sp in inclusive_spans_dict[k]]

                indices = Variable(torch.LongTensor(inclusive_spans).unsqueeze(0)).to(device)    
                span_representations = extractor(batch_embeddings[sid].unsqueeze(0), indices)
                                
                for spid, item in enumerate(inclusive_spans_withKEY):
                    k = item[1]
                    candidate_cnt[k]+=1 
                    if k in cand2emb:
                        cand2emb[k] += span_representations[0, spid, :]
                    else:
                        cand2emb[k] = span_representations[0, spid, :]
                        
    torch.cuda.empty_cache() 
    for cand in cand2emb:
        cand2emb[cand] = cand2emb[cand]/candidate_cnt[cand]    
        cand2emb[cand] = cand2emb[cand].to('cpu')    
    print('number of candidates: ', len(cand2emb))
    return cand2emb