import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from tqdm import tqdm

from wiser.data.dataset_readers import *
from wiser.rules import TaggingRule, LinkingRule, UMLSMatcher, DictionaryMatcher
from wiser.generative import get_label_to_ix, get_rules
from labelmodels import *
from wiser.generative import train_generative_model
from labelmodels import LearningConfig
from wiser.generative import evaluate_generative_model
from wiser.data import save_label_distribution
from wiser.eval import *
from wiser.rules import ElmoLinkingRule
from collections import Counter


import string
import pickle
import random
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--rule_type', type=str)
args = parser.parse_args()

dataset = args.dataset
rule_type = args.rule_type

assert rule_type in ['SurfaceForm', 'Suffix', 'Prefix', 'InclusivePostNgram', 'InclusivePreNgram', 'ExclusivePreNgram', 'Dependency']


file_path_to_train_data = '../datasets/{}/train.pickle'.format(dataset)
file_path_to_dev_data = '../datasets/{}/dev.pickle'.format(dataset)
file_path_to_test_data = '../datasets/{}/test.pickle'.format(dataset)
file_path_to_dict_core = '../datasets/AutoNER_dicts/{}/dict_core.txt'.format(dataset)

output_dir = '../cached_seeds_and_embeddings/{}'.format(dataset)

    
with open(file_path_to_train_data, 'rb') as f:
    train_data = pickle.load(f)
with open(file_path_to_dev_data, 'rb') as f:
    test_data = pickle.load(f)
with open(file_path_to_test_data, 'rb') as f:
    dev_data = pickle.load(f)
    
print('Train: ', len(train_data))
print('Dev: ', len(dev_data))
print('Test: ', len(test_data))


from utils_prepare_data import *
from customized_models import *
from pre_defined_variables import *


seeds = get_seed_list(dataset, rule_type)

pos_cnt = Counter()
for sent in dev_data:
    pos_cnt += collect_POS(sent, label='I')
pos_set = set(pos_cnt.keys())
freq_pos_set = sorted(list(pos_set), key=lambda x: pos_cnt[x], reverse=True)[:15]
print("top frequent POS patterns: {}".format(freq_pos_set))


exception_list = get_exception_list(dataset, rule_type)

candidates_final = None
if rule_type=='SurfaceForm':
    surface_dict = collect_SurfaceForm_candidates(train_data, freq_pos_set)
    candidates = set([k for k in surface_dict.keys()])
    for seed in seeds:
        candidates.add(seed.lower())
    print(exception_list)
    for item in exception_list:
        if item in candidates:
            candidates.remove(item)
    candidates_final = set()
    for item in candidates:
        candidates_final.add(tuple(item.split()))

elif rule_type=='Suffix':
    suffix_dict = collect_suffix_candidates(train_data, [4,5,6], exception_list)
    candidates_final = set(suffix_dict.keys())
    for w in seeds:
        candidates_final.add(w)

elif rule_type=='Prefix':
    prefix_dict = collect_prefix_candidates(train_data, [4,5,6], exception_list)
    candidates_final = set(prefix_dict.keys())
    for w in seeds:
        candidates_final.add(w)
    
elif rule_type=='InclusivePostNgram':
    surface_dict = collect_SurfaceForm_candidates(train_data, freq_pos_set)
    surface_candidates = set([k for k in surface_dict.keys()])
    for item in exception_list:
        if item in surface_candidates:
            surface_candidates.remove(item)
            
    surface_seeds = get_seed_list(dataset, "SurfaceForm")
    for item in exception_list:
        if item in surface_candidates:
            surface_candidates.remove(item)
    for seed in surface_seeds:
        surface_candidates.add(seed)
    
    postngram_dict = collect_inclusive_postNgram_candidates(surface_candidates, ngram_list=[1,2,3])

    postngram_seeds = set(seeds)
    candidates_final = set([])
    for item in postngram_seeds:
        candidates_final.add(item)
    for item in postngram_dict:
        candidates_final.add(item)
elif rule_type=='InclusivePreNgram':
    surface_dict = collect_SurfaceForm_candidates(train_data, freq_pos_set)
    surface_candidates = set([k for k in surface_dict.keys()])
    surface_seeds = get_seed_list(dataset, "SurfaceForm")
    for item in exception_list:
        if item in surface_candidates:
            surface_candidates.remove(item)
    for seed in surface_seeds:
        surface_candidates.add(seed)

    prengram_dict = collect_inclusive_preNgram_candidates(surface_candidates, ngram_list=[1,2,3])
    prengram_seeds = set(seeds)
    candidates_final = set([])
    for item in prengram_seeds:
        candidates_final.add(item)
    for item in prengram_dict:
        candidates_final.add(item)
elif rule_type=='ExclusivePreNgram':
    surface_dict = collect_SurfaceForm_candidates(train_data, freq_pos_set)
    surface_candidates = set([k for k in surface_dict.keys()])
    surface_seeds = get_seed_list(dataset, "SurfaceForm")
    for item in exception_list:
        if item in surface_candidates:
            surface_candidates.remove(item)
    for seed in surface_seeds:
        surface_candidates.add(seed)
    
    prengram_dict = collect_exclusive_preNgram_candidates(train_data,ngram_list=[1,2,3],exceptions=exception_list)
    
    prengram_seeds = set(seeds)
    candidates_final = set([])
    for item in prengram_seeds:
        candidates_final.add(item)
    for item in prengram_dict:
        candidates_final.add(item)
elif rule_type=='Dependency':
    surface_dict = collect_SurfaceForm_candidates(train_data, freq_pos_set)
    surface_candidates = set([k for k in surface_dict.keys()])
    surface_seeds = get_seed_list(dataset, "SurfaceForm")
    
    for item in exception_list:
        if item in surface_candidates:
            surface_candidates.remove(item)
    
    candidates_final = surface_candidates.union(set([tuple(k.split()) for k in surface_dict.keys()]))
############ Extract candidate embeddings
lf = None
if rule_type=='SurfaceForm':
    lf = CustomizedDictionaryMatcher("CoreDictionaryUncased",list(candidates_final),uncased=True,i_label="I", match_lemmas=True)
elif rule_type=='Suffix':
    lf = CustomizedCommonSuffixes(candidates_final, threshold=7)
elif rule_type=='Prefix':
    lf = CustomizedCommonPrefixes(candidates_final,  threshold=5)
elif rule_type=='InclusivePostNgram':
    lf = CustomizedInclusivePostNgram(tuple(candidates_final), length_list=[1,2])
elif rule_type=='InclusivePreNgram':
    lf = CustomizedInclusivePreNgram(tuple(candidates_final), length_list=[1,2,3])
elif rule_type=='ExclusivePreNgram':
    lf = CustomizedExclusivePreNgram(tuple(candidates_final), length_list=[1,2])
elif rule_type=='Dependency':
    lf = CustomizedDependency("Dependency",list(candidates_final),uncased=True,i_label="I", match_lemmas=True)

cand2emb = extract_candidate_embeddings(train_data, lf)

################# Write postive seeds
pos_seeds = seeds
with open('{}/{}_Pos_Seeds.pk'.format(output_dir, rule_type), 'wb') as fw:
    pickle.dump(pos_seeds, fw, protocol=pickle.HIGHEST_PROTOCOL)

################ Write negative seeds
neg_seeds = get_negative_seed_list(dataset, rule_type) 
if rule_type=='SurfaceForm':
    for w in cand2emb:
        if w.split()[0] in ["a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", "just", "my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", "which", "who", "with", ]:
            neg_seeds.add(w)
        
with open('{}/{}_Neg_Seeds.pk'.format(output_dir, rule_type), 'wb') as fw:
    pickle.dump(neg_seeds, fw, protocol=pickle.HIGHEST_PROTOCOL)
    
# ################# Write node2idx
node2idx = {}
for node in cand2emb:
    node2idx[node] = len(node2idx)
idx2node = {v:k for k,v in node2idx.items()}
with open('{}/{}_node2idx.pk'.format(output_dir, rule_type), 'wb') as fw:
    pickle.dump(node2idx, fw, protocol=pickle.HIGHEST_PROTOCOL)


################## Write node embedding matrix
nodes = None
for idx in tqdm(range(len(node2idx))):
    c = idx2node[idx]
    if nodes is None:
        nodes = cand2emb[c].unsqueeze(0)
    else:
        nodes = torch.cat((nodes, cand2emb[c].unsqueeze(0)), dim=0)
print(nodes.shape)
with open('{}/{}_node_embeddings.pk'.format(output_dir, rule_type), 'wb') as fw:
    pickle.dump(nodes, fw, protocol=pickle.HIGHEST_PROTOCOL)


############## Write edges based on top similar neighbors
top = 10
sim_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
source_nodes, target_nodes = [], []
for i in tqdm(range(nodes.shape[0]-1)):
    e1 = nodes[i,:].unsqueeze(0)
    s = sim_function(e1.expand(nodes.shape[0], -1), nodes)
    top_neighbors = sorted(enumerate(s.tolist()), key=lambda x: x[1], reverse=True)[:top+1]
    for nei in top_neighbors:
        if not i==nei[0]:
            source_nodes.extend([i, nei[0]])
            target_nodes.extend([nei[0], i])    

fname = '{}/{}_edges.pk'.format(output_dir, rule_type)
with open(fname, 'wb') as fw:
    edge_index = torch.LongTensor([source_nodes, target_nodes])
    pickle.dump(edge_index, fw, protocol=pickle.HIGHEST_PROTOCOL)