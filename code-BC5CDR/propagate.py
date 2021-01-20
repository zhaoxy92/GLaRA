import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from gnn import Graph

import json
import string
import pickle
import random
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--rule_type', type=str)
parser.add_argument('--label_type', type=str)
args = parser.parse_args()



with open('config-{}.json'.format(args.label_type), 'r') as f:
    config = json.loads(f.read())

dataset = args.dataset
rule_type = args.rule_type
label_type = args.label_type

output_directory = '../candidates/{}'.format(args.dataset)


if not rule_type in config:
    print("No {} pattern propagation, please checn configuration file for available patterns. ".format(rule_type))


train_file_path = '../datasets/{}/train.pickle'.format(dataset)
dev_file_path = '../datasets/{}/dev.pickle'.format(dataset)



nodes2idx_path, pos_seeds_path, neg_seeds_path, nodes_embedding_path, edge_path=None, None, None, None, None

# if not pattern_type.startswith('SurfaceForm'):
#     tmp_pattern_type = '_'.join(pattern_type.split('_')[:-1])
#     nodes2idx_path = 'cached_embeddings/{}_node2idx.pk'.format(tmp_pattern_type)
#     pos_seeds_path = 'cached_embeddings/{}_Pos_Seeds.pk'.format(tmp_pattern_type)
#     neg_seeds_path = 'cached_embeddings/{}_Neg_Seeds.pk'.format(tmp_pattern_type)
#     nodes_embedding_path = 'cached_embeddings/{}_node_embeddings.pk'.format(tmp_pattern_type)
#     edge_path = 'cached_embeddings/{}_edges.pk'.format(tmp_pattern_type)
    
# else:
nodes2idx_path = '../cached_seeds_and_embeddings/{}/{}_{}_node2idx.pk'.format(dataset, label_type, rule_type)
pos_seeds_path = '../cached_seeds_and_embeddings/{}/{}_{}_Pos_Seeds.pk'.format(dataset, label_type, rule_type)
neg_seeds_path = '../cached_seeds_and_embeddings/{}/{}_{}_Neg_Seeds.pk'.format(dataset, label_type, rule_type)
nodes_embedding_path = '../cached_seeds_and_embeddings/{}/{}_{}_node_embeddings.pk'.format(dataset, label_type, rule_type)
edge_path = '../cached_seeds_and_embeddings/{}/{}_{}_edges.pk'.format(dataset, label_type, rule_type)

epochs = config[rule_type]['epochs']
num_of_pattern_to_save = config[rule_type]['num_of_pattern_to_save']
num_round_to_integrate = config[rule_type]['num_round_to_integrate']
group_total = 5

output_file_prefix = "{}_{}".format(label_type, rule_type)


"""
NOTE: to increase model stability and performance, for each group, we will actually train the model 5 (num_round_to_integrate) times to integrate propagtion results. Then we will run 5 groups to calculate model performance mean and standard deviation.

NOTE: the output files will be save to the format: 
<output_directory>/<output_file_prefix>_g<group>_r<1..5>.txt.
e.g. candidates/SurfaceForm_g1_r1.txt. 
"""

for group in range(1, group_total+1):

    with open(train_file_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(dev_file_path, 'rb') as f:
        dev_data = pickle.load(f)

    print('Train: ', len(train_data))
    print('Dev: ', len(dev_data))


    ## Load cached nodes and edges data
    with open(nodes2idx_path, 'rb') as f:
        node2idx = pickle.load(f)
        idx2node = {v:k for k,v in node2idx.items()}
    print("node2idx: ", len(node2idx))

    with open(pos_seeds_path, 'rb') as f:
        pos_seeds = list(pickle.load(f))
    print("pos_seeds: ", len(pos_seeds))

    with open(neg_seeds_path, 'rb') as f:
        neg_seeds = list(pickle.load(f)) 
    print("neg_seeds: ", len(neg_seeds))


    with open(nodes_embedding_path, 'rb') as f:
        node2emb = pickle.load(f)
    print("node2emb: ", len(node2emb))

    with open(edge_path, 'rb') as f:
        edge_index = pickle.load(f)
    print("edge_index: ", edge_index.shape)

    # create label
    y = []
    for i in range(len(node2idx)):
        node = idx2node[i]
        if node in pos_seeds:
            y.append(1)
        elif node in neg_seeds:
            y.append(0)
        else:
            y.append(-1)

    random.shuffle(pos_seeds)
    random.shuffle(neg_seeds)

    pos_train_percent = config[rule_type]['pos_train_percent']  #0.8 (surface)
    neg_train_percent = config[rule_type]['neg_train_percent']  #0.8 (surface)
    train_pos_seeds = pos_seeds[:int(len(pos_seeds)*pos_train_percent)]
    test_pos_seeds = pos_seeds[int(len(pos_seeds)*pos_train_percent):]

    train_neg_seeds = neg_seeds[:int(len(neg_seeds)*neg_train_percent)]
    test_neg_seeds = neg_seeds[int(len(neg_seeds)*neg_train_percent):]

    print("pos train num: ", len(train_pos_seeds), "neg train num:", len(train_neg_seeds))
    print("pos test num: ", len(test_pos_seeds), "neg test num:", len(test_neg_seeds))
    train_list = train_pos_seeds + train_neg_seeds
    test_list = test_pos_seeds + test_neg_seeds
    print(len(train_list))
    print(len(test_list))

    train_idx_list = [node2idx[w] for w in train_list if w in node2idx]
    test_idx_list = [node2idx[w] for w in test_list if w in node2idx]

    ## build graph
    y_longtensor = torch.LongTensor(y)
    y_tensor = torch.Tensor(y)
    train_mask = torch.BoolTensor([False]*len(y))
    train_mask[train_idx_list] = True
    test_mask = torch.BoolTensor([False]*len(y))
    test_mask[test_idx_list] = True

    train_pos_mask = torch.BoolTensor([False]*len(y))
    train_pos_mask[[node2idx[w] for w in train_pos_seeds if w in node2idx]] = True
    train_neg_mask = torch.BoolTensor([False]*len(y))
    train_neg_mask[[node2idx[w] for w in train_neg_seeds if w in node2idx]] = True

    graph_data = Data(
        x=node2emb, edge_index=edge_index, y_longtensor=y_longtensor, y_tensor=y_tensor,
        train_mask=train_mask, test_mask=test_mask,
        train_pos_mask=train_pos_mask, train_neg_mask=train_neg_mask
    )
    graph_data


    for r in range(1, num_round_to_integrate+1):
        torch.cuda.empty_cache()

        model = Graph(node_feature_dim=2048, output_dim=1).to(device)

        graph_data = graph_data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

        sim_function = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logit, out = model(graph_data)
            loss = F.binary_cross_entropy(F.sigmoid(logit[graph_data.train_mask]), graph_data.y_tensor[graph_data.train_mask].unsqueeze(1))
            loss2 = F.mse_loss(F.sigmoid(out[graph_data.edge_index[0]]), F.sigmoid(out[graph_data.edge_index[1]]))
            loss3 = sim_function(out[graph_data.train_pos_mask].mean(dim=0), out[graph_data.train_neg_mask].mean(dim=0))
            total_loss = loss + loss2 + loss3
            print(total_loss.item())
            total_loss.backward()
            optimizer.step()

        sim_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model.eval()
        logit, out = model(graph_data)
        pos_centroid = out[graph_data.train_pos_mask].mean(dim=0)
        neg_centroid = out[graph_data.train_neg_mask].mean(dim=0)
        print(sim_function(pos_centroid.unsqueeze(0), neg_centroid.unsqueeze(0)))

        dis2pos = sim_function(out, pos_centroid.unsqueeze(0).expand(out.shape[0], -1))
        dis2neg = sim_function(out, neg_centroid.unsqueeze(0).expand(out.shape[0], -1))

        pred = torch.zeros(out.shape[0]).to(device)
        pred[dis2pos>dis2neg] = 1

        correct = float (pred[graph_data.test_mask].eq(graph_data.y_longtensor[graph_data.test_mask]).sum().item())
        acc = correct / graph_data.test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))

        dist_diff = dis2pos-dis2neg
        propogated = [w for w, _ in sorted([(idx2node[ix], diff) for ix, diff in enumerate(dist_diff.tolist())], key=lambda item:item[1], reverse=True)[:num_of_pattern_to_save]]

        candidates = set([tuple(item.split()) for item in propogated])
        print(len(candidates))

        with open('{}/{}_g{}_r{}.txt'.format(output_directory, output_file_prefix, group, r), 'wb') as fw:
            pickle.dump([item for item in propogated], fw, protocol=pickle.HIGHEST_PROTOCOL)