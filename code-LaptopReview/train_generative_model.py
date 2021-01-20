from wiser.data.dataset_readers import LaptopsDatasetReader
from wiser.rules import TaggingRule, LinkingRule, DictionaryMatcher
from wiser.generative import get_label_to_ix, get_rules
from labelmodels import *
from wiser.generative import train_generative_model
from labelmodels import LearningConfig
from wiser.generative import evaluate_generative_model
from wiser.data import save_label_distribution
from wiser.rules import ElmoLinkingRule
from wiser.eval import *
from collections import Counter
import random
import pickle

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str)

parser.add_argument('--group', default=1, type=int)
parser.add_argument('--use_SurfaceForm', action='store_true')
parser.add_argument('--use_Suffix', action='store_true')
parser.add_argument('--use_Prefix', action='store_true')
parser.add_argument('--use_InclusivePreNgram', action='store_true')
parser.add_argument('--use_ExclusivePreNgram', action='store_true')
parser.add_argument('--use_InclusivePostNgram', action='store_true')
parser.add_argument('--use_ExclusivePostNgram', action='store_true')
parser.add_argument('--use_Dependency', action='store_true')

parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--init_acc', default=0.9, type=float)
parser.add_argument('--acc_prior', default=1, type=int)
parser.add_argument('--balance_prior', default=10, type=int)

parser.add_argument('--model_name', default=None, type=str)
args = parser.parse_args()






with open('../datasets/{}/train.pickle'.format(args.dataset), 'rb') as f:
    train_data = pickle.load(f)
with open('../datasets/{}/test.pickle'.format(args.dataset), 'rb') as f:
    test_data = pickle.load(f)
with open('../datasets/{}/dev.pickle'.format(args.dataset), 'rb') as f:
    dev_data = pickle.load(f)

laptops_docs = train_data + test_data + dev_data


base_folder = '../candidates/{}'.format(args.dataset)


dict_core = set()
with open('../datasets/AutoNER_dicts/{}/dict_core.txt'.format(args.dataset), 'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        term = tuple(line[1:])
        dict_core.add(term)


dict_full = set()

with open('../datasets/AutoNER_dicts/{}/dict_full.txt'.format(args.dataset), 'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        if len(line) > 1:
            dict_full.add(tuple(line))



############### SurfaceForm
if args.use_SurfaceForm:
    for ix in range(1, 6):
        with open('{}/SurfaceForm_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            propogated = pickle.load(f)[:20]
            exceptions = set(['function','level', 'curve'])
            for item in propogated:
                if not item in exceptions:
                    dict_core.add(tuple(item.split()))
                    #dict_core_exact.add(tuple(item.split()))
    print('propogated surface forms applied: ', len(propogated))
    print('expanded dict core: ', len(dict_core))
########################################################

############## Suffix
propogated_suffix = set()
for ix in range(1, 6):
    with open('{}/Suffix_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
        for item in pickle.load(f)[:15]:
            propogated_suffix.add(item)
manual = {'pad', 'oto', 'fox', 'chpad', 'rams'}
for item in manual:
    propogated_suffix.add(item)


propogated_suffix = tuple(propogated_suffix)
class CommonSuffixes(TaggingRule):
    def __init__(self, match_lemma=True):
        self.suffixes = propogated_suffix
        self.match_lemma = match_lemma
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            w = instance['tokens'][i].text
            if self.match_lemma:
                w = instance['tokens'][i].lemma_
            for suffix in self.suffixes:
                if w.endswith(suffix) and len(w)>len(suffix):
                    labels[i] = 'I'
        return labels
    
if args.use_Suffix:
    lf = CommonSuffixes(match_lemma=True)
    lf.apply(laptops_docs)
    print('propogated suffixes applied')

################ Prefix
propogated_prefix = set()
for ix in range(1, 6):
    with open('{}/Prefix_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
        for item in pickle.load(f)[:1]:
            propogated_prefix.add(item)
manual = ['feat', 'softw', 'batt', 'Win', 'osx']
for item in manual:
    propogated_prefix.add(item)
    
propogated_prefix = tuple(propogated_prefix)
class CommonPrefixes(TaggingRule):
    def __init__(self, match_lemma=True):
        self.prefixes = propogated_prefix
        self.match_lemma = True
        self.exceptions = set(['computer', 'scratch'])
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        for i, t in enumerate(instance['tokens']):
            w = t.text
            if w.lower() in self.exceptions:
                continue
            if self.match_lemma:
                w = t.lemma_
                if w.lower() in self.exceptions:
                    continue
            for prefix in self.prefixes:
                if len(w)>3 and len(w)>len(prefix) and w.startswith(prefix) and t.pos_=='NOUN' and not t.lemma_ in self.exceptions:
                    labels[i] = 'I'
        return labels

if args.use_Prefix:
    lf = CommonPrefixes(match_lemma=True)
    lf.apply(laptops_docs)
    print('propogated prefix applied.')


###############  PreNgram Inclusive #########################
propogated_inclusive_prengram = set()
for ix in range(1, 6):
    with open('{}/InclusivePreNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
        for item in pickle.load(f)[:1]:
            propogated_inclusive_prengram.add(item)
            
manual = ['windows', 'hard', 'extended', 'touch', 'boot']
for item in manual:
    propogated_inclusive_prengram.add(item)

propogated_inclusive_prengram = tuple(propogated_inclusive_prengram)
class CustomizedInclusivePreNgram(TaggingRule):
    def __init__(self, seed_list, label_type = 'I', length_list=None):
        self.label_type=label_type
        self.seeds = seed_list
        self.length_list = length_list
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        
        for seed in self.seeds:
            seed_len = len(seed.split())
            for i in range(len(instance['tokens'])-seed_len-1):
                cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                if cand == seed:
                    right = i+seed_len
                    
                    if instance['tokens'][right].pos_=='NOUN':
                        for j in range(i, right+1):
                            labels[j] = self.label_type
        return labels

if args.use_InclusivePreNgram:
    lf = CustomizedInclusivePreNgram(propogated_inclusive_prengram)
    lf.apply(laptops_docs)
#######################################################

############## PreNgram Exclusive ###################
propogated_exclusive_prengram = set()
for ix in range(1, 6):
    with open('{}/ExclusivePreNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
        for item in tuple(pickle.load(f)[:5]):
            propogated_exclusive_prengram.add(item)
    
propogated_exclusive_prengram = tuple(propogated_exclusive_prengram)
class CustomizedExclusivePreNgram(TaggingRule):
    def __init__(self, seed_list, label_type = 'I', length_list=None):
        self.label_type=label_type
        self.seeds = seed_list
        self.length_list = length_list

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        tokens = instance['tokens']
        for seed in self.seeds:
            seed_len = len(seed.split())
            for i in range(len(instance['tokens'])-seed_len-1):
                cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                if cand == seed:
                    left = i+seed_len
                    if instance['tokens'][left].pos_ in ['NOUN', 'PROPN', 'ADJ']:
                        right = left+1
                        while right<len(tokens) and instance['tokens'][right].pos_ in ['NOUN', 'PROPN', 'ADJ']:
                            right+=1
                        for j in range(left, right):
                            labels[j] = self.label_type
        return labels

if args.use_ExclusivePreNgram:
    lf = CustomizedExclusivePreNgram(propogated_exclusive_prengram)
    lf.apply(laptops_docs)


############### PostNgram Inclusive
propogated_inclusive_postngram = set()
for ix in range(1, 6):
    with open('{}/InclusivePostNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
        for item in tuple(pickle.load(f)[:1]):
            propogated_inclusive_postngram.add(item)
            
manual = ['x', 'xp', 'vista', 'drive', 'processing']
for item in manual:
    propogated_inclusive_postngram.add(item)            

propogated_inclusive_postngram = tuple(propogated_inclusive_postngram)

class CustomizedInclusivePostNgram(TaggingRule):
    def __init__(self, seed_list, label_type = 'I', length_list=None):
        self.label_type=label_type
        self.seeds = seed_list
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        for seed in self.seeds:
            seed_len = len(seed.split())
            for i in range(1,len(instance['tokens'])-seed_len):
                cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                if cand == seed:
                    if instance['tokens'][i-1].pos_ in ['NOUN', 'ADJ', 'PROPN', 'NUM']:
                        left = i-2
                        for j in range(left+1, i+seed_len):
                            labels[j] = self.label_type
        return labels
if args.use_InclusivePostNgram:
    lf = CustomizedInclusivePostNgram(propogated_inclusive_postngram)
    lf.apply(laptops_docs)
######################################################


################## Dependency
propogated_dependency = set()
# for ix in range(1, 6):
#     with open('{}/Dependency_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
#         for item in tuple(pickle.load(f)):
#             propogated_dependency.add(item)
manual = ['StartDep:compound|HeadSurf:port', 'StartDep:compound|HeadSurf:button',
    'StartDep:nummod|HeadSurf:ram', 'StartDep:amod|HeadSurf:drive']
for item in manual:
    propogated_dependency.add(item)
    
    
propogated_dependency = tuple(propogated_dependency)

class CustomizedDependencyMatcher(TaggingRule):
    def __init__(self, seed_list, label_type='I', exceptions=None):
        self.label_type=label_type
        self.seeds = seed_list
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        tokens = instance['tokens']
        deps = [tk.dep_ for tk in tokens]
        for i in range(len(tokens)):
            if tokens[i].pos_ in ['NOUN', 'PROPN'] and (i==0 or tokens[i-1].pos_ not in ['NOUN','PROPN']):
                left = i
                right = i+1
                while right<len(tokens) and tokens[right].pos_ in ['NOUN', 'PROPN']:
                    right+=1
                if right > i+1:
                    cand = 'StartDep:{}|HeadSurf:{}'.format(deps[i], tokens[right-1].text.lower())
                    cand2 = 'StartDep:{}|HeadSurf:{}'.format(deps[i], tokens[right-1].lemma_.lower()) 
                    if cand in self.seeds or cand2 in self.seeds:   
                        for j in range(i, right):
                            labels[j] = self.label_type
                    
                    cand = 'EndDep:{}|HeadSurf:{}'.format(deps[right-2], tokens[right-1].text.lower())
                    cand2 = 'EndDep:{}|HeadSurf:{}'.format(deps[right-2], tokens[right-1].lemma_.lower())
                    if cand in self.seeds or cand2 in self.seeds:  
                        for j in range(i, right):
                            labels[j] = self.label_type                  
        return labels
    
if args.use_Dependency:
    lf = CustomizedDependencyMatcher(propogated_dependency)
    lf.apply(laptops_docs)
    
######################################################

lf = DictionaryMatcher("CoreDictionary", dict_core, uncased=True, i_label="I")
lf.apply(laptops_docs)

other_terms = [['BIOS'], ['color'], ['cord'], ['hinge'], ['hinges'],
               ['port'], ['speaker']]
lf = DictionaryMatcher("OtherTerms", other_terms, uncased=True, i_label="I")
lf.apply(laptops_docs)


class ReplaceThe(TaggingRule):
    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens) - 2):
            if tokens[i].lower() == 'replace' and tokens[i +
                                                         1].lower() == 'the':
                if instance['tokens'][i + 2].pos_ == "NOUN":
                    labels[i] = 'O'
                    labels[i + 1] = 'O'
                    labels[i + 2] = 'I'

        return labels


lf = ReplaceThe()
lf.apply(laptops_docs)


class iStuff(TaggingRule):
    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens)):
            if len(
                    tokens[i]) > 1 and tokens[i][0] == 'i' and tokens[i][1].isupper():
                labels[i] = 'I'

        return labels


lf = iStuff()
lf.apply(laptops_docs)


class Feelings(TaggingRule):
    feeling_words = {"like", "liked", "love", "dislike", "hate"}

    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens) - 2):
            if tokens[i].lower() in self.feeling_words and tokens[i +
                                                                  1].lower() == 'the':
                if instance['tokens'][i + 2].pos_ == "NOUN":
                    labels[i] = 'O'
                    labels[i + 1] = 'O'
                    labels[i + 2] = 'I'

        return labels


lf = Feelings()
lf.apply(laptops_docs)


class ProblemWithThe(TaggingRule):
    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens) - 3):
            if tokens[i].lower() == 'problem' and tokens[i + \
                               1].lower() == 'with' and tokens[i + 2].lower() == 'the':
                if instance['tokens'][i + 3].pos_ == "NOUN":
                    labels[i] = 'O'
                    labels[i + 1] = 'O'
                    labels[i + 2] = 'O'
                    labels[i + 3] = 'I'

        return labels


lf = ProblemWithThe()
lf.apply(laptops_docs)


class External(TaggingRule):
    def apply_instance(self, instance):
        tokens = [token.text for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        for i in range(len(tokens) - 1):
            if tokens[i].lower() == 'external':
                labels[i] = 'I'
                labels[i + 1] = 'I'

        return labels


lf = External()
lf.apply(laptops_docs)


stop_words = {"a", "and", "as", "be", "but", "do", "even",
              "for", "from",
              "had", "has", "have", "i", "in", "is", "its", "just",
              "my", "no", "not", "of", "on", "or",
              "that", "the", "these", "this", "those", "to", "very",
              "what", "which", "who", "with"}


class StopWords(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in stop_words:
                labels[i] = 'O'
        return labels


lf = StopWords()
lf.apply(laptops_docs)


class Punctuation(TaggingRule):
    pos = {"PUNCT"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
            if pos in self.pos:
                labels[i] = 'O'

        return labels


lf = Punctuation()
lf.apply(laptops_docs)


class Pronouns(TaggingRule):
    pos = {"PRON"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
            if pos in self.pos:
                labels[i] = 'O'

        return labels


lf = Pronouns()
lf.apply(laptops_docs)


class NotFeatures(TaggingRule):
    keywords = {"laptop", "computer", "pc"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in self.keywords:
                labels[i] = 'O'
        return labels


lf = NotFeatures()
lf.apply(laptops_docs)


class Adv(TaggingRule):
    pos = {"ADV"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
            if pos in self.pos:
                labels[i] = 'O'

        return labels


lf = Adv()
lf.apply(laptops_docs)


class CompoundPhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i - 1].dep_ == "compound":
                links[i] = 1

        return links


lf = CompoundPhrase()
lf.apply(laptops_docs)


lf = ElmoLinkingRule(.8)
lf.apply(laptops_docs)


class ExtractedPhrase(LinkingRule):
    def __init__(self, terms):
        self.term_dict = {}

        for term in terms:
            term = [token.lower() for token in term]
            if term[0] not in self.term_dict:
                self.term_dict[term[0]] = []
            self.term_dict[term[0]].append(term)

        # Sorts the terms in decreasing order so that we match the longest
        # first
        for first_token in self.term_dict.keys():
            to_sort = self.term_dict[first_token]
            self.term_dict[first_token] = sorted(
                to_sort, reverse=True, key=lambda x: len(x))

    def apply_instance(self, instance):
        tokens = [token.text.lower() for token in instance['tokens']]
        links = [0] * len(instance['tokens'])

        i = 0
        while i < len(tokens):
            if tokens[i] in self.term_dict:
                candidates = self.term_dict[tokens[i]]
                for c in candidates:
                    # Checks whether normalized AllenNLP tokens equal the list
                    # of string tokens defining the term in the dictionary
                    if i + len(c) <= len(tokens):
                        equal = True
                        for j in range(len(c)):
                            if tokens[i + j] != c[j]:
                                equal = False
                                break

                        # If tokens match, labels the instance tokens
                        if equal:
                            for j in range(i + 1, i + len(c)):
                                links[j] = 1
                            i = i + len(c) - 1
                            break
            i += 1

        return links


lf = ExtractedPhrase(dict_full)
lf.apply(laptops_docs)


class ConsecutiveCapitals(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        # We skip the first pair since the first
        # token is almost always capitalized
        for i in range(2, len(instance['tokens'])):
            # We skip this token if it all capitals
            all_caps = True
            text = instance['tokens'][i].text
            for char in text:
                if char.islower():
                    all_caps = False
                    break

            if not all_caps and text[0].isupper(
            ) and instance['tokens'][i - 1].text[0].isupper():
                links[i] = 1

        return links


lf = ConsecutiveCapitals()
lf.apply(laptops_docs)


print(score_labels_majority_vote(test_data, span_level=True))
print('--------------------')

save_label_distribution('../output-gen/{}/dev_data.p'.format(args.dataset), dev_data)
save_label_distribution('../output-gen/{}/test_data.p'.format(args.dataset), test_data)


cnt = Counter()
for instance in train_data + dev_data:
    for tag in instance['tags']:
        cnt[tag] += 1

disc_label_to_ix = {value[0]: ix for ix, value in enumerate(cnt.most_common())}

gen_label_to_ix = {'ABS': 0, 'I': 1, 'O': 2}

batch_size = 64

""" Linked HMM Model """
# Defines the model
tagging_rules, linking_rules = get_rules(train_data)
link_hmm = LinkedHMM(
    num_classes=len(gen_label_to_ix) - 1,
    num_labeling_funcs=len(tagging_rules),
    num_linking_funcs=len(linking_rules),
    init_acc=args.init_acc,
    acc_prior=args.acc_prior,
    balance_prior=args.balance_prior)

# Trains the model
p, r, f1 = train_generative_model(
    link_hmm, train_data, dev_data, label_to_ix=gen_label_to_ix, config=LearningConfig(epochs=args.epochs, batch_size=batch_size))

# Evaluates the model
print('Linked HMM: \n' + str(evaluate_generative_model(model=link_hmm,
                                                       data=test_data, label_to_ix=gen_label_to_ix)))


# Saves the model
inputs = get_generative_model_inputs(train_data, gen_label_to_ix)
p_unary, p_pairwise = link_hmm.get_label_distribution(*inputs)
save_label_distribution(
   '../output-gen/{}/train_data_link_hmm.p'.format(args.dataset),
    train_data,
    p_unary,
    p_pairwise,
    gen_label_to_ix,
    disc_label_to_ix)