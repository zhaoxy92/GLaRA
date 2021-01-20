from wiser.data.dataset_readers import NCBIDiseaseDatasetReader
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
import random
import pickle

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
parser.add_argument('--init_acc', default=0.85, type=float)
parser.add_argument('--acc_prior', default=55, type=int)
parser.add_argument('--balance_prior', default=450, type=int)

parser.add_argument('--model_name', default=None, type=str)

args = parser.parse_args()

print(args)

reader = NCBIDiseaseDatasetReader()

with open('../datasets/{}/train.pickle'.format(args.dataset), 'rb') as f:
    train_data = pickle.load(f)
with open('../datasets/{}/test.pickle'.format(args.dataset), 'rb') as f:
    test_data = pickle.load(f)
with open('../datasets/{}/dev.pickle'.format(args.dataset), 'rb') as f:
    dev_data = pickle.load(f)

ncbi_docs = train_data + dev_data + test_data

dict_core = set()
dict_core_exact = set()
with open('../datasets/AutoNER_dicts/{}/dict_core.txt'.format(args.dataset), 'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        term = tuple(line[1:])

        if len(term) > 1 or len(term[0]) > 3:
            dict_core.add(term)
        else:
            dict_core_exact.add(term)
            
dict_full = set()
with open('../datasets/AutoNER_dicts/{}/dict_full.txt'.format(args.dataset), 'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        dict_full.add(tuple(line))      
        
base_folder = '../candidates/{}'.format(args.dataset)
############### SurfaceForm
if args.use_SurfaceForm:
    for ix in range(1, 6):
        with open('{}/SurfaceForm_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            propogated = pickle.load(f)
            exceptions = set()
            for item in propogated:
                if not item in exceptions:
                    dict_core.add(tuple(item.split()))
                    #dict_core_exact.add(tuple(item.split()))
    print('expanded dict core: ', len(dict_core))
##############################################
##########

# Prepends common modifiers
to_add = set()
for term in dict_core:
    to_add.add(("inherited", ) + term)
    to_add.add(("Inherited", ) + term)
    to_add.add(("hereditary", ) + term)
    to_add.add(("Hereditary", ) + term)
dict_core |= to_add

if "WT1" in dict_core_exact:
    dict_core_exact.remove(("WT1",))
if "VHL" in dict_core_exact:
    dict_core_exact.remove(("VHL",))


lf = DictionaryMatcher(
    "CoreDictionaryUncased",
    dict_core,
    uncased=True,
    i_label="I")
lf.apply(ncbi_docs)


lf = DictionaryMatcher("CoreDictionaryExact", dict_core_exact, i_label="I")
lf.apply(ncbi_docs)


class CancerLike(TaggingRule):
    def apply_instance(self, instance):
        tokens = [token.text.lower() for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        suffixes = ("edema", "toma", "coma", "noma")

        for i, token in enumerate(tokens):
            for suffix in suffixes:
                if token.endswith(suffix) or token.endswith(suffix + "s"):
                    labels[i] = 'I'
        return labels


lf = CancerLike()
lf.apply(ncbi_docs)


################################ suffix 
propogated_suffix = set()
if args.use_Suffix:
    for ix in range(1, 6):
        with open('{}/Suffix_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in pickle.load(f):
                propogated_suffix.add(item)
manual = {'skott', 'drich', 'umour', 'axia', 'iridia'}
for item in manual:
    propogated_suffix.add(item)

propogated_suffix = tuple(propogated_suffix)

class CommonSuffixes(TaggingRule):

    suffixes = {
        "agia",
        "cardia",
        "trophy",
        "toxic",
        "itis",
        "emia",
        "pathy",
        "plasia"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            for suffix in self.suffixes:
                if instance['tokens'][i].lemma_.endswith(suffix):
                    labels[i] = 'I'
            
            for suffix in propogated_suffix:
                w = instance['tokens'][i].lemma_.lower()
                if len(w)>5 and len(w)>len(suffix) and w.endswith(suffix) and instance['tokens'][i].pos_=='NOUN':
                    labels[i] = 'I'
                    break

        return labels

lf = CommonSuffixes()
lf.apply(ncbi_docs)

############################### Prefix
propogated_prefix = set()
if args.use_Prefix:
    for ix in range(1, 6):
        with open('{}/Prefix_g{}_r{}.txt'.format(base_folder,args.group, ix), 'rb') as f:
            for item in pickle.load(f):
                propogated_prefix.add(item)

manual_prefix = ('carc', 'myot', 'tela', 'ovari', 'atax', 'carcin', 'dystro')
for item in manual_prefix:
    propogated_prefix.add(item)

propogated_prefix = tuple(propogated_prefix)

class CustomizedCommonPrefixes(TaggingRule):
    def __init__(self, prefixes_list):
        self.prefixes = prefixes_list
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        spans = {}
        for i, t in enumerate(instance['tokens']):
            w = t.lemma_
            for prefix in self.prefixes:
                if len(w)>5 and len(w) > len(prefix) and \
                        w.startswith(prefix) and t.pos_=='NOUN':
                    labels[i] = 'I'
        return labels

lf = CustomizedCommonPrefixes(propogated_prefix)
lf.apply(ncbi_docs)

############################### PreNgram Inclusive
propogated_inclusive_prengram = set()
if args.use_InclusivePreNgram:
    for ix in range(1, 6):
        with open('{}/InclusivePreNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in pickle.load(f):
                propogated_inclusive_prengram.add(item)

manual_inclusive_prengram = ('breast and ovarian', 'x - link', 'breast and', 'stage iii', 'myotonic','hereditary')
for item in manual_inclusive_prengram:
    propogated_inclusive_prengram.add(item)

propogated_inclusive_prengram = tuple(propogated_inclusive_prengram)

class CustomizedInclusivePreNgram(TaggingRule):
    def __init__(self, seed_list, length_list=None):
        self.seeds = seed_list
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
                            labels[j] = 'I'
        return labels

lf = CustomizedInclusivePreNgram(propogated_inclusive_prengram)
lf.apply(ncbi_docs)

############################### PreNgram Exlusive
propogated_exclusive_prengram = set()
if args.use_ExclusivePreNgram:
    for ix in range(1, 6):
        with open('{}/ExclusivePreNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in pickle.load(f)[:15]:
                propogated_exclusive_prengram.add(item)
manual_exclusive_prengram = ('suffer from', 'fraction of', 'pathogenesis of', 'cause severe')
for item in manual_exclusive_prengram:
    propogated_exclusive_prengram.add(item)

propogated_exclusive_prengram = tuple(propogated_exclusive_prengram)

class CustomizedExclusivePreNgram(TaggingRule):
    def __init__(self, seed_list):
        self.seeds = seed_list
        
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
                            labels[j] = 'I'
        return labels

lf = CustomizedExclusivePreNgram(propogated_exclusive_prengram)
lf.apply(ncbi_docs)

################################ PostNgram Inclusive
propogated_inclusive_postngram = set()
if args.use_InclusivePostNgram:
    for ix in range(1, 6):
        with open('{}/InclusivePostNgram_g{}_r{}.txt'.format(base_folder,args.group, ix), 'rb') as f:
            for item in pickle.load(f):
                propogated_inclusive_postngram.add(item)

manual_inclusive_postngram = ('- t', 'cell carcinoma', 'muscular dystrophy', "'s disease", 'carcinoma', 'dystrophy')
for item in manual_inclusive_postngram:
    propogated_inclusive_postngram.add(item)

propogated_inclusive_postngram = tuple(propogated_inclusive_postngram)

class CustomizedInclusivePostNgram(TaggingRule):
    def __init__(self, seed_list):
        self.seeds = seed_list
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        for seed in self.seeds:
            seed_len = len(seed.split())
            for i in range(1,len(instance['tokens'])-seed_len):
                cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                if cand == seed:
                    if instance['tokens'][i-1].pos_ in ['NOUN', 'ADJ', 'PROPN']:
                        left = i-2
                        for j in range(left+1, i+seed_len):
                            labels[j] = 'I'
        return labels

if use_InclusivePostNgram:
    lf = CustomizedInclusivePostNgram(propogated_inclusive_postngram)
    lf.apply(ncbi_docs)

################################ Dependency
propogated_dependency = set()
if args.use_Dependency:
    for ix in range(1, 6):
        with open('{}/Dependency_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in pickle.load(f):
                propogated_dependency.add(item)
manual_dependency =(
    'StartDep:compound|HeadSurf:disease','StartDep:amod|HeadSurf:dystrophy','StartDep:punct|HeadSurf:telangiectasia'
        ,'StartDep:compound|HeadSurf:t','StartDep:amod|HeadSurf:dysplasia'
)
for item in manual_dependency:
    propogated_dependency.add(item)

propogated_dependency= tuple(propogated_dependency)

class CustomizedDependencyMatcher(TaggingRule):
    def __init__(self, seed_list, exceptions=None):
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
                            labels[j] = 'I'
                    
                    cand = 'EndDep:{}|HeadSurf:{}'.format(deps[right-2], tokens[right-1].text.lower())
                    cand2 = 'EndDep:{}|HeadSurf:{}'.format(deps[right-2], tokens[right-1].lemma_.lower())
                    if cand in self.seeds or cand2 in self.seeds:  

                        for j in range(i, right):
                            labels[j] = 'I'               
        return labels

if args.use_Dependency:
    lf = CustomizedDependencyMatcher(propogated_dependency)
    lf.apply(ncbi_docs)
#####################################################################


class Deficiency(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        # "___ deficiency"
        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'deficiency':
                labels[i] = 'I'
                labels[i + 1] = 'I'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I'
                    else:
                        break

        # "deficiency of ___"
        for i in range(len(instance['tokens']) - 2):
            if instance['tokens'][i].lemma_ == 'deficiency' and instance['tokens'][i + 1].lemma_ == 'of':
                labels[i] = 'I'
                labels[i + 1] = 'I'
                nnp_active = False
                for j in range(i + 2, len(instance['tokens'])):
                    if instance['tokens'][j].pos_ in ('NOUN', 'PROPN'):
                        if not nnp_active:
                            nnp_active = True
                    elif nnp_active:
                        break
                    labels[j] = 'I'

        return labels


lf = Deficiency()
lf.apply(ncbi_docs)


class Disorder(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'disorder':
                labels[i] = 'I'
                labels[i + 1] = 'I'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I'
                    else:
                        break

        return labels


lf = Disorder()
lf.apply(ncbi_docs)


class Lesion(TaggingRule):
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'lesion':
                labels[i] = 'I'
                labels[i + 1] = 'I'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I'
                    else:
                        break
        return labels


lf = Lesion()
lf.apply(ncbi_docs)


class Syndrome(TaggingRule):
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'syndrome':
                labels[i] = 'I'
                labels[i + 1] = 'I'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I'
                    else:
                        break
        return labels

lf = Syndrome()
lf.apply(ncbi_docs)

terms = []
with open('../datasets/umls/umls_body_part.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))
lf = DictionaryMatcher("TEMP", terms, i_label='TEMP', uncased=True, match_lemmas=True)
lf.apply(ncbi_docs)

class BodyTerms(TaggingRule):
    def apply_instance(self, instance):
        tokens = [token.text.lower() for token in instance['tokens']]
        labels = ['ABS'] * len(tokens)

        terms = set([
            "cancer", "cancers",
            "damage",
            "disease", "diseases"
                       "pain",
            "injury", "injuries",
        ])

        for i in range(0, len(tokens) - 1):
            if instance['WISER_LABELS']['TEMP'][i] == 'TEMP':
                if tokens[i + 1] in terms:
                    labels[i] = "I"
                    labels[i + 1] = "I"
        return labels


lf = BodyTerms()
lf.apply(ncbi_docs)

for doc in ncbi_docs:
    del doc['WISER_LABELS']['TEMP']


class OtherPOS(TaggingRule):
    other_pos = {"ADP", "ADV", "DET", "VERB"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(0, len(instance['tokens'])):
            if instance['tokens'][i].pos_ in self.other_pos:
                labels[i] = "O"
        return labels


lf = OtherPOS()
lf.apply(ncbi_docs)


stop_words = {"a", "as", "be", "but", "do", "even",
              "for", "from",
              "had", "has", "have", "i", "in", "is", "its", "just",
              "my", "no", "not", "on", "or",
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
lf.apply(ncbi_docs)


class Punctuation(TaggingRule):

    other_punc = {".", ",", "?", "!", ";", ":", "(", ")",
                  "%", "<", ">", "=", "+", "/", "\\"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].text in self.other_punc:
                labels[i] = 'O'
        return labels


lf = Punctuation()
lf.apply(ncbi_docs)


class PossessivePhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i -
                                  1].text == "'s" or instance['tokens'][i].text == "'s":
                links[i] = 1

        return links


lf = PossessivePhrase()
lf.apply(ncbi_docs)


class HyphenatedPhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i -
                                  1].text == "-" or instance['tokens'][i].text == "-":
                links[i] = 1

        return links


lf = HyphenatedPhrase()
lf.apply(ncbi_docs)


lf = ElmoLinkingRule(.8)
lf.apply(ncbi_docs)


class CommonBigram(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        tokens = [token.text.lower() for token in instance['tokens']]

        bigrams = {}
        for i in range(1, len(tokens)):
            bigram = tokens[i - 1], tokens[i]
            if bigram in bigrams:
                bigrams[bigram] += 1
            else:
                bigrams[bigram] = 1

        for i in range(1, len(tokens)):
            bigram = tokens[i - 1], tokens[i]
            count = bigrams[bigram]
            if count >= 6:
                links[i] = 1

        return links


lf = CommonBigram()
lf.apply(ncbi_docs)


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
lf.apply(ncbi_docs)


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