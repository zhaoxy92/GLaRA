from wiser.data.dataset_readers.cdr import CDRCombinedDatasetReader
from wiser.rules import TaggingRule, LinkingRule, DictionaryMatcher
from wiser.generative import get_label_to_ix, get_rules
from labelmodels import *
from wiser.generative import train_generative_model
from labelmodels import LearningConfig
from wiser.generative import evaluate_generative_model
from wiser.data import save_label_distribution
from wiser.eval import *
from collections import Counter

import random
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--group', default=1, type=int)

parser.add_argument('--use_dis_SurfaceForm', action='store_true')
parser.add_argument('--use_dis_Suffix', action='store_true')
parser.add_argument('--use_dis_Prefix', action='store_true')
parser.add_argument('--use_dis_InclusivePreNgram', action='store_true')
parser.add_argument('--use_dis_ExclusivePreNgram', action='store_true')
parser.add_argument('--use_dis_InclusivePostNgram', action='store_true')
parser.add_argument('--use_dis_ExclusivePostNgram', action='store_true')
parser.add_argument('--use_dis_Dependency', action='store_true')

parser.add_argument('--use_chem_SurfaceForm', action='store_true')
parser.add_argument('--use_chem_Suffix', action='store_true')
parser.add_argument('--use_chem_Prefix', action='store_true')
parser.add_argument('--use_chem_InclusivePreNgram', action='store_true')
parser.add_argument('--use_chem_ExclusivePreNgram', action='store_true')
parser.add_argument('--use_chem_InclusivePostNgram', action='store_true')
parser.add_argument('--use_chem_ExclusivePostNgram', action='store_true')
parser.add_argument('--use_chem_Dependency', action='store_true')

parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--init_acc', default=0.85, type=float)
parser.add_argument('--acc_prior', default=5, type=int)
parser.add_argument('--balance_prior', default=450, type=int)

parser.add_argument('--model_name', default=None, type=str)
args = parser.parse_args()

cdr_reader = CDRCombinedDatasetReader()

with open('../datasets/{}/train.pickle'.format(args.dataset), 'rb') as f:
    train_data = pickle.load(f)
with open('../datasets/{}/test.pickle'.format(args.dataset), 'rb') as f:
    test_data = pickle.load(f)
with open('../datasets/{}/dev.pickle'.format(args.dataset), 'rb') as f:
    dev_data = pickle.load(f)
    
cdr_docs = train_data + dev_data + test_data

base_folder = '../candidates/{}'.format(args.dataset)


dict_core_chem = set()
dict_core_chem_exact = set()
dict_core_dis = set()
dict_core_dis_exact = set()

with open('../datasets/AutoNER_dicts/{}/dict_core.txt'.format(args.dataset)) as f:
    for line in f.readlines():
        line = line.strip().split(None, 1)
        entity_type = line[0]
        tokens = cdr_reader.get_tokenizer()(line[1])
        term = tuple([str(x) for x in tokens])

        if len(term) > 1 or len(term[0]) > 3:
            if entity_type == 'Chemical':
                dict_core_chem.add(term)
            elif entity_type == 'Disease':
                dict_core_dis.add(term)
            else:
                raise Exception()
        else:
            if entity_type == 'Chemical':
                dict_core_chem_exact.add(term)
            elif entity_type == 'Disease':
                dict_core_dis_exact.add(term)
            else:
                raise Exception()

print('original dict_dis core: ', len(dict_core_dis))
print('original dict_dis exact: ', len(dict_core_dis_exact))
print('original dict_chem core: ', len(dict_core_chem))
print('original dict_chem exact: ', len(dict_core_chem_exact))

############### SurfaceForm 

if args.use_dis_SurfaceForm:
    propogated_dis = set()
    for ix in range(1,6):
        with open('{}/Disease_SurfaceForm_g{}_r{}.txt'.format(base_folder, group, ix), 'rb') as f:
            propogated = pickle.load(f)
            for item in propogated:
                propogated_dis.add(tuple(item.split()))
                #dict_core_dis.add(tuple(item.split()))
    print('propogated Dis surface forms applied:', len(propogated_dis))
    lf = DictionaryMatcher("propogated-Disease-surface", propogated_dis, i_label="I-Disease", uncased=False)
    lf.apply(cdr_docs)

if args.use_chem_SurfaceForm:
    propogated_chem = set()
    for ix in range(1, 6):
        with open('{}/Chemical_SurfaceForm_g{}_r{}.txt'.format(base_folder, group, ix), 'rb') as f:
            propogated = pickle.load(f)
            for item in propogated:
                #dict_core_chem.add(tuple(item.split()))
                propogated_chem.add(tuple(item.split()))
    print('propogated Chem surface forms applied: ', len(propogated_chem))
    lf = DictionaryMatcher("propogated-Chemical-surface",propogated_chem, i_label="I-Chemical",uncased=True)
    lf.apply(cdr_docs)
########################################################



lf = DictionaryMatcher(
    "DictCore-Chemical",
    dict_core_chem,
    i_label="I-Chemical",
    uncased=True)
lf.apply(cdr_docs)
lf = DictionaryMatcher(
    "DictCore-Chemical-Exact",
    dict_core_chem_exact,
    i_label="I-Chemical",
    uncased=False, match_lemmas=True)
lf.apply(cdr_docs)
lf = DictionaryMatcher(
    "DictCore-Disease",
    dict_core_dis,
    i_label="I-Disease",
    uncased=True)
lf.apply(cdr_docs)
lf = DictionaryMatcher(
    "DictCore-Disease-Exact",
    dict_core_dis_exact,
    i_label="I-Disease",
    uncased=False, match_lemmas=True)
lf.apply(cdr_docs)



terms = []
with open('../datasets/umls/umls_element_ion_or_isotope.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))

lf = DictionaryMatcher(
    "Element, Ion, or Isotope",
    terms,
    i_label='I-Chemical',
    uncased=True,
    match_lemmas=True)

lf.apply(cdr_docs)

terms = []
with open('../datasets/umls/umls_organic_chemical.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))

lf = DictionaryMatcher(
    "Organic Chemical",
    terms,
    i_label='I-Chemical',
    uncased=True,
    match_lemmas=True)
lf.apply(cdr_docs)

terms = []
with open('../datasets/umls/umls_antibiotic.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))
lf = DictionaryMatcher(
    "Antibiotic",
    terms,
    i_label='I-Chemical',
    uncased=True,
    match_lemmas=True)
lf.apply(cdr_docs)

terms = []
with open('../datasets/umls/umls_disease_or_syndrome.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))


lf = DictionaryMatcher(
    "Disease or Syndrome",
    terms,
    i_label='I-Disease',
    uncased=True,
    match_lemmas=True)
lf.apply(cdr_docs)

terms = []
with open('../datasets/umls/umls_body_part.txt', 'r') as f:
    for line in f.readlines():
        terms.append(line.strip().split(" "))


lf = DictionaryMatcher(
    "TEMP",
    terms,
    i_label='TEMP',
    uncased=True,
    match_lemmas=True)
lf.apply(cdr_docs)

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
                    labels[i] = "I-Disease"
                    labels[i + 1] = "I-Disease"
        return labels


lf = BodyTerms()
lf.apply(cdr_docs)

for doc in cdr_docs:
    del doc['WISER_LABELS']['TEMP']


class Acronyms(TaggingRule):
    other_lfs = {
        'I-Chemical': ("Antibiotic", "Element, Ion, or Isotope", "Organic Chemical"),
        'I-Disease': ("BodyTerms", "Disease or Syndrome")
    }

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        active = False
        for tag, lf_names in self.other_lfs.items():
            acronyms = set()
            for lf_name in lf_names:
                for i in range(len(instance['tokens']) - 2):
                    if instance['WISER_LABELS'][lf_name][i] == tag:
                        active = True
                    elif active and instance['tokens'][i].text == '(' and instance['tokens'][i + 2].pos_ == "PUNCT" and instance['tokens'][i + 1].pos_ != "NUM":
                        acronyms.add(instance['tokens'][i + 1].text)
                        active = False
                    else:
                        active = False

            for i, token in enumerate(instance['tokens']):
                if token.text in acronyms:
                    labels[i] = tag

        return labels


lf = Acronyms()
lf.apply(cdr_docs)

class Damage(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'damage':
                labels[i] = 'I-Disease'
                labels[i + 1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Damage()
lf.apply(cdr_docs)

class Disease(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'disease':
                labels[i] = 'I-Disease'
                labels[i + 1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Disease()
lf.apply(cdr_docs)

class Disorder(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'disorder':
                labels[i] = 'I-Disease'
                labels[i + 1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Disorder()
lf.apply(cdr_docs)

class Lesion(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'lesion':
                labels[i] = 'I-Disease'
                labels[i + 1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Lesion()
lf.apply(cdr_docs)

class Syndrome(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                               1].lemma_ == 'syndrome':
                labels[i] = 'I-Disease'
                labels[i + 1] = 'I-Disease'

                # Adds any other compound tokens before the phrase
                for j in range(i - 1, -1, -1):
                    if instance['tokens'][j].dep_ == 'compound':
                        labels[j] = 'I-Disease'
                    else:
                        break

        return labels


lf = Syndrome()
lf.apply(cdr_docs)


###################### Disease Suffix 
exceptions = {'diagnosis', 'apoptosis', 'prognosis', 'metabolism', 'homeostasis', 'emphasis'}

propogated_dis_suffix = set()
if args.use_dis_Suffix:
    for ix in range(1, 6):
        with open('{}/Disease_Suffix_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in pickle.load(f)[:15]:
                propogated_dis_suffix.add(item)
manual = {'epsy', 'nson'}
for item in manual:
    propogated_dis_suffix.add(item)
propogated_dis_suffix = tuple(propogated_dis_suffix)
    
suffixes =  ("agia", "cardia", "trophy", "itis","emia", "enia", "pathy", "plasia", "lism", "osis")

class DiseaseSuffixes(TaggingRule):
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, t in enumerate(instance['tokens']):
            if len(t.lemma_) >= 5 and t.lemma_.lower(
            ) not in exceptions and t.lemma_.lower().endswith(suffixes):
                labels[i] = 'I-Disease'
            
            if len(t.lemma_)>=7 and t.lemma_.lower().endswith(propogated_dis_suffix) and not t.lemma_ in exceptions and t.pos_=='NOUN':
                labels[i] = 'I-Disease'

        return labels

lf = DiseaseSuffixes()
lf.apply(cdr_docs)

###################### Chemical Suffix 

exceptions = {'determine', 'baseline', 'decline',
              'examine', 'pontine', 'vaccine',
              'routine', 'crystalline', 'migraine',
              'alkaline', 'midline', 'borderline',
              'cocaine', 'medicine', 'medline',
              'asystole', 'control', 'protocol',
              'alcohol', 'aerosol', 'peptide',
              'provide', 'outside', 'intestine',
              'combine', 'delirium', 'VIP',
             'aprotinin', 'candidate', 'elucidate',
             }
propogated_chem_suffix = set()
if args.use_chem_Suffix:
    for ix in range(1, 6):
        with open('{}/Chemical_Suffix_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in pickle.load(f)[:15]:
                propogated_chem_suffix.add(item)
manual = {'pine','icin','dine','ridol', 'athy', 'zure', 'mide', 'fen', 'phine'}
for item in manual:
    propogated_chem_suffix.add(item)

propogated_chem_suffix = tuple(propogated_chem_suffix)

suffixes = ('ine', 'ole', 'ol', 'ide', 'ine', 'ium', 'epam')

class ChemicalSuffixes(TaggingRule):
    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        acronyms = set()
        for i, t in enumerate(instance['tokens']):
            if len(t.lemma_) >= 7 and t.lemma_ not in exceptions and t.lemma_.endswith(
                    suffixes):
                labels[i] = 'I-Chemical'

                if i < len(instance['tokens']) - 3 and instance['tokens'][i + \
                           1].text == '(' and instance['tokens'][i + 3].text == ')':
                    acronyms.add(instance['tokens'][i + 2].text)

            if args.use_chem_Suffix and len(t.lemma_)>=7 and t.lemma_.lower().endswith(propogated_chem_suffix) and t.pos_=='NOUN' and not t.lemma_.lower() in exceptions:
                labels[i] = 'I-Chemical'

        for i, t in enumerate(instance['tokens']):
            if t.text in acronyms and t.text not in exceptions:
                labels[i] = 'I-Chemical'

        return labels

lf = ChemicalSuffixes()
lf.apply(cdr_docs)
#############################################


#################### Disease Prefix
exceptions = {'hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric', 'stenosis', 'toxicology','delivery','psychiatry'}
prefixes = ('hyper', 'hypo')
propogated_dis_prefix = set()

if args.use_dis_Prefix:
    for ix in range(1, 6):
        with open('{}/Disease_Prefix_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in pickle.load(f)[:1]:
                propogated_dis_prefix.add(item)
manual = ['anemi','dyski', 'heada', 'hypok', 'hypert','ische', 'arthr', 'hypox', 'toxic', 'arrhyt', 'ischem', 'hypert', 'dysfunc']
for item in manual:
    propogated_dis_prefix.add(item)
propogated_dis_prefix = tuple(propogated_dis_prefix)


class DiseasePrefixes(TaggingRule):
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i, t in enumerate(instance['tokens']):
            if len(t.lemma_) >= 5 and t.lemma_.lower(
            ) not in exceptions and t.lemma_.startswith(prefixes):
                if instance['tokens'][i].pos_ == "NOUN":
                    labels[i] = 'I-Disease'
            w = t.lemma_.lower()
            for prefix in propogated_dis_prefix:
                if len(w)>7 and len(w)>len(prefix) and w.startswith(prefix) and t.pos_=='NOUN' and not w in exceptions:
                    labels[i] = 'I-Disease'
        return labels

lf = DiseasePrefixes()
lf.apply(cdr_docs)

#################### Chemical Prefix

propogated_chem_prefix = set()

if args.use_chem_Prefix:
    for ix in range(1, 6):
        with open('{}/Chemical_Prefix_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in pickle.load(f)[:1]:
                propogated_chem_prefix.add(item)
manual = ['chlor','levo','doxor','lithi','morphi','hepari','ketam','potas']
for item in manual:
    propogated_chem_prefix.add(item)
propogated_chem_prefix = tuple(propogated_chem_prefix)

class CustomizedCommonPrefixes(TaggingRule):
    def __init__(self, prefixes_list, label_type='I', threshold=7, match_lemma=True):
        self.label_type=label_type
        self.prefixes = prefixes_list
        self.threshold = threshold
        self.match_lemma=match_lemma
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        for i, t in enumerate(instance['tokens']):
            w = t.text
            if self.match_lemma:
                w = t.lemma_
            for prefix in self.prefixes:
                if len(w)>self.threshold and len(w) > len(prefix) and \
                        w.startswith(prefix) and t.pos_=='NOUN' :
                    labels[i] = self.label_type
        return labels

lf = CustomizedCommonPrefixes(list(propogated_chem_prefix), threshold=5,label_type='I-Chemical', match_lemma=True)
lf.apply(cdr_docs)
##############################################################


#################### Inclusive PostNgram

class CustomizedInclusivePostNgram(TaggingRule):
    def __init__(self, seed_list, label_type = 'I'):
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
    
    
propogated_dis_inclusive_postngram = set()
if args.use_dis_InclusivePostNgram:
    for ix in range(1, 6):
        with open('{}/Disease_InclusivePostNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in tuple(pickle.load(f)[:1]):
                propogated_dis_inclusive_postngram.add(item)
            
manual = ["'s disease", 'infarction', "'s sarcoma" ,'epilepticus' , 'artery disease', 'de pointe',
                   'insufficiency', 'with aura', 'artery spasm', "'s encephalopathy"]
for item in manual:
    propogated_dis_inclusive_postngram.add(item)            

propogated_dis_inclusive_postngram = tuple(propogated_dis_inclusive_postngram)

    
lf = CustomizedInclusivePostNgram(propogated_dis_inclusive_postngram, label_type='I-Disease')
lf.apply(cdr_docs)


propogated_chem_inclusive_postngram = set()
if args.use_chem_InclusivePostNgram:
    for ix in range(1, 6):
        with open('{}/Chemical_InclusivePostNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in tuple(pickle.load(f)[:1]):
                propogated_chem_inclusive_postngram.add(item)
            
manual = ['aminocaproic acid', '- aminocaproic acid', 'retinoic acid','dopa', 'tc', '- aminopyridine', 'aminopyridine','- penicillamine', '- dopa', '- aspartate','fu', 'hydrochloride']
for item in manual:
    propogated_chem_inclusive_postngram.add(item)            

propogated_chem_inclusive_postngram = tuple(propogated_chem_inclusive_postngram)

    
lf = CustomizedInclusivePostNgram(propogated_chem_inclusive_postngram, label_type='I-Chemical')
lf.apply(cdr_docs)
#################################################################

######################### Exclusive PreNgram
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
                    if instance['tokens'][left].pos_ in ['NOUN']:
                        right = left+1
                        while right<len(tokens) and instance['tokens'][right].pos_ in ['NOUN']:
                            right+=1
                        for j in range(left, right):
                            labels[j] = self.label_type
        return labels
    
propogated_dis_exclusive_prengram = set()
if args.use_dis_ExclusivePreNgram:
    for ix in range(1, 6):
        with open('{}/Disease_ExclusivePreNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in tuple(pickle.load(f)[:5]):
                propogated_dis_exclusive_prengram.add(item)
manual = ['to induce', 'w -', 'and severe', 'suspicion of', 'die of', 'have severe', 'of persistent', 'cyclophosphamide associate']
for item in manual:
    propogated_dis_exclusive_prengram.add(item)           
propogated_dis_exclusive_prengram = tuple(propogated_dis_exclusive_prengram)

lf = CustomizedExclusivePreNgram(list(propogated_dis_exclusive_prengram), label_type='I-Disease')
lf.apply(cdr_docs)


propogated_chem_exclusive_prengram = set()
if args.use_chem_ExclusivePreNgram:
    for ix in range(1, 6):
        with open('{}/Chemical_ExclusivePreNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in tuple(pickle.load(f)[:5]):
                propogated_chem_exclusive_prengram.add(item)
manual = ['dosage of', 'sedation with', 'mg of', 'application of','- release', 'ingestion of', 'intake of']
for item in manual:
    propogated_chem_exclusive_prengram.add(item)           
propogated_chem_exclusive_prengram = tuple(propogated_chem_exclusive_prengram)

lf = CustomizedExclusivePreNgram(list(propogated_chem_exclusive_prengram), label_type='I-Chemical')
lf.apply(cdr_docs)
######################################################################


########################### Inclusive PreNgram
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
    
propogated_dis_inclusive_prengram = set()
if args.use_dis_InclusivePreNgram:
    for ix in range(1, 6):
        with open('{}/Disease_InclusivePreNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in tuple(pickle.load(f)[:5]):
                propogated_dis_inclusive_prengram.add(item)
manual = ["parkinson 's", 'torsade de', 'acute liver','neuroleptic malignant',"alzheimer 's",'congestive heart', 'migraine with','sexual side','renal cell', 'tic -']
for item in manual:
    propogated_dis_inclusive_prengram.add(item)           
propogated_dis_inclusive_prengram = tuple(propogated_dis_inclusive_prengram)

lf = CustomizedInclusivePreNgram(list(propogated_dis_inclusive_prengram), label_type='I-Disease')
lf.apply(cdr_docs)


propogated_chem_inclusive_prengram = set()
if args.use_chem_InclusivePreNgram:
    for ix in range(1, 6):
        with open('{}/Chemical_InclusivePreNgram_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in tuple(pickle.load(f)[:5]):
                propogated_chem_inclusive_prengram.add(item)
manual = ['external', 'vitamin', 'mk', 'mk -', 'cis', 'cis -', 'nik', 'nik -', 'ly', 'ly -', 'puromycin']
for item in manual:
    propogated_chem_inclusive_prengram.add(item)           
propogated_chem_inclusive_prengram = tuple(propogated_chem_inclusive_prengram)

lf = CustomizedInclusivePreNgram(list(propogated_chem_inclusive_prengram), label_type='I-Chemical')
lf.apply(cdr_docs)

######################################################################

########################### Dependency 

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
                if right > i+1 and not tokens[right-1].lemma_.lower() in ['disease', 'disorder', 'disorders']:
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
    
propogated_dis_dependency = set()
if args.use_dis_Dependency:
    for ix in range(1, 6):
        with open('{}/Disease_Dependency_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in tuple(pickle.load(f)[:5]):
                propogated_dis_dependency.add(item)
manual = [ 'StartDep:poss|HeadSurf:disease','EndDep:pobj:HeadSurf:disease','StartDep:compound|HeadSurf:cancer',
    'StartDep:amod|HeadSurf:dysfunction', 'StartDep:compound|HeadSurf:disease','StartDep:compound|HeadSurf:failure',
    'StartDep:compound|HeadSurf:anemia','StartDep:compound|HeadSurf:cancer']
for item in manual:
    propogated_dis_dependency.add(item)           
propogated_dis_dependency = tuple(propogated_dis_dependency)

lf = CustomizedDependencyMatcher(propogated_dis_dependency, label_type='I-Disease')
lf.apply(cdr_docs)

propogated_chem_dependency = set()
if args.use_chem_Dependency:
    for ix in range(1, 6):
        with open('{}/Chemical_Dependency_g{}_r{}.txt'.format(base_folder, args.group, ix), 'rb') as f:
            for item in tuple(pickle.load(f)[:5]):
                propogated_chem_dependency.add(item)
manual = [ 'StartDep:amod|HeadSurf:oxide','StartDep:compound|HeadSurf:chloride',
    'EndDep:amod:HeadSurf:aminonucleoside', 'StartDep:compound|HeadSurf:hydrochloride']
for item in manual:
    propogated_chem_dependency.add(item)           
propogated_chem_dependency = tuple(propogated_chem_dependency)

lf = CustomizedDependencyMatcher(propogated_chem_dependency, label_type='I-Chemical')
lf.apply(cdr_docs)

#######################################################################

exceptions = {
    "drug",
    "pre",
    "therapy",
    "anesthetia",
    "anesthetic",
    "neuroleptic",
    "saline",
    "stimulus"}


class Induced(TaggingRule):
    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        for i in range(1, len(instance['tokens']) - 3):
            lemma = instance['tokens'][i].lemma_.lower()
            if instance['tokens'][i].text == '-' and instance['tokens'][i +
                                                                        1].lemma_ == 'induce':
                labels[i] = 'O'
                labels[i + 1] = 'O'
                if instance['tokens'][i -
                                      1].lemma_ in exceptions or instance['tokens'][i -
                                                                                    1].pos_ == "PUNCT":
                    labels[i - 1] = 'O'
                else:
                    labels[i - 1] = 'I-Chemical'
        return labels


lf = Induced()
lf.apply(cdr_docs)

class Vitamin(TaggingRule):
    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens']) - 1):
            text = instance['tokens'][i].text.lower()
            if instance['tokens'][i].text.lower() == 'vitamin':
                labels[i] = 'I-Chemical'
                if len(instance['tokens'][i +
                                          1].text) <= 2 and instance['tokens'][i +
                                                                               1].text.isupper():
                    labels[i + 1] = 'I-Chemical'

        return labels


lf = Vitamin()
lf.apply(cdr_docs)


class Acid(TaggingRule):
    def apply_instance(self, instance):

        labels = ['ABS'] * len(instance['tokens'])

        tokens = instance['tokens']

        for i, t in enumerate(tokens):
            if i > 0 and t.text.lower(
            ) == 'acid' and tokens[i - 1].text.endswith('ic'):
                labels[i] = 'I-Chemical'
                labels[i - 1] = 'I-Chemical'

        return labels


lf = Acid()
lf.apply(cdr_docs)

class OtherPOS(TaggingRule):
    other_pos = {"ADP", "ADV", "DET", "VERB"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(0, len(instance['tokens'])):
            # Some chemicals with long names get tagged as verbs
            if instance['tokens'][i].pos_ in self.other_pos and instance['WISER_LABELS'][
                    'Organic Chemical'][i] == 'ABS' and instance['WISER_LABELS']['DictCore-Chemical'][i] == 'ABS':
                labels[i] = "O"
        return labels


lf = OtherPOS()
lf.apply(cdr_docs)


stop_words = {"a", "an", "as", "be", "but", "do", "even",
              "for", "from",
              "had", "has", "have", "i", "in", "is", "its", "just",
              "may", "my", "no", "not", "on", "or",
              "than", "that", "the", "these", "this", "those", "to", "very",
              "what", "which", "who", "with"}

class StopWords(TaggingRule):

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in stop_words:
                labels[i] = 'O'
        return labels


lf = StopWords()
lf.apply(cdr_docs)

class CommonOther(TaggingRule):
    other_lemmas = {'patient', '-PRON-', 'induce', 'after', 'study',
                    'rat', 'mg', 'use', 'treatment', 'increase',
                    'day', 'group', 'dose', 'treat', 'case', 'result',
                    'kg', 'control', 'report', 'administration', 'follow',
                    'level', 'suggest', 'develop', 'week', 'compare',
                    'significantly', 'receive', 'mouse',
                    'protein', 'infusion', 'output', 'area', 'effect',
                    'rate', 'weight', 'size', 'time', 'year',
                    'clinical', 'conclusion', 'outcome', 'man', 'woman',
                    'model', 'concentration'}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].lemma_ in self.other_lemmas:
                labels[i] = 'O'
        return labels


lf = CommonOther()
lf.apply(cdr_docs)


class Punctuation(TaggingRule):

    other_punc = {"?", "!", ";", ":", ".", ",",
                  "%", "<", ">", "=", "\\"}

    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])

        for i in range(len(instance['tokens'])):
            if instance['tokens'][i].text in self.other_punc:
                labels[i] = 'O'
        return labels


lf = Punctuation()
lf.apply(cdr_docs)

class PossessivePhrase(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i -
                                  1].text == "'s" or instance['tokens'][i].text == "'s":
                links[i] = 1

        return links


lf = PossessivePhrase()
lf.apply(cdr_docs)


class HyphenatedPrefix(LinkingRule):
    chem_mods = set(["alpha", "beta", "gamma", "delta", "epsilon"])

    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if (instance['tokens'][i - 1].text.lower() in self.chem_mods or
                    len(instance['tokens'][i - 1].text) < 2) \
                    and instance['tokens'][i].text == "-":
                links[i] = 1

        return links


lf = HyphenatedPrefix()
lf.apply(cdr_docs)

class PostHyphen(LinkingRule):
    def apply_instance(self, instance):
        links = [0] * len(instance['tokens'])
        for i in range(1, len(instance['tokens'])):
            if instance['tokens'][i - 1].text == "-":
                links[i] = 1

        return links


lf = PostHyphen()
lf.apply(cdr_docs)

dict_full = set()

with open('../datasets/AutoNER_dicts/{}/dict_full.txt'.format(args.dataset)) as f:
    for line in f.readlines():
        tokens = cdr_reader.get_tokenizer()(line.strip())
        term = tuple([str(x) for x in tokens])
        if len(term) > 1:
            dict_full.add(tuple(term))

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
lf.apply(cdr_docs)


print(score_labels_majority_vote(test_data, span_level=True))

save_label_distribution('../output-gen/{}/dev_data.p'.format(args.dataset), dev_data)
save_label_distribution('../output-gen/{}/test_data.p'.format(args.dataset), test_data)

cnt = Counter()
for instance in train_data + dev_data:
    for tag in instance['tags']:
        cnt[tag] += 1

disc_label_to_ix = {value[0]: ix for ix, value in enumerate(cnt.most_common())}
gen_label_to_ix = {'ABS': 0, 'I-Chemical': 1, 'I-Disease': 2, 'O': 3}

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