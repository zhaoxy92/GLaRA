from wiser.data.dataset_readers import *

cdr_reader = CDRCombinedDatasetReader()

exception_list_surface = [ "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could', 'determine', 'baseline', 'decline', 'examine', 'pontine', 'vaccine','routine', 'crystalline', 'migraine','alkaline', 'midline', 'borderline','cocaine', 'medicine', 'medline','asystole', 'control', 'protocol','alcohol', 'aerosol', 'peptide', 'provide', 'outside', 'intestine', 'combine', 'delirium', 'VIP'] + ['diagnosis', 'apoptosis', 'prognosis', 'metabolism', 'hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric', "drug","pre","therapy","anesthetia",  "anesthetic","neuroleptic","saline","stimulus", "aprotinin",'patient', '-PRON-', 'induce', 'after', 'study','rat', 'mg', 'use', 'treatment', 'increase','day', 'group', 'dose', 'treat', 'case', 'result','kg', 'control', 'report', 'administration', 'follow',  'level', 'suggest', 'develop', 'week', 'compare','significantly', 'receive', 'mouse','us', 'other', 'protein', 'infusion', 'output', 'area', 'effect','rate', 'weight', 'size', 'time', 'year','clinical', 'conclusion', 'outcome', 'man', 'woman','model', 'concentration', "?", "!", ";", ":", ".", ",","%", "<", ">", "=", "\\"]

exception_list_suffix = ['determine', 'baseline', 'decline', 'examine', 'pontine', 'vaccine','routine', 'crystalline', 
'migraine','alkaline', 'midline', 'borderline','cocaine', 'medicine', 'medline','asystole', 'control', 'protocol','alcohol', 'aerosol', 'peptide', 'provide', 'outside', 'intestine','combine', 'delirium', 'VIP', 'diagnosis', 'apoptosis', 'prognosis', 'metabolism', 'hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric', "drug","pre","therapy","anesthetia", "anesthetic","neuroleptic","saline","stimulus", "aprotinin", 'analysis', 'solution', 'analog', 'anaesthesia', 'patient', '-PRON-', 'induce', 'after', 'study','rat', 'mg', 'use', 'treatment', 'increase','analogue', 'urinary','day', 'group', 'dose', 'treat', 'case', 'result','kg', 'control', 'report', 'administration', 'follow','level', 'suggest', 'develop', 'week', 'compare','significantly', 'receive', 'mouse','us', 'other','healthy','protein', 'infusion', 'output', 'area', 'effect','rate', 'weight', 'size', 'time', 'year','candidate', 'clinical', 'conclusion', 'outcome', 'man', 'woman','model', 'concentration', 'blood', 'juice','decline', "?", "!", ";", ":", ".", ",","%", "<", ">", "=", "\\"]

exception_list_prefix = ['determine', 'baseline', 'decline', 'examine', 'pontine', 'vaccine','routine', 'crystalline', 'migraine','alkaline', 'midline', 'borderline','cocaine', 'medicine', 'medline','asystole', 'control', 'protocol','alcohol', 'aerosol', 'peptide', 'provide', 'outside', 'intestine', 'combine', 'delirium', 'VIP'] + ['diagnosis', 'apoptosis', 'prognosis', 'metabolism', 'hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric', "drug","pre","therapy","anesthetia", "anesthetic","neuroleptic","saline","stimulus", "aprotinin", 'patient', '-PRON-', 'induce', 'after', 'study','rat', 'mg', 'use', 'treatment', 'increase', 'day', 'group', 'dose', 'treat', 'case', 'result','kg', 'control', 'report', 'administration', 'follow','level', 'suggest', 'develop', 'week', 'compare','significantly', 'receive', 'mouse','us', 'other', 'protein', 'infusion', 'output', 'area', 'effect','rate', 'weight', 'size', 'time', 'year', 'clinical', 'conclusion', 'outcome', 'man', 'woman','model', 'concentration', 'blood', 'juice', "?", "!", ";", ":", ".", ",","%", "<", ">", "=", "\\"]

exception_list_inclusve_postngram = []

exception_list_inclusive_prengram = []

exception_list_exclusive_prengram = []

exception_list_dependency = []

def get_exception_list(task, rule_type):
    if task=='BC5CDR' and rule_type=='SurfaceForm':
        return exception_list_surface
    elif task=='BC5CDR' and rule_type=='Suffix':
        return exception_list_suffix
    elif task=='BC5CDR' and rule_type=='Prefix':
        return exception_list_prefix
    elif task=='BC5CDR' and rule_type=='InclusivePostNgram':
        return exception_list_inclusve_postngram
    elif task=='BC5CDR' and rule_type=='InclusivePreNgram':
        return exception_list_inclusive_prengram
    elif task=='BC5CDR' and rule_type=='ExclusivePreNgram':
        return exception_list_exclusive_prengram
    elif task=='BC5CDR' and rule_type=='Dependency':
        return exception_list_dependency
    return []



def get_seed_list(task, rule_type):
    seeds = None
    if task=='BC5CDR' and rule_type=='SurfaceForm':            
        seeds = set()
        with open('../datasets/AutoNER_dicts/BC5CDR/dict_core.txt') as f:
            for line in f.readlines():
                line = line.strip().split(None, 1)
                entity_type = line[0]
                tokens = cdr_reader.get_tokenizer()(line[1])
                term = tuple([str(x) for x in tokens])

                if len(term) > 1 or len(term[0]) > 3:
                    if entity_type == 'Disease':
                        seeds.add(' '.join(term))
#         with open('../../data/umls/umls_disease_or_syndrome.txt', 'r') as f:
#             for line in f.readlines():
#                 seeds.add(line.strip())

    elif task=='BC5CDR' and rule_type=='Suffix':
        seeds = set([ "agia", "cardia", "trophy", "itis","emia", "enia", "pathy", "plasia", "lism", "osis",'epsy', 'nson'])
    elif task=='BC5CDR' and rule_type=='Prefix':
        seeds = set(['anemi','dyski', 'heada', 'hypok', 'hypert','hyper', 'hypo', 'ische', 'arthr', 'hypox', 'toxic', 'arrhyt', 'ischem', 'hypert', 'dysfunc'])
    elif task=='BC5CDR' and rule_type=='InclusivePostNgram':
        seeds = set(["'s disease", 'infarction', "'s sarcoma" ,'epilepticus' , 'artery disease', 'de pointe','insufficiency', 'with aura', 'artery spasm', "'s encephalopathy"])
    elif task=='BC5CDR' and rule_type=='InclusivePreNgram':
        seeds = set([ "parkinson 's", 'torsade de', 'acute liver','neuroleptic malignant',"alzheimer 's",'congestive heart', 'migraine with','sexual side','renal cell', 'tic -'])
    elif task=='BC5CDR' and rule_type=='ExclusivePreNgram':
        seeds = set(['to induce', 'w -', 'and severe', 'suspicion of', 'die of', 'have severe', 
    'of persistent', 'cyclophosphamide associate'])
    elif task=='BC5CDR' and rule_type=='Dependency':
        seeds = set(['StartDep:poss|HeadSurf:disease','EndDep:pobj:HeadSurf:disease','StartDep:compound|HeadSurf:cancer',
    'StartDep:amod|HeadSurf:dysfunction', 'StartDep:compound|HeadSurf:disease','StartDep:compound|HeadSurf:failure',
    'StartDep:compound|HeadSurf:disorder','StartDep:compound|HeadSurf:anemia','StartDep:compound|HeadSurf:cancer'
])
    return seeds
                         
                         
import string                 
def get_negative_seed_list(task, rule_type):
    seeds = None
    if task=='BC5CDR' and rule_type=='SurfaceForm':
        neg_seeds = list(string.punctuation) + list(string.ascii_letters[:26])
        neg_seeds.pop(neg_seeds.index('-'))
        neg_seeds += ["a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", "just","my", "no", "not", "of", "on", "or", "that", "the", "these", "this", "those", "to", "very","what", "which", "who", "with", ] + ['illness', 'renal', 'liver', 'acute', 'aprotinin', 'respiratory']
        return set(neg_seeds)
    elif task=='BC5CDR' and rule_type=='Suffix':
        return ('ing', 'tion', 'tive','lity' ,'mone', 'fect', 'crease', 'sion', 'lion', 'etic','ency',
             'ture','elet','gical','nosis','sive', 'ment', 'tory', 'sion')
    elif task=='BC5CDR' and rule_type=='Prefix':
        return ('symp', 'resp', 'funct', 'inter', 'decre', 'prote', 'neuro', 'cardi', 'myoca', 'ventr','decre','syst')
    elif task=='BC5CDR' and rule_type=='InclusivePostNgram':
        return ('toxicity', 'pain', 'fever','function','blood pressure','effect', 'impairment', 'loss','event' ,'protein', 'pressure', 'impair','phenomenon', 'side effect','system','of disease')
    elif task=='BC5CDR' and rule_type=='InclusivePreNgram':
        return ('renal function', 'decrease in', 'increase in', 'reduction in', 'rise in', 'loss of', 'chronic liver','abnormality in', 'human immunodeficiency','optic nerve', 'drug -', 'non -')
    elif task=='BC5CDR' and rule_type=='ExclusivePreNgram':
        return ('seizure and', 'symptom and', 'dysfunction and', 'failure with', 'sign of', 'lead to')
    elif task=='BC5CDR' and rule_type=='Dependency':
        return ('StartDep:amod|HeadSurf:toxicity','StartDep:amod|HeadSurf:impairment','StartDep:amod|HeadSurf:syndrome'
    'StartDep:amod|HeadSurf:complication','StartDep:amod|HeadSurf:symptom','StartDep:amod|HeadSurf:damage',
    'StartDep:amod|HeadSurf:disease','EndDep:nmod:HeadSurf:b','EndDep:conj:HeadSurf:arrhythmia',
    'EndDep:pobj:HeadSurf:symptom','StartDep:amod|HeadSurf:function',' StartDep:amod|HeadSurf:damage',
    'StartDep:compound|HeadSurf:loss')