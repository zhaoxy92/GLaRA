from wiser.data.dataset_readers import *

cdr_reader = CDRCombinedDatasetReader()

exception_list_surface = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    
    'determine', 'baseline', 'decline', 'examine', 'pontine', 'vaccine','routine', 'crystalline', 'migraine','alkaline', 
    'midline', 'borderline','cocaine', 'medicine', 'medline','asystole', 'control', 'protocol','alcohol', 'aerosol', 
    'peptide', 'provide', 'outside', 'intestine', 'combine', 'delirium', 'VIP', 'diagnosis', 'apoptosis', 'prognosis',
    'metabolism','hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric',"drug","pre","therapy","anesthetia", "anesthetic",
    "neuroleptic","saline","stimulus", "aprotinin",'patient', '-PRON-', 'induce', 'after', 'study','rat', 'mg', 'use', 'treatment', 'increase', 'day', 'group', 'dose', 'treat', 'case', 'result','kg', 'control', 'report', 'administration', 
    'follow','level', 'suggest', 'develop', 'week', 'compare','significantly', 'receive', 'mouse','us', 'other', 'protein', 
    'infusion', 'output', 'area', 'effect','rate', 'weight', 'size', 'time', 'year', 'clinical', 'conclusion', 'outcome', 
    'man', 'woman','model', 'concentration',"?", "!", ";", ":", ".", ",","%", "<", ">", "=", "\\"
]

exception_list_suffix = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    
    'determine', 'baseline', 'decline', 'examine', 'pontine', 'vaccine','routine', 'crystalline', 'migraine','alkaline', 
    'midline', 'borderline','cocaine', 'medicine', 'medline','asystole', 'control', 'protocol','alcohol', 'aerosol', 
    'peptide', 'provide', 'outside', 'intestine', 'combine', 'delirium', 'VIP', 'diagnosis', 'apoptosis', 'prognosis',
    'metabolism','hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric',"drug","pre","therapy","anesthetia", "anesthetic",
    "neuroleptic","saline","stimulus", "aprotinin",'patient', '-PRON-', 'induce', 'after', 'study','rat', 'mg', 'use', 'treatment', 'increase', 'day', 'group', 'dose', 'treat', 'case', 'result','kg', 'control', 'report', 'administration', 
    'follow','level', 'suggest', 'develop', 'week', 'compare','significantly', 'receive', 'mouse','us', 'other', 'protein', 
    'infusion', 'output', 'area', 'effect','rate', 'weight', 'size', 'time', 'year', 'clinical', 'conclusion', 'outcome', 
    'man', 'woman','model', 'concentration',"?", "!", ";", ":", ".", ",","%", "<", ">", "=", "\\"
]

exception_list_prefix = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    
    'determine', 'baseline', 'decline', 'examine', 'pontine', 'vaccine','routine', 'crystalline', 'migraine','alkaline', 
    'midline', 'borderline','cocaine', 'medicine', 'medline','asystole', 'control', 'protocol','alcohol', 'aerosol', 
    'peptide', 'provide', 'outside', 'intestine', 'combine', 'delirium', 'VIP', 'diagnosis', 'apoptosis', 'prognosis',
    'metabolism','hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric',"drug","pre","therapy","anesthetia", "anesthetic",
    "neuroleptic","saline","stimulus", "aprotinin",'patient', '-PRON-', 'induce', 'after', 'study','rat', 'mg', 'use', 'treatment', 'increase', 'day', 'group', 'dose', 'treat', 'case', 'result','kg', 'control', 'report', 'administration', 
    'follow','level', 'suggest', 'develop', 'week', 'compare','significantly', 'receive', 'mouse','us', 'other', 'protein', 
    'infusion', 'output', 'area', 'effect','rate', 'weight', 'size', 'time', 'year', 'clinical', 'conclusion', 'outcome', 
    'man', 'woman','model', 'concentration',"?", "!", ";", ":", ".", ",","%", "<", ">", "=", "\\"
]

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
                    if entity_type == 'Chemical':
                        seeds.add(' '.join(term))

#         with open('../datasets/umls/umls_element_ion_or_isotope.txt', 'r') as f:   
#             for line in f.readlines():
#                 seeds.add(line.strip())

#         with open('../datasets/umls/umls_organic_chemical.txt', 'r') as f:
#             for line in f.readlines():
#                 seeds.add(line.strip())

#         with open('../datasets/umls/umls_antibiotic.txt', 'r') as f:
#             for line in f.readlines():
#                 seeds.add(line.strip())                    
                    
    elif task=='BC5CDR' and rule_type=='Suffix':
        seeds = set(['ine', 'ole', 'ol', 'ide', 'ine', 'ium', 'epam', 'pine','icin','dine',
                     'ridol', 'athy', 'zure', 'mide', 'fen', 'phine'])
    elif task=='BC5CDR' and rule_type=='Prefix':
        seeds = set(['chlor','levo','doxor','lithi','morphi','hepari','ketam','potas'])
    elif task=='BC5CDR' and rule_type=='InclusivePostNgram':
        seeds = set(['aminocaproic acid', '- aminocaproic acid', 'retinoic acid','dopa', 'tc', '- aminopyridine', 
                     'aminopyridine','- penicillamine', '- dopa', '- aspartate','fu', 'hydrochloride'])
    elif task=='BC5CDR' and rule_type=='InclusivePreNgram':
        seeds = set(['external', 'vitamin', 'mk', 'mk -', 'cis', 'cis -', 'nik', 'nik -', 'ly', 'ly -', 'puromycin'])
    elif task=='BC5CDR' and rule_type=='ExclusivePreNgram':
        seeds = set(['dosage of', 'sedation with', 'mg of', 'application of','- release', 'ingestion of', 'intake of'])
    elif task=='BC5CDR' and rule_type=='Dependency':
        seeds = set([
            'StartDep:amod|HeadSurf:oxide','StartDep:compound|HeadSurf:chloride','StartDep:amod|HeadSurf:acid',  
            'StartDep:compound|HeadSurf:acid','EndDep:amod:HeadSurf:aminonucleoside', 
            'StartDep:compound|HeadSurf:hydrochloride'
        ])
    return seeds
                         
                         
import string                 
def get_negative_seed_list(task, rule_type):
    seeds = None
    if task=='BC5CDR' and rule_type=='SurfaceForm':
        neg_seeds = list(string.punctuation) + list(string.ascii_letters[:26])
        neg_seeds.pop(neg_seeds.index('-'))
        neg_seeds += ["a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its",
                      "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very",
                      "what", "which", "who", "with", ]
        return set(neg_seeds)
    elif task=='BC5CDR' and rule_type=='Suffix':
        return ('ing', 'tion', 'tive', 'tory', 'inal','ance', 'duce', 'atory', 'mine', 'line', 'tin',
             'rate', 'late','ular', 'etic', 'onic', 'ment', 'nary', 'lion', 'ysis', 'logue', 'mone')
    elif task=='BC5CDR' and rule_type=='Prefix':
        return ('meth', 'hepa','prop','contr','pheno','contra','acetyl','dopami')
    elif task=='BC5CDR' and rule_type=='InclusivePostNgram':
        return ('drug', 'cocaine', 'calcium', 'receptor agonist', 'blockers', 'block agent', 'inflammatory drug')
    elif task=='BC5CDR' and rule_type=='InclusivePreNgram':
        return ('reduce', 'all', 'a', 'the', 'of', 'alpha', 'alpha -', 'beta', 'beta -')
    elif task=='BC5CDR' and rule_type=='ExclusivePreNgram':
        return ('of to', 'to the', 'be the', 'with the', 'in the', 'on the', 'for the')
    elif task=='BC5CDR' and rule_type=='Dependency':
        return ('EndDep:pobj:HeadSurf:acid', 'EndDep:pobj:HeadSurf:a', 'EndDep:pobj:HeadSurf:the','EndDep:pobj:HeadSurf:a',
                'StartDep:compound|HeadSurf:a', )