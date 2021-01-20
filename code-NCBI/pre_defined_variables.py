exception_list_NCBI_surface = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    
    'severe', 'early', 'onset', 'mild', "cancer", "cancers", "damage","disease", "diseases",
    "pain","injury", "injuries",
]

exception_list_NCBI_suffix = [
    "a", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "on", "or","that", "the", "these", "this", "those", "to", "very",
    "what", "which", "who", "with",'severe', 'early', 'onset', 'mild', "cancer", "cancers", "damage","disease", 
    "diseases","pain","injury", "injuries"
]

exception_list_NCBI_prefix = [
    "a", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "on", "or","that", "the", "these", "this", "those", "to", "very",
    "what", "which", "who", "with",'severe', 'early', 'onset', 'mild', "cancer", "cancers", "damage",
    "disease", "diseases","pain","injury", "injuries"
]

exception_list_NCBI_inclusve_postngram = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    'severe', 'early', 'onset', 'mild', "cancer", "cancers", "damage","disease", "diseases",
    "pain","injury", "injuries",
]

exception_list_NCBI_inclusive_prengram = [
     "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    'severe', 'early', 'onset', 'mild', "cancer", "cancers", "damage","disease", "diseases",
    "pain","injury", "injuries",
]

exception_list_NCBI_exclusive_prengram = [
     "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    'severe', 'early', 'onset', 'mild', "cancer", "cancers", "damage","disease", "diseases",
    "pain","injury", "injuries",
]

exception_list_NCBI_dependency = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could','severe', 'early', 'onset', 'mild', "cancer", "cancers", "damage","disease", "diseases",
    "pain","injury", "injuries",
]

def get_exception_list(task, rule_type):
    if task=='NCBI' and rule_type=='SurfaceForm':
        return exception_list_NCBI_surface
    elif task=='NCBI' and rule_type=='Suffix':
        return exception_list_NCBI_suffix
    elif task=='NCBI' and rule_type=='Prefix':
        return exception_list_NCBI_prefix
    elif task=='NCBI' and rule_type=='InclusivePostNgram':
        return exception_list_NCBI_inclusve_postngram
    elif task=='NCBI' and rule_type=='InclusivePreNgram':
        return exception_list_NCBI_inclusive_prengram
    elif task=='NCBI' and rule_type=='ExclusivePreNgram':
        return exception_list_NCBI_exclusive_prengram
    elif task=='NCBI' and rule_type=='Dependency':
        return exception_list_NCBI_dependency
    return []



def get_seed_list(task, rule_type):
    seeds = None
    if task=='NCBI' and rule_type=='SurfaceForm':
        seeds = set()
        with open('datasets/AutoNER_dicts/{}/dict_core.txt'.format(task)) as f:
            for line in f.readlines():
                line = line.strip().split()
                term = tuple(line[1:])
                if len(term) > 1 or len(term[0]) > 3:
                    seeds.add(' '.join(term))
    elif task=='NCBI' and rule_type=='Suffix':
        seeds = set([
            "edema", "toma", "coma", "noma","agia","cardia","trophy","toxic",
            "itis","emia","pathy","plasia",'skott', 'drich', 'umour', 'axia','iridia'
        ])
    elif task=='NCBI' and rule_type=='Prefix':
        seeds = set(['carc', 'myot', 'tela', 'ovari', 'atax', 'carcin', 'dystro'])
    elif task=='NCBI' and rule_type=='InclusivePostNgram':
        seeds = set(['- t', 'cell carcinoma', 'muscular dystrophy', "'s disease", 'carcinoma', 'dystrophy'])
    elif task=='NCBI' and rule_type=='InclusivePreNgram':
        seeds = set(['deficiency of', 'breast and ovarian', 'x - link', 'breast and', 'stage iii', 'myotonic','hereditary'])
    elif task=='NCBI' and rule_type=='ExclusivePreNgram':
        seeds = set(['suffer from', 'fraction of', 'pathogenesis of', 'cause severe'])
    elif task=='NCBI' and rule_type=='Dependency':
        seeds = set(['StartDep:compound|HeadSurf:syndrome','StartDep:compound|HeadSurf:disease',
                    'EndDep:compound:HeadSurf:syndrome','StartDep:compound|HeadSurf:deficiency',
                    'StartDep:amod|HeadSurf:dystrophy','StartDep:punct|HeadSurf:telangiectasia',
                    'StartDep:compound|HeadSurf:t','StartDep:amod|HeadSurf:dysplasia'])
    return seeds
                         
                         
import string                 
def get_negative_seed_list(task, rule_type):
    seeds = None
    if task=='NCBI' and rule_type=='SurfaceForm':
        neg_seeds = list(string.punctuation) + list(string.ascii_letters[:26])
        return set(neg_seeds)
    elif task=='NCBI' and rule_type=='Suffix':
        return ('ness', 'nant', 'tion', 'ting', 'enesis', 'riant', 'tein', 'sion', 'osis', 'lity')
    elif task=='NCBI' and rule_type=='Prefix':
        return ('defi', 'comp', 'fami', 'poly', 'chro', 'prot', 'enzym', 'sever', 'develo', 'varian')
    elif task=='NCBI' and rule_type=='InclusivePostNgram':
        return ('muscle', 'ataxia', 'system', 'defect', 'other cancer', 'of', 'i','ii')
    elif task=='NCBI' and rule_type=='InclusivePreNgram':
        return ('enzyme', 'primary', 'non -', 'a', 'the', 'that', 'and')
    elif task=='NCBI' and rule_type=='ExclusivePreNgram':
        return ('of', 'for', 'and', 'is', 'in the', 'on the', 'for the', '-pron', 'be', 'the',
             'suggest that', '- cell', 'presence of', 'expression of', 'majority of', 'associate with',
             'cause of', 'defect in', 'family with', 'impair in', 'loss of')
    elif task=='NCBI' and rule_type=='Dependency':
        return ('EndDep:appos:HeadSurf:t','EndDep:compound:HeadSurf:t','StartDep:amod|HeadSurf:defect',
    'StartDep:pobj|HeadSurf:cancer','EndDep:compound:HeadSurf:cancer','EndDep:compound:HeadSurf:disease',
    'StartDep:amod|HeadSurf:tumour','StartDep:amod|HeadSurf:deficiency')