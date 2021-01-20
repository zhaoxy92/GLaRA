exception_list_surface = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    
    'curve', 'function', 'level', 'computer', 'laptop', 'pc', 'mac', 'iphone', 'phone', 'imac', 'inch', 'surface'
]

exception_list_suffix = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    
    'curve', 'function', 'level', 'computer', 'laptop', 'pc', 'mac', 'iphone', 'phone', 'imac', 'inch', 'surface',
    
    'very', 'every', 'answer', 'between', 'music', 'mac', 'pc','computer','laptop','phone', 'iphone','imac','apple'
]

exception_list_prefix = [
    "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", "its", 
    "just","my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", "very","what", 
    "which", "who", "with", 'could', 'would', 'why', 'what', 'how', 'when', 'can', 'could',
    
    'curve', 'function', 'level', 'computer', 'laptop', 'pc', 'mac', 'iphone', 'phone', 'imac', 'inch', 'surface',
    
    'very', 'every', 'answer', 'between', 'music', 'mac', 'pc','computer', 'laptop','phone','iphone',
    'imac','apple', 'device', 'equipment'
]

exception_list_inclusve_postngram = []

exception_list_inclusive_prengram = []

exception_list_exclusive_prengram = []

exception_list_dependency = []

def get_exception_list(task, rule_type):
    if task=='LaptopReview' and rule_type=='SurfaceForm':
        return exception_list_surface
    elif task=='LaptopReview' and rule_type=='Suffix':
        return exception_list_suffix
    elif task=='LaptopReview' and rule_type=='Prefix':
        return exception_list_prefix
    elif task=='LaptopReview' and rule_type=='InclusivePostNgram':
        return exception_list_inclusve_postngram
    elif task=='LaptopReview' and rule_type=='InclusivePreNgram':
        return exception_list_inclusive_prengram
    elif task=='LaptopReview' and rule_type=='ExclusivePreNgram':
        return exception_list_exclusive_prengram
    elif task=='LaptopReview' and rule_type=='Dependency':
        return exception_list_dependency
    return []



def get_seed_list(task, rule_type):
    seeds = None
    if task=='LaptopReview' and rule_type=='SurfaceForm':
        seeds = set()
        with open('../datasets/AutoNER_dicts/{}/dict_core.txt'.format(task)) as f:
            for line in f.readlines():
                line = line.strip().split()
                term = tuple(line[1:])
                if len(term) > 1 or len(term[0]) > 3:
                    seeds.add(' '.join(term))
    elif task=='LaptopReview' and rule_type=='Suffix':
        seeds = set(["pad","oto","fox", 'chpad', 'rams'])
    elif task=='LaptopReview' and rule_type=='Prefix':
        seeds = set(['feat', 'softw', 'batt', 'Win', 'osx'])
    elif task=='LaptopReview' and rule_type=='InclusivePostNgram':
        seeds = set(['x', 'xp', 'vista', 'drive', 'processing'])
    elif task=='LaptopReview' and rule_type=='InclusivePreNgram':
        seeds = set(['windows', 'hard', 'extended', 'touch', 'boot'])
    elif task=='LaptopReview' and rule_type=='ExclusivePreNgram':
        seeds = set(['replace the', 'like the', 'love the', 'dislike the', 'hate the'])
    elif task=='LaptopReview' and rule_type=='Dependency':
        seeds = set([
            'StartDep:compound|HeadSurf:port', 'StartDep:compound|HeadSurf:button',
            'StartDep:nummod|HeadSurf:ram', 'StartDep:amod|HeadSurf:drive',
        ])
    return seeds
                         
                         
import string                 
def get_negative_seed_list(task, rule_type):
    seeds = None
    if task=='LaptopReview' and rule_type=='SurfaceForm':
        neg_seeds = list(string.punctuation) + list(string.ascii_letters[:26])
        neg_seeds +=[ "a", "and", "as", "be", "but", "do", "even","for", "from","had", "has", "have", "i", "in", "is", 
                     "its", "just", "my", "no", "not", "of", "on", "or","that", "the", "these", "this", "those", "to", 
                     "very","what", "which", "who", "with", "laptop", "computer", "pc"]
        return set(neg_seeds)
    elif task=='LaptopReview' and rule_type=='Suffix':
        return ('ion', 'ness', 'nant', 'lly', 'ary', 'est', 'ing', 'ist')
    elif task=='LaptopReview' and rule_type=='Prefix':
        return ('pro', 'edit', 'repa', 'rep', 'con', 'dis', 'appl', 'equip')
    elif task=='LaptopReview' and rule_type=='InclusivePostNgram':
        return ('screen', 'software', 'quality', 'technical', 'cut')
    elif task=='LaptopReview' and rule_type=='InclusivePreNgram':
        return ('mac', 'apple', 'a', 'launch', 'software')
    elif task=='LaptopReview' and rule_type=='ExclusivePreNgram':
        return ('of', 'for', 'and', 'is', 'in the', 'on the', 'for the', '-pron', 'be', 'the')
    elif task=='LaptopReview' and rule_type=='Dependency':
        return ('StartDep:compound|HeadSurf:option', 'EndDep:pobj:HeadSurf:plan','EndDep:nsubj:HeadSurf:design',
    'EndDep:pobj:HeadSurf:plan')