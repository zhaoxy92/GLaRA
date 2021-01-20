from tqdm.auto import tqdm

from wiser.rules import TaggingRule, LinkingRule, UMLSMatcher, DictionaryMatcher


class CustomizedDictionaryMatcher(TaggingRule):
    def __init__(self, name, terms, uncased=False, match_lemmas=False, i_label="I", abs_label="ABS", return_only_spans=False):
        self.name = name
        self.uncased = uncased
        self.match_lemmas = match_lemmas
        self.i_label = i_label
        self.abs_label = abs_label
        self.return_only_spans = return_only_spans
        self._load_terms(terms)

    def apply_instance(self, instance):
        tokens = self._normalize_instance_tokens(instance['tokens'])
        labels = [self.abs_label] * len(instance['tokens'])
        spans = {}
        # Checks whether any terms in the dictionary appear in the instance
        i = 0
        while i < len(tokens):
            if tokens[i] in self.term_dict:
                candidates = self.term_dict[tokens[i]]
                for c in candidates:
                    if i + len(c) <= len(tokens):
                        equal = True
                        for j in range(len(c)):
                            if tokens[i + j] != c[j]:
                                equal = False
                                break

                        # If tokens match, labels the instance tokens
                        if equal:
                            cand = ' '.join([tk for tk in tokens[i:i+len(c)]])
                            if self.uncased:
                                cand = cand.lower()
                            if cand in spans:
                                spans[cand].append((i, i+len(c)))
                            else:  
                                spans[cand] = [(i, i+len(c))]
                                
                            for j in range(i, i + len(c)):
                                labels[j] = self.i_label
                            
#                             i = i + len(c) - 1 # if comment out, matches the longest only.
#                             break
            i += 1

        # Additionally checks lemmas if requested. This will not overwrite
        # existing votes
        if self.match_lemmas:
            tokens = self._normalize_instance_tokens(instance['tokens'], lemmas=True)
            i = 0
            while i < len(tokens):
                if tokens[i] in self.term_dict:
                    candidates = self.term_dict[tokens[i]]
                    for c in candidates:
                        if i + len(c) <= len(tokens):
                            equal = True
                            for j in range(len(c)):
                                if tokens[i + j] != c[j]:
                                    equal = False
                                    break

                            # If tokens match, labels the instance tokens using map
                            if equal:
                                cand = ' '.join([tk for tk in tokens[i:i+len(c)]])
                                if self.uncased:
                                    cand = cand.lower()
                                if cand in spans:
                                    spans[cand].append((i, i+len(c)))
                                else:  
                                    spans[cand] = [(i, i+len(c))]
                                
                                for j in range(i, i + len(c)):
                                    labels[j] = self.i_label
                                
#                                 i = i + len(c) - 1  # if comment out, match the longest only
#                                 break
                i += 1
        if self.return_only_spans:
            return spans
        return labels, spans

    def _get_tr_name(self):
        return self.name

    def _normalize_instance_tokens(self, tokens, lemmas=False):
        if lemmas:
            normalized_tokens = [token.lemma_ for token in tokens]
        else:
            normalized_tokens = [token.text for token in tokens]

        if self.uncased:
            normalized_tokens = [token.lower() for token in normalized_tokens]

        return normalized_tokens

    def _normalize_terms(self, tokens):
        if self.uncased:
            return [token.lower() for token in tokens]
        return tokens

    def _load_terms(self, terms):
        self.term_dict = {}
        for term in terms:
            normalized_term = self._normalize_terms(term)

            if normalized_term[0] not in self.term_dict:
                self.term_dict[normalized_term[0]] = []

            self.term_dict[normalized_term[0]].append(normalized_term)

        # Sorts the terms in decreasing order so that we match the longest first
#         for first_token in self.term_dict.keys():
#             to_sort = self.term_dict[first_token]
#             self.term_dict[first_token] = sorted(
#                 to_sort, reverse=True, key=lambda x: len(x))



class CustomizedCommonSuffixes(TaggingRule):
    def __init__(self, suffixes_list, label_type='I', exceptions=set(), threshold=7, match_lemma=True):
        self.label_type=label_type
        self.suffixes = suffixes_list
        self.exceptions = exceptions
        self.threshold = threshold
        self.match_lemma=match_lemma
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        spans = {}
        for i in range(len(instance['tokens'])):
            w = instance['tokens'][i].text
            if self.match_lemma:
                w = instance['tokens'][i].lemma_
            if instance['tokens'][i].text.lower() in self.exceptions or instance['tokens'][i].lemma_.lower() in self.exceptions:
                continue
            
            if instance['tokens'][i].pos_!='NOUN':
                continue
                
            for suffix in self.suffixes:
                
                if len(w)>self.threshold and len(w)>len(suffix):
                    if w.endswith(suffix):
                        labels[i] = self.label_type
                        if suffix in spans:
                            spans[suffix].append((i, i+1))
                        else:
                            spans[suffix] = [(i, i+1)]
        return labels, spans
    
    
class CustomizedCommonPrefixes(TaggingRule):
    def __init__(self, prefixes_list, label_type='I', exceptions=set(), threshold=7, match_lemma=True):
        self.label_type=label_type
        self.prefixes = prefixes_list
        self.exceptions = exceptions
        self.threshold = threshold
        self.match_lemma=match_lemma
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        spans = {}
        for i, t in enumerate(instance['tokens']):
            w = t.text
            if self.match_lemma:
                w = t.lemma_
            for prefix in self.prefixes:
                if len(w)>self.threshold and len(w) > len(prefix) and \
                        w.startswith(prefix) and t.pos_=='NOUN' and not t.lemma_ in self.exceptions:
                    labels[i] = self.label_type
                    if prefix in spans:
                        spans[prefix].append((i,i+1))
                    else:
                        spans[prefix] = [(i, i+1)]
        return labels, spans
    
class CustomizedInclusivePostNgram(TaggingRule):
    def __init__(self, seed_list, label_type = 'I', length_list=None):
        self.label_type=label_type
        self.seeds = seed_list
        self.length_list = length_list
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        spans = {}  
        
        if self.length_list:
            for seed_len in self.length_list:
                for i in range(1,len(instance['tokens'])-seed_len):
                    cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                    if cand in self.seeds:
                        if instance['tokens'][i-1].pos_ == 'NOUN':
                            left = i-2
                            while left>=0 and instance['tokens'][left].pos_=='NOUN':
                                left-=1
                            if cand in spans:
                                spans[cand].append((left+1, i+seed_len))
                            else:
                                spans[cand] = [(left+1, i+seed_len)] 
                            for j in range(left+1, i+seed_len):
                                labels[j] = self.label_type
                    
                        elif instance['tokens'][i-1].pos_ in ['PROPN', 'ADJ']:
                            if cand in spans:
                                spans[cand].append((i-1, i+seed_len))
                            else:
                                spans[cand] = [(i-1, i+seed_len)] 
                            for j in range(i-1, i+seed_len):
                                labels[j] = self.label_type
        else:
            for seed in self.seeds:
                seed_len = len(seed.split())
                for i in range(1,len(instance['tokens'])-seed_len):
                    cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                    if cand == seed:
                        if instance['tokens'][i-1].pos_ in ['NOUN', 'ADJ', 'PROPN']:
                            left = i-2
                            if cand in spans:
                                spans[cand].append((left+1, i+seed_len))
                            else:
                                spans[cand] = [(left+1, i+seed_len)] 
                            for j in range(left+1, i+seed_len):
                                labels[j] = self.label_type
        return labels, spans
    
class CustomizedInclusivePreNgram(TaggingRule):
    def __init__(self, seed_list, label_type = 'I', length_list=None):
        self.label_type=label_type
        self.seeds = seed_list
        self.length_list = length_list
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        spans = {}  
        
        if self.length_list:
            for seed_len in self.length_list:
                for i in range(len(instance['tokens'])-seed_len-1):
                    cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                    if cand in self.seeds:
                        right = i+seed_len
                        if instance['tokens'][right].pos_ == 'NOUN':
#                             while right<len(instance['tokens']) and instance['tokens'][right].pos_=='NOUN':
#                                 right+=1
                            if cand in spans:
                                spans[cand].append((i, right+1))
                            else:
                                spans[cand] = [(i, right+1)] 
                            for j in range(i, right+1):
                                labels[j] = self.label_type
        else:
            for seed in self.seeds:
                seed_len = len(seed.split())
                for i in range(len(instance['tokens'])-seed_len-1):
                    cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                    if cand == seed:
                        right = i+seed_len
                        
                        if instance['tokens'][right].pos_=='NOUN':
                            if cand in spans:
                                spans[cand].append((i, right+1))
                            else:
                                spans[cand] = [(i, right+1)] 
                            for j in range(i, right+1):
                                labels[j] = self.label_type
                    

        return labels, spans
    
class CustomizedExclusivePreNgram(TaggingRule):
    def __init__(self, seed_list, label_type = 'I', length_list=None):
        self.label_type=label_type
        self.seeds = seed_list
        self.length_list = length_list
        
    def apply_instance(self, instance):
        labels = ['ABS'] * len(instance['tokens'])
        spans = {}  
    
        if self.length_list:
            for seed_len in self.length_list:
                for i in range(len(instance['tokens'])-seed_len-1):
                    cand = ' '.join([tk.lemma_ for tk in instance['tokens'][i:i+seed_len]]).lower()
                    if cand in self.seeds:
                        left = i+seed_len

                        if left<len(instance['tokens']) and instance['tokens'][left].pos_ == 'NOUN':
                            right = left+1
                            while right<len(instance['tokens']) and instance['tokens'][right].pos_=='NOUN':
                                if cand in spans:
                                    spans[cand].append((left, right))
                                else:
                                    spans[cand] = [(left, right)] 
                                for j in range(left, right):
                                    labels[j] = self.label_type
                                right+=1
                                
                            if cand in spans:
                                spans[cand].append((left, right))
                            else:
                                spans[cand] = [(left, right)] 
                            for j in range(left, right):
                                labels[j] = self.label_type
        else:
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

                            if cand in spans:
                                spans[cand].append((left, right))
                            else:
                                spans[cand] = [(left, right)] 
                            for j in range(left, right):
                                labels[j] = self.label_type

        return labels, spans
    
    
class CustomizedDependency(TaggingRule):
    def __init__(self, name, terms, uncased=False, match_lemmas=False, i_label="I", abs_label="ABS", return_only_spans=False):
        self.name = name
        self.uncased = uncased
        self.match_lemmas = match_lemmas
        self.i_label = i_label
        self.abs_label = abs_label
        self.return_only_spans = return_only_spans
        self._load_terms(terms)

    def apply_instance(self, instance):
        tokens = self._normalize_instance_tokens(instance['tokens'])
        deps = [tk.dep_ for tk in instance['tokens']]
        
        labels = [self.abs_label] * len(instance['tokens'])
        spans = {}
        i = 0
        while i < len(tokens):
            if tokens[i] in self.term_dict:
                candidates = self.term_dict[tokens[i]]
                for c in candidates:
                    if i + len(c) <= len(tokens):
                        equal = True
                        for j in range(len(c)):
                            if tokens[i + j] != c[j]:
                                equal = False
                                break
                        if equal and len(c)>=2:
                            
                            cand = 'StartDep:{}|HeadSurf:{}'.format(deps[i], tokens[i+len(c)-1].lower())  
                            if cand in spans:
                                spans[cand].append((i, i+len(c)))
                            else:  
                                spans[cand] = [(i, i+len(c))]
                            cand = 'EndDep:{}|HeadSurf:{}'.format(deps[i+len(c)-2], tokens[i+len(c)-1].lower())  
                            if cand in spans:
                                spans[cand].append((i, i+len(c)))
                            else:  
                                spans[cand] = [(i, i+len(c))]
                            
                            for j in range(i, i + len(c)):
                                labels[j] = self.i_label

            i += 1

        if self.match_lemmas:
            tokens = self._normalize_instance_tokens(instance['tokens'], lemmas=True)
            i = 0
            while i < len(tokens):
                if tokens[i] in self.term_dict:
                    candidates = self.term_dict[tokens[i]]
                    for c in candidates:
                        if i + len(c) <= len(tokens):
                            equal = True
                            for j in range(len(c)):
                                if tokens[i + j] != c[j]:
                                    equal = False
                                    break

                            if equal and len(c)>=2:
                                cand = ' '.join([tk for tk in tokens[i:i+len(c)]]).lower()
                                
                                cand = 'StartDep:{}|HeadSurf:{}'.format(deps[i], tokens[i+len(c)-1].lower())
                                
                                if cand in spans:
                                    spans[cand].append((i, i+len(c)))
                                else:  
                                    spans[cand] = [(i, i+len(c))]
                                cand = 'EndDep:{}|HeadSurf:{}'.format(deps[i+len(c)-2], tokens[i+len(c)-1].lower())  
                                if cand in spans:
                                    spans[cand].append((i, i+len(c)))
                                else:  
                                    spans[cand] = [(i, i+len(c))]
                                                               
                i += 1
        if self.return_only_spans:
            return spans
        return labels, spans

    def _get_tr_name(self):
        return self.name

    def _normalize_instance_tokens(self, tokens, lemmas=False):
        if lemmas:
            normalized_tokens = [token.lemma_ for token in tokens]
        else:
            normalized_tokens = [token.text for token in tokens]

        if self.uncased:
            normalized_tokens = [token.lower() for token in normalized_tokens]

        return normalized_tokens

    def _normalize_terms(self, tokens):
        if self.uncased:
            return [token.lower() for token in tokens]
        return tokens

    def _load_terms(self, terms):
        self.term_dict = {}
        for term in terms:
            normalized_term = self._normalize_terms(term)
            if normalized_term[0] not in self.term_dict:
                self.term_dict[normalized_term[0]] = []
            self.term_dict[normalized_term[0]].append(normalized_term)