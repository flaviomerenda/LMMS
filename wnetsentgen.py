from nltk.corpus import wordnet as wn
import mwtok

syn_str_relations = {
    'definition': {
        'fn': lambda syn: [syn.definition()],
        'patterns': ['%(sub)s means : %(obj)s']
    }
}

syn_relations = {
    'hyponym': {
        'fn': lambda syn: syn.hyponyms(),
        'patterns': [
            '%(sub)s has a narrower sense %(obj)s ',
            'A %(obj)s is a kind of %(sub)s']
    },
    'instance_hyponym': {
        'fn': lambda syn: syn.instance_hyponyms(),
        'patterns': [
            'a %(sub)s like %(obj)s']
    },
    'member_meronym': {
        'fn': lambda syn: syn.member_meronyms(),
        'patterns': ['%(obj)s is a member of %(sub)s']
    },
    'has_part': {
        'fn': lambda syn: syn.part_meronyms(),
        'patterns': ['%(obj)s is a part of %(sub)s']
    },
    'topic_domain': {
        'fn': lambda syn: syn.topic_domains(),
        'patterns': ['the word %(sub)s is used when talking about %(obj)s']
    },
    'usage_domain': {
        'fn': lambda syn: syn.usage_domains(),
        'patterns': ['the word %(sub)s can be used as a %(obj)s']
    },
    'member_of_domain_region': {
        'fn': lambda syn: syn.region_domains(),
        'patterns': ['the word %(sub)s is used in %(obj)s']
    },
    'attribute': {
        'fn': lambda syn: syn.attributes(),
        'patterns': [
            'a %(sub)s thing or person refers to the %(obj)s of that thing or person']
    },
    'entailment': {
        'fn': lambda syn: syn.entailments(),
        'patterns': ['if you %(sub)s something , you also %(obj)s that thing']
    },
    'cause': {
        'fn': lambda syn: syn.causes(),
        'patterns': ['When you %(sub)s something , that thing may %(obj)s']
    },
    'also_see': {
        'fn': lambda syn: syn.also_sees(),
        'patterns': ['An %(sub)s object could also be %(obj)s']
    },
    'verb_group': {
        'fn': lambda syn: syn.verb_groups(),
        'paterns': ['To %(sub)s or %(obj)s something is almost the same']
    },
    'similar_to': {
        'fn': lambda syn: syn.similar_tos(),
        'patterns': ['You can say it is %(sub)s or %(obj)s']
    }
}


lem_relations = {
    'antonym': {
        'fn': lambda lem: lem.antonyms(),
        'patterns': ['%(obj)s means the opposite of %(sub)s']
    },
    'derivationally_related_form': {
        'fn': lambda lem: lem.derivationally_related_forms(),
        'patterns': ['%(sub)s is derived from %(obj)s']
    },
    'pertainym': {
        'fn': lambda lem: lem.pertainyms(),
        'patterns': ['if something is %(sub)s , it involves a %(obj)s']
    }
}


def generate_synset_triples(syn_rels):
    for synset in list(wn.all_synsets()):
        for synrel, sr in syn_rels.items():
            for objsyn in sr['fn'](synset):
                yield (synset, synrel, objsyn)


def generate_lem_triples(lem_rels):
    for lemname in list(wn.all_lemma_names()):
        _lems = wn.lemmas(lemname)
        for lemrel, lr in lem_rels.items():
            for _lem in _lems:
                for objlem in lr['fn'](_lem):
                    yield (_lem, lemrel, objlem)


def generate_sensekey_triples(syn_rels, syn_str_rels, lem_rels):
    for synsub, synrel, synobj in generate_synset_triples(syn_rels):
        for lemsub in synsub.lemmas():
            for lemobj in synobj.lemmas():
                yield (lemsub.key(), synrel, lemobj.key())
    for synsub, synrel, strobj in generate_synset_triples(syn_str_rels):
        for lemsub in synsub.lemmas():
            yield (lemsub.key(), synrel, strobj)
    for lemsub, lemrel, lemobj in generate_lem_triples(lem_rels):
        yield (lemsub.key(), lemrel, lemobj.key())


def is_sensekey(strval):
    if ':' not in strval:
        return False
    if '%' not in strval:
        return False
    try:
        lem = wn.lemma_from_key(strval)
        return True
    except Exception as e:
        return False


def get_sk_lemma(sensekey):
    return sensekey.split('%')[0]


def sense_triple_pattern_as_entry(pattern, subsk, rel, obj):
    if not is_sensekey(subsk):
        print('Expecting %s to be a sensekey, but it is not?' % subsk)
        return None
    sublem = get_sk_lemma(subsk)
    objlem = get_sk_lemma(obj) if is_sensekey(obj) else None
    objval = obj if objlem is None else objlem
    if sublem == objval:
        # avoid, since generated sentence won't make much sense
        return None
    replacements = {'sub': sublem, 'obj': objval}
    replacement_senses = {sublem: subsk,
                          objval: obj if is_sensekey(obj) else None}
    # sent = pattern % replacements
    entry = {f: [] for f in ['token', 'token_mw', 'lemma', 'senses',
                             'pos', 'id']}
    for tok in pattern.split():
        is_slot = tok.startswith('%(')
        tok_mw = tok % replacements if is_slot else tok
        sense = replacement_senses[tok_mw] if is_slot else None
        entry['token'].extend(tok_mw.split())
        entry['token_mw'].append(tok_mw)
        entry['senses'].append(sense)
        entry['lemma'].append(None)
        entry['pos'].append(None)
        entry['id'].append(None)
    entry['sentence'] = ' '.join([t for t in entry['token_mw']])
    entry['idx_map_abs'] = mwtok.calc_idx_map_abs(entry['token_mw'])
    return entry


def sensekey_triple_as_entries(subsk, rel, obj,
                               syn_rels, syn_str_rels, lem_rels):
    if rel in syn_rels:
        for pat in syn_rels[rel].get('patterns', []):
            entry = sense_triple_pattern_as_entry(pat, subsk, rel, obj)
            if entry is not None:
                yield entry
    if rel in syn_str_rels:
        for pat in syn_str_rels[rel].get('patterns', []):
            entry = sense_triple_pattern_as_entry(pat, subsk, rel, obj)
            if entry is not None:
                yield entry
    if rel in lem_rels:
        for pat in lem_rels[rel].get('patterns', []):
            entry = sense_triple_pattern_as_entry(pat, subsk, rel, obj)
            if entry is not None:
                yield entry


def gen_wnet_sents(syn_rels, syn_str_rels, lem_rels):
    for subsk, rel, obj in generate_sensekey_triples(syn_rels,
                                                     syn_str_rels, lem_rels):
        for entry in sensekey_triple_as_entries(
                subsk, rel, obj, syn_rels, lem_rels):
            yield entry
