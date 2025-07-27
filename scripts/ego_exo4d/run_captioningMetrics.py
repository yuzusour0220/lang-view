from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb

import subprocess
import threading
import multiprocessing

import sys, math, re
import copy
from collections import defaultdict
import math
import six
from six.moves import cPickle

import json
import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

import nltk
import spacy
import spacy_transformers
import pathlib

from typing import Dict, Pattern, Set, Iterable, List, Tuple, Union
import itertools
from spacy import attrs
from spacy.language import Language
from spacy.symbols import PROPN, VERB
from spacy.tokens import Doc, Span, Token
from pathlib import Path
import collections
from ast import literal_eval
from spacy.symbols import NOUN, VERB

import evaluate


def ospif(file):
    return os.path.isfile(file)

def ospid(dir_):
    return os.path.isdir(dir_)

def pkl_dmp(obj, fp):
    with open(fp, "wb") as fo:
        pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)
        
def pkl_ld(fp):
    with open(fp, "rb") as fi:
        pkl_content = pickle.load(fi)
    return pkl_content

def json_ld(fp):
    with open(fp, "r") as fi:
        json_content = json.load(fi)
    return json_content

def json_dmp(obj, fp):
    with open(fp, "w") as fo:
        json.dump(obj, fo)


# DEFAULT_DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve() / "data"

NUMERIC_ENT_TYPES: Set[str] = {
    "ORDINAL",
    "CARDINAL",
    "MONEY",
    "QUANTITY",
    "PERCENT",
    "TIME",
    "DATE",
}
SUBJ_DEPS: Set[str] = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS: Set[str] = {"attr", "dobj", "dative", "oprd"}
AUX_DEPS: Set[str] = {"aux", "auxpass", "neg"}

REPORTING_VERBS: Set[str] = {
    "according",
    "accuse",
    "acknowledge",
    "add",
    "admit",
    "agree",
    "allege",
    "announce",
    "argue",
    "ask",
    "assert",
    "believe",
    "blame",
    "charge",
    "cite",
    "claim",
    "complain",
    "concede",
    "conclude",
    "confirm",
    "contend",
    "criticize",
    "declare",
    "decline",
    "deny",
    "describe",
    "disagree",
    "disclose",
    "estimate",
    "explain",
    "fear",
    "hope",
    "insist",
    "maintain",
    "mention",
    "note",
    "observe",
    "order",
    "predict",
    "promise",
    "recall",
    "recommend",
    "reply",
    "report",
    "say",
    "state",
    "stress",
    "suggest",
    "tell",
    "testify",
    "think",
    "urge",
    "warn",
    "worry",
    "write",
}

MATCHER_VALID_OPS: Set[str] = {"!", "+", "?", "*"}

POS_REGEX_PATTERNS: Dict[str, Dict[str, str]] = {
    "en": {
        "NP": r"<DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+",
        "PP": r"<ADP> <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN> <PART>?)+",
        "VP": r"<AUX>* <ADV>* <VERB>",
    }
}

RE_MATCHER_TOKPAT_DELIM: Pattern = re.compile(r"\s+")
RE_MATCHER_SPECIAL_VAL: Pattern = re.compile(r"^(int|bool)\([^: ]+\)$", flags=re.UNICODE)

RE_ACRONYM: Pattern = re.compile(
    r"(?:^|(?<=\W))"
    r"(?:"
    r"(?:(?:(?:[A-Z]\.?)+[a-z0-9&/-]?)+(?:[A-Z][s.]?|\ds?))"
    r"|"
    r"(?:\d(?:\-?[A-Z])+)"
    r")"
    r"(?:$|(?=\W))",
    flags=re.UNICODE,
)

RE_LINEBREAK: Pattern = re.compile(r"(\r\n|[\n\v])+")
RE_NONBREAKING_SPACE: Pattern = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)

# regexes for cleaning up crufty terms
RE_DANGLING_PARENS_TERM: Pattern = re.compile(
    r"(?:\s|^)(\()\s{1,2}(.*?)\s{1,2}(\))(?:\s|$)", flags=re.UNICODE
)
RE_LEAD_TAIL_CRUFT_TERM: Pattern = re.compile(r"^[^\w(-]+|[^\w).!?]+$", flags=re.UNICODE)
RE_LEAD_HYPHEN_TERM: Pattern = re.compile(r"^-([^\W\d_])", flags=re.UNICODE)
RE_NEG_DIGIT_TERM: Pattern = re.compile(r"(-) (\d)", flags=re.UNICODE)
RE_WEIRD_HYPHEN_SPACE_TERM: Pattern = re.compile(
    r"(?<=[^\W\d]) (-[^\W\d])", flags=re.UNICODE
)
RE_WEIRD_APOSTR_SPACE_TERM: Pattern = re.compile(
    r"([^\W\d]+) ('[a-z]{1,2}\b)", flags=re.UNICODE
)


# CUSTOM CONSTANTS

RE_SUMMARY = re.compile(r"# *sum[m]+[ae][r]+y")

_parser_order = ['trf', 'lg', 'md', 'sm']


def get_main_verbs_of_sent(sent: Span) -> List[Token]:
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [
        tok for tok in sent if tok.pos == VERB and tok.dep_ not in AUX_DEPS
    ]


def get_subjects_of_verb(verb: Token) -> List[Token]:
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts if tok.dep_ in SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs


def get_objects_of_verb(verb: Token) -> List[Token]:
    """
    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """
    objs = [tok for tok in verb.rights if tok.dep_ in OBJ_DEPS]
    # get open clausal complements (xcomp)
    objs.extend(tok for tok in verb.rights if tok.dep_ == "xcomp")
    # get additional conjunct objects
    objs.extend(tok for obj in objs for tok in _get_conjuncts(obj))
    return objs


def _get_conjuncts(tok: Token) -> List[Token]:
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights if right.dep_ == "conj"]


def get_span_for_compound_noun(noun: Token) -> Tuple[int, int]:
    """Return document indexes spanning all (adjacent) tokens in a compound noun."""
    min_i = noun.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ == "compound", reversed(list(noun.lefts))
        )
    )
    return (min_i, noun.i)


def get_span_for_verb_auxiliaries(verb: Token) -> Tuple[int, int]:
    """
    Return document indexes spanning all (adjacent) tokens
    around a verb that are auxiliary verbs or negations.
    """
    min_i = verb.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in AUX_DEPS, reversed(list(verb.lefts))
        )
    )
    max_i = verb.i + sum(
        1
        for _ in itertools.takewhile(lambda x: x.dep_ in AUX_DEPS, verb.rights)
    )
    return (min_i, max_i)


def get_compound_nouns(sent):
    nouns = []
    for c in sent.root.children:
        pot_noun = get_compound_recursive(c).strip()
        if pot_noun != '':
            nouns.append(pot_noun)
    return nouns


def get_compound_recursive(node, part_of_compound_noun=False):
    if (part_of_compound_noun or node.pos_ == 'NOUN') and node.pos_ not in {'DET'}:
        return '{} {}'.format(node.text, ' '.join([get_compound_recursive(c, True) for c in node.children]))
    else:
        return ' '.join([get_compound_recursive(c, part_of_compound_noun) for c in node.children])


def load_sense_dict(sense_dict_path):
    df = pd.read_csv(sense_dict_path, converters={"senses": literal_eval}).to_dict()
    sense_dict = {}
    for row in df['lemma']:
        sense_dict[df['lemma'][row]] = df['senses'][row]
    return sense_dict


def get_word_sense(parsed_noun, sentence, sense_dict):
    noun_senses = sense_dict.values()

    noun_sense = None

    word_list = [w.lemma_ for w in sentence]

    for sense_word in sense_dict[parsed_noun.lemma_]:
        if sense_word in word_list:
            noun_sense = sense_word.replace(' ', '')
            break
    return noun_sense


# remove hashtags so spacy doesn't have any problems
def clean(text, verbose=False):
    empty_text_count = 0
    text = text.lower()
    text = re.sub("# +", "#", text)
    text = re.sub("#unsure|#unknown|#summary| [xyz] ", " ", text)
    text = re.sub("#[a-z]", " ", text)
    text = re.sub("^[b-z] | [b-z] | [b-z]$", " the person ", text)
    text = re.sub("#", " ", text)
    text = re.sub(" +", " ", text)
    text = text.strip()
    try:
        # Try removing full stops at the end of a narration to reduce num. unique narrations
        text = text[:-1] if text[-1] == "." else text
    except:
        empty_text_count += 1
    text = text.strip()
    if verbose:
        print(f"-Found {empty_text_count} instances of empty text.")
    return text


def extract_vo_from_sent(sent, compound_whitelist, noun_sense_dict, verb_sense_dict):

    vo_tuples = []

    start_i = sent[0].i

    verbs = get_main_verbs_of_sent(sent)

    for verb in verbs:
        objs = get_objects_of_verb(verb)
        modifiers = [
            child
            for child in verb.children
            if child.dep_ in ["prep", "advmod", "prt"]
            and child.pos_ in ["ADP", "PART"]
        ]
        #Only take the first modifier as others can be incorrect/too much detail
        if len(modifiers) > 1:
            modifiers = modifiers[0]
        else:
            modifiers = None
        if not objs:
            continue

        verb_vs = None
        if verb.lemma_ in verb_sense_dict:
            verb_vs = get_word_sense(verb, sent, verb_sense_dict)

        verb_span = get_span_for_verb_auxiliaries(verb)
        verb = sent[verb_span[0] - start_i : verb_span[1] - start_i + 1]

        for obj in objs:

            # corner case for "NOUN of X" or "X NOUN"
            if obj.lemma_ in compound_whitelist:
                true_obj = obj
                if len(list(obj.rights))>0:
                    rchild = list(obj.rights)[0]
                    if rchild.lemma_=='of' and len(list(rchild.rights))>0:
                        true_obj = list(rchild.rights)[0]
                elif len(list(obj.lefts))>0:
                    true_obj = list(obj.lefts)[-1]
                obj = true_obj if true_obj.pos == NOUN else obj


            if obj.pos == NOUN:
                span = get_span_for_compound_noun(obj)
            elif obj.pos == VERB:
                span = get_span_for_verb_auxiliaries(obj)
            else:
                span = (obj.i, obj.i)
            np = sent[span[0] - start_i : span[1] - start_i + 1]
            compound_nouns = get_compound_recursive(obj)

            obj_ns = None
            if obj.text in noun_sense_dict:
                obj_ns = get_word_sense(obj, sent, noun_sense_dict)

            vo_tuples.append((verb, verb_vs, modifiers, obj, obj_ns, np, compound_nouns))

    return vo_tuples, {'verb_missing': len(verbs)==0}


def extract_verbNnoun_v1(narration_text,
                         nlp, 
                         all_stopwords, 
                         dct_nn2lbl=None, 
                         dct_vrb2lbl=None,
                         compound_whitelist=None,
                         verb_sense_dict=None,
                         noun_sense_dict=None,):
    assert dct_nn2lbl
    assert dct_vrb2lbl

    clean_narration_text = clean(narration_text)
    narration_text = narration_text.lower()
    narration_text = narration_text.replace('\n', '')
    
    text = nltk.word_tokenize(narration_text)
    tags = nltk.pos_tag(text)
    del text
    
    unq_vrbs = set() 
    unq_nns = set()
    unq_nnChnks = set()
    if len(tags) > 4:
        text = nlp(narration_text)
        noun_list = list(text.noun_chunks)
        
        noun_vec = []
        verb_vec = []
    
        for word in text:
            if (word.pos_ == 'VERB'):
                word_lemma = word.lemma_
                if dct_vrb2lbl:
                    if word_lemma in dct_vrb2lbl:
                        unq_vrbs.add(list(dct_vrb2lbl[word_lemma])[0])
                    else:
                        unq_vrbs.add(word_lemma)
                else:
                    unq_vrbs.add(word_lemma)
            
            if (word.pos_ == 'NOUN'):
                word_lemma = word.lemma_
                if dct_nn2lbl:
                    if word_lemma in dct_nn2lbl:
                        unq_nns.add(list(dct_nn2lbl[word_lemma])[0])
                    else:
                        unq_nns.add(word_lemma)
                else:
                    unq_nns.add(word_lemma)
        
        for word in noun_list:
            word_lemma = word.lemma_
            word_lemma = ' '.join([w for w in word_lemma.split() if w not in all_stopwords])
            if dct_nn2lbl: 
                if word_lemma in dct_nn2lbl:
                    unq_nnChnks.add(list(dct_nn2lbl[word_lemma])[0])
                else:
                    unq_nnChnks.add(word_lemma)
            else:
                unq_nnChnks.add(word_lemma)
    
    text = nlp(clean_narration_text)
    if len(text) > 0:
        vnn, sent_stats = extract_vo_from_sent(text,
                                               compound_whitelist,
                                               noun_sense_dict,
                                               verb_sense_dict)
    else:
        vnn = []

    vo_vrb = set()
    vo_nn = set()
    vo_vn = set()
    if len(vnn) > 0:
        verb, verb_sense, noun, noun_sense = vnn[0][0], vnn[0][1], vnn[0][3], vnn[0][4]
        if dct_vrb2lbl: 
            if verb.lemma_ in dct_nn2lbl:
                vo_vrb.add(list(dct_nn2lbl[verb.lemma_])[0])
            else:
                vo_vrb.add(verb.lemma_)
        else:
            vo_vrb.add(verb.lemma_)
        if dct_nn2lbl: 
            if noun.lemma_ in dct_nn2lbl:
                vo_nn.add(list(dct_nn2lbl[noun.lemma_])[0])
            else:
                vo_nn.add(noun.lemma_)
        else:
            vo_nn.add(noun.lemma_)

    assert len(vo_vrb) in [0, 1]
    assert len(vo_nn) in [0, 1]
    if len(vo_vrb) == len(vo_nn) == 1:
        vo_vn.add(list(vo_vrb)[0] + " " + list(vo_nn)[0])
    elif len(vo_vrb) == 1:
        vo_vn.add(list(vo_vrb)[0])
    elif len(vo_nn) == 1:
        vo_vn.add(list(vo_nn)[0])

    return unq_vrbs, unq_nns, unq_nnChnks, vo_vrb, vo_nn, vo_vn


# ----------------------- BLEU ------------------------
def precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return (len(words), counts)


def cook_refs(refs, eff=None, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram,count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    # Calculate effective reference sentence length.
    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen))/len(reflen)

    ## lhuang: N.B.: leave reflen computaiton to the very end!!
    
    ## lhuang: N.B.: in case of "closest", keep a list of reflens!! (bad design)

    return (reflen, maxcounts)


def cook_test(test, refinfo, eff=None, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''
    reflen, refmaxcounts = refinfo
    testlen, counts = precook(test, n, True)

    result = {}

    # Calculate effective reference sentence length.
    
    if eff == "closest":
        result["reflen"] = min((abs(l-testlen), l) for l in reflen)[1]
    else: ## i.e., "average" or "shortest" or None
        result["reflen"] = reflen

    result["testlen"] = testlen

    result["guess"] = [max(0,testlen-k+1) for k in range(1,n+1)]

    result['correct'] = [0]*n
    for (ngram, count) in counts.items():
        result["correct"][len(ngram)-1] += min(refmaxcounts.get(ngram,0), count)

    return result


class BleuScorer(object):
    """Bleu scorer.
    """

    __slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"
    # special_reflen is used in oracle (proportional effective ref len for a node).

    def copy(self):
        ''' copy the refs.'''
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        ''' singular instance '''

        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''
        
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

        self._score = None ## need to recompute

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        '''return (bleu, len_ratio) pair'''
        return (self.fscore(option=option), self.ratio(option=option))

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen        

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(cook_test(t, rs))
        self._score = None

        return self

    def rescore(self, new_test):
        ''' replace test(s) with new test(s), and returns the new score.'''
        
        return self.retest(new_test).compute_score()

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''
        # print(other)
        if type(other) is tuple:
            ## avoid creating new BleuScorer instances
            # print(other[0], other[1])
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None ## need to recompute

        return self        

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option="average"):
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(self, reflens, option=None, testlen=None):
        
        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens))/len(reflens)
        elif option == "closest":
            reflen = min((abs(l-testlen), l) for l in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option

        return reflen

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)
        
    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15 ## so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*n, 'correct':[0]*n}

        # for each sentence
        for comps in self.ctest:            
            testlen = comps['testlen']
            self._testlen += testlen

            if self.special_reflen is None: ## need computation
                reflen = self._single_reflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen
                
            for key in ['guess','correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) \
                        /(float(comps['guess'][k]) + small) 
                bleu_list[k].append(bleu ** (1./(k+1)))
            ratio = (testlen + tiny) / (reflen + small) ## N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1/ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps['reflen'] = self._reflen
        totalcomps['testlen'] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) \
                    / (totalcomps['guess'][k] + small)
            bleus.append(bleu ** (1./(k+1)))
        ratio = (self._testlen + tiny) / (self._reflen + small) ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1/ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)

        self._score = bleus
        return self._score, bleu_list


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=0)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"


# ----------------------- CIDER ------------------------
def cider_precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cider_cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [cider_precook(ref, n) for ref in refs]


def cider_cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return cider_precook(test, n, True)


class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def copy_empty(self):
        new = CiderScorer(df_mode="corpus", n=self.n, sigma=self.sigma)
        new.df_mode = self.df_mode
        new.ref_len = self.ref_len
        new.document_frequency = self.document_frequency
        return new

    def __init__(self, df_mode="corpus", test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.df_mode = df_mode
        self.document_frequency = defaultdict(float)
        self.ref_len = None
        if self.df_mode != "corpus":
            pkl_file = cPickle.load(open(df_mode,'rb'), **(dict(encoding='latin1') if six.PY3 else {}))
            self.ref_len = np.log(float(pkl_file['ref_len']))
            self.document_frequency = pkl_file['document_frequency']
        self.cook_append(test, refs)
    
    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cider_cook_refs(refs))
            if test is not None:
                self.ctest.append(cider_cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        # compute log reference length
        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))
        #elif self.df_mode == "coco-val-df":
            # if coco option selected, use length of coco-val set
        #    self.ref_len = np.log(float(40504))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            # score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        if self.df_mode == "corpus":
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            # assert to check document frequency
            assert(len(self.ctest) >= max(self.document_frequency.values()))
            # import json for now and write the corresponding files
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)


class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, n=4, sigma=6.0, df="corpus"):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        # set which where to compute document frequencies from
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df)

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()
        imgIds = res.keys()
        for id in imgIds:

            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            tmp_cider_scorer += (hypo[0], ref)

        (score, scores) = tmp_cider_scorer.compute_score()

        return score, scores # score * 10, scores * 10

    def method(self):
        return "CIDEr"


# ----------------------- METEOR ------------------------
# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = '../data/meteor-1.5.jar'
# FILE = "./run_captioningMetrics.py"

class Meteor:
  def __init__(self):
    self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
        '-', '-', '-stdio', '-l', 'en', '-norm']
    
    self.meteor_p = subprocess.Popen(self.meteor_cmd, \
        cwd="./",    # os.path.dirname(os.path.abspath(FILE)), \
        stdin=subprocess.PIPE, \
        stdout=subprocess.PIPE, \
        stderr=subprocess.PIPE)
    # Used to guarantee thread safety
    self.lock = threading.Lock()

  def compute_score(self, gts, res, vid_order=None):
    # assert(gts.keys() == res.keys())
    if vid_order is None:
      vid_order = gts.keys()
    scores = []

    eval_line = 'EVAL'
    self.lock.acquire()
    for i in vid_order:
      assert(len(res[i]) == 1)
      stat = self._stat(res[i][0], gts[i])
      eval_line += ' ||| {}'.format(stat)

    self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    self.meteor_p.stdin.flush()
    for i in range(0,len(vid_order)):
      scores.append(float(self.meteor_p.stdout.readline().strip()))
    score = float(self.meteor_p.stdout.readline().strip())
    self.lock.release()

    return score, scores

  def method(self):
    return "METEOR"

  def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
    score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()

  def _score(self, hypothesis_str, reference_list):
    self.lock.acquire()
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
    score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    self.meteor_p.stdin.write('{}\n'.format(score_line))
    self.meteor_p.stdin.flush()
    stats = self.meteor_p.stdout.readline().strip()
    eval_line = 'EVAL ||| {}'.format(stats)
    # EVAL ||| stats 
    self.meteor_p.stdin.write('{}\n'.format(eval_line))
    self.meteor_p.stdin.flush()
    score = float(self.meteor_p.stdout.readline().strip())
    # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
    # thanks for Andrej for pointing this out
    score = float(self.meteor_p.stdout.readline().strip())
    self.lock.release()
    return score
 
  def __exit__(self):
    self.lock.acquire()
    self.meteor_p.stdin.close()
    self.meteor_p.wait()
    self.lock.release()


def producer_fn(q, scorer, gts, res, vid_order):
  _, ss = scorer.compute_score(gts, res, vid_order=vid_order)
  vid_ss = {}
  for vid, s in zip(vid_order, ss):
    vid_ss[vid] = s
  q.put(vid_ss)


class MeteorMulti(object):
  def __init__(self, num_process=4):
    self.num_process = num_process
    self.scorers = []
    for i in xrange(num_process):
      self.scorers.append(Meteor())

  def compute_score(self, gts, res, vid_order=None):
    if vid_order is None:
      vid_order = gts.keys()
    num_vid = len(vid_order)
    num_split = min(self.num_process, num_vid)
    split_idxs = np.linspace(0, num_vid, num_split+1).astype(np.int32)

    q = Queue(num_split)
    producers = []
    for i in xrange(num_split):
      sub_vid_order = vid_order[split_idxs[i]: split_idxs[i+1]]
      sub_gts = {key: gts[key] for key in sub_vid_order}
      sub_res = {key: res[key] for key in sub_vid_order}

      producers.append(Process(target=producer_fn, 
        args=(q, self.scorers[i], sub_gts, sub_res, sub_vid_order)))
      producers[-1].start()

    vid_score = {}
    for i in xrange(num_split):
      sub_vid_ss = q.get()
      vid_score.update(sub_vid_ss)
    scores = [vid_score[vid] for vid in vid_order]

    return np.mean(scores), scores


def second_max_val(arr):
    if len(arr) < 2:
        raise ValueError("Array must have at least two elements")
    
    """ Find the val of the maximum value """
    max_val = np.max(arr)
    
    """ Copy the array to avoid modifying the original array """
    arr_copy = arr.copy()
    
    for idx in range(len(arr_copy)):
        if arr_copy[idx] == max_val:
            """ Set the maximum value to negative infinity """
            arr_copy[idx] = -np.inf
    
    """ Find the value of the new maximum value, which is the second largest """
    second_max_val_ = np.max(arr_copy)
    
    return second_max_val_


SPACY_LANG_MOD = "en_core_web_lg" 
VIEWS = ["ARIA", "1", "2", "3", "4"]   
METRICS = [ "cider", "meteor", "verb_iou", "noun_iou", "nounChunk_iou"]   
PRED_RESULTS_SUBDIR = "egoExo4d_release" 

TK_NM_N_TMSTMP__FILTERER__FP = "captioner_inputData/test_filtered.pkl"

target_tkNmNtmstmps_set = set()
if TK_NM_N_TMSTMP__FILTERER__FP is not None:
    assert ospif(TK_NM_N_TMSTMP__FILTERER__FP), print(TK_NM_N_TMSTMP__FILTERER__FP)
    fltrd_tknmNtmstmp_dct = pkl_ld(TK_NM_N_TMSTMP__FILTERER__FP)
    for k, v in fltrd_tknmNtmstmp_dct.items():
        for k1 in v:
            target_tkNmNtmstmps_set.add((k, k1))

TK_NM__2__TMSTMP__2__LST_ATMC_DSCS___FP = "captioner_inputData/test.pkl"
assert ospif(TK_NM__2__TMSTMP__2__LST_ATMC_DSCS___FP)

PRED_RESULTS_ROOT_DR = f"../../runs" 
assert ospid(PRED_RESULTS_ROOT_DR)

PRED___TK_NM__2__STRT_N_END_TMSTMP__2__SCORES___FP = f"{PRED_RESULTS_ROOT_DR}/{PRED_RESULTS_SUBDIR}/take2startNendTimestamp2predScores_checkpoint-maxCaptioniningScore.pkl" 
assert ospif(PRED___TK_NM__2__STRT_N_END_TMSTMP__2__SCORES___FP), print(PRED___TK_NM__2__STRT_N_END_TMSTMP__2__SCORES___FP)

VRB2LBL_FP = f"../../data/captioningEval_miscData/verb2label.pkl"
assert ospif(VRB2LBL_FP)

NN2LBL_FP = f"../../data/captioningEval_miscData/noun2label.pkl"
assert ospif(NN2LBL_FP)

NOUNS_WHITELIST_CSV_PATH = f"../../data/captioningEval_miscData/compound_noun_whitelist.csv"
assert ospif(NOUNS_WHITELIST_CSV_PATH)

NOUNS_SENSE_DICT_CSV_PATH = f"../../data/captioningEval_miscData/noun_sense_list.csv"
assert ospif(NOUNS_WHITELIST_CSV_PATH)

VERBS_SENSE_DICT_CSV_PATH = f"../../data/captioningEval_miscData/verb_sense_list.csv"
assert ospif(NOUNS_WHITELIST_CSV_PATH)

SCORES_FP = "captioner_outputData/scores.pkl"
assert ospif(SCORES_FP), print(SCORES_FP)

tkNm2tmstmp2lstAtmcDscs =\
    pkl_ld(TK_NM__2__TMSTMP__2__LST_ATMC_DSCS___FP)

nlp = spacy.load(SPACY_LANG_MOD)
all_stopwords = nlp.Defaults.stop_words

dct_nn2lbl = pkl_ld(NN2LBL_FP)
dct_vrb2lbl = pkl_ld(VRB2LBL_FP)

noun_whitelist = pd.read_csv(NOUNS_WHITELIST_CSV_PATH, header=None).values
verb_sense_dict = load_sense_dict(VERBS_SENSE_DICT_CSV_PATH)
noun_sense_dict = load_sense_dict(NOUNS_SENSE_DICT_CSV_PATH)

print("LOADING SCORERS")
cdr_scrr = Cider()
bl_scrr = Bleu(4)
mtr_scrr = evaluate.load('meteor')
print("DONE LOADING SCORERS")
print("-" * 50)
print("METRICS: ", METRICS)

gt_tkNm_2_tmstmpNtxt = {}
tkNm_2_cdrPrStpPrVw = {}
tkNm_2_mtrPrStpPrVw = {}
tkNm_2_vrbIouPrStpPrVw = {}
tkNm_2_nnIouPrStpPrVw = {}
tkNm_2_nnChnkIouPrStpPrVw = {}
mtrc_2_tkNm_2_vlPrStpPrVw = {}
vw_2_tkNm_2_tmstmpNtxt = {}

for vw_idx, vw in enumerate(VIEWS):
    np.random.seed(42)

    vwRslts_fp = f"captioner_outputData/{vw}.json"
    assert ospif(vwRslts_fp), print(vwRslts_fp)

    tkNm_2_tmstmpNtxt = json_ld(vwRslts_fp)

    if len(target_tkNmNtmstmps_set) > 0:
        tkNm_2_tmstmpNtxt_tmp = {}

        tmp_dct = pkl_ld(SCORES_FP)

        for tk_nm in tkNm_2_tmstmpNtxt:
            assert tk_nm in tmp_dct
            assert len(tmp_dct[tk_nm]) == len(tkNm_2_tmstmpNtxt[tk_nm])
            assert tk_nm not in tkNm_2_tmstmpNtxt_tmp
            tkNm_2_tmstmpNtxt_tmp[tk_nm] = []
            tmp_cnt = 0
            for k in tmp_dct[tk_nm]:
                assert float(k[0]) == float(tkNm_2_tmstmpNtxt[tk_nm][tmp_cnt][0]), print(float(k[0]), )
                tmp_lst = [tkNm_2_tmstmpNtxt[tk_nm][tmp_cnt][0], tkNm_2_tmstmpNtxt[tk_nm][tmp_cnt][1], float(k[1]), float(k[2])]
                tkNm_2_tmstmpNtxt_tmp[tk_nm].append(tmp_lst)
                tmp_cnt += 1
        tkNm_2_tmstmpNtxt = tkNm_2_tmstmpNtxt_tmp
        tkNm_2_tmstmpNtxt_tmp2 = {}
        tkNm2tmstmp2lstAtmcDscs_tmp = {}
        for tk_nm, tmp_tmstmpNtxts in tkNm_2_tmstmpNtxt.items():
            for tmp_ele in tmp_tmstmpNtxts:
                if (tk_nm, (tmp_ele[0], tmp_ele[2], tmp_ele[3])) in target_tkNmNtmstmps_set:
                    if tk_nm not in tkNm_2_tmstmpNtxt_tmp2:
                        tkNm_2_tmstmpNtxt_tmp2[tk_nm] = []
                    tkNm_2_tmstmpNtxt_tmp2[tk_nm].append(tmp_ele)

                    assert tk_nm in  tkNm2tmstmp2lstAtmcDscs
                    assert (tmp_ele[2], tmp_ele[3]) in tkNm2tmstmp2lstAtmcDscs[tk_nm]
                    if tk_nm not in tkNm2tmstmp2lstAtmcDscs_tmp:
                        tkNm2tmstmp2lstAtmcDscs_tmp[tk_nm] = {}
                    tkNm2tmstmp2lstAtmcDscs_tmp[tk_nm][(tmp_ele[2], tmp_ele[3])] = tkNm2tmstmp2lstAtmcDscs[tk_nm][(tmp_ele[2], tmp_ele[3])]
        tkNm_2_tmstmpNtxt = tkNm_2_tmstmpNtxt_tmp2
        tkNm2tmstmp2lstAtmcDscs = tkNm2tmstmp2lstAtmcDscs_tmp

    assert vw not in vw_2_tkNm_2_tmstmpNtxt
    vw_2_tkNm_2_tmstmpNtxt[vw] = tkNm_2_tmstmpNtxt   

    tkNm2strtNendTmstmp2bstPrdScrPrVw = pkl_ld(PRED___TK_NM__2__STRT_N_END_TMSTMP__2__SCORES___FP)

    gt_tkNm_2_tmstmpNtxt = {}
    for tk_nm, tmstmpNtxt in tqdm(tkNm_2_tmstmpNtxt.items()):
        assert tk_nm not in gt_tkNm_2_tmstmpNtxt
        gt_tkNm_2_tmstmpNtxt[tk_nm] = []

        assert tk_nm in tkNm2tmstmp2lstAtmcDscs
        if tk_nm not in tkNm2strtNendTmstmp2bstPrdScrPrVw:
            if tk_nm in gt_tkNm_2_tmstmpNtxt:
                del gt_tkNm_2_tmstmpNtxt[tk_nm]
                continue

        if len(target_tkNmNtmstmps_set) > 0:
            tkNm2strtNendTmstmp2bstPrdScrPrVw_tmp = {}
            for tmp_tk_nm in tkNm2tmstmp2lstAtmcDscs:
                for tmp_tmpstmp in tkNm2tmstmp2lstAtmcDscs[tmp_tk_nm]:
                    assert tmp_tk_nm in tkNm2strtNendTmstmp2bstPrdScrPrVw
                    assert tmp_tmpstmp in tkNm2strtNendTmstmp2bstPrdScrPrVw[tmp_tk_nm]
                    if tmp_tk_nm not in tkNm2strtNendTmstmp2bstPrdScrPrVw_tmp:
                        tkNm2strtNendTmstmp2bstPrdScrPrVw_tmp[tmp_tk_nm] = {}
                    tkNm2strtNendTmstmp2bstPrdScrPrVw_tmp[tmp_tk_nm][tmp_tmpstmp] =\
                        tkNm2strtNendTmstmp2bstPrdScrPrVw[tmp_tk_nm][tmp_tmpstmp]

            tkNm2strtNendTmstmp2bstPrdScrPrVw = tkNm2strtNendTmstmp2bstPrdScrPrVw_tmp
        else:
            assert len(tmstmpNtxt) == len(tkNm2strtNendTmstmp2bstPrdScrPrVw[tk_nm]),\
                print(tk_nm, len(tmstmpNtxt), len(tkNm2strtNendTmstmp2bstPrdScrPrVw[tk_nm]))

        for i in range(len(tmstmpNtxt)):
            ele_tmstmpNtxt = tmstmpNtxt[i]

            ky = list(tkNm2tmstmp2lstAtmcDscs[tk_nm].keys())[i]
            ky1 = tkNm2tmstmp2lstAtmcDscs[tk_nm][ky]['timestamp']

            if (len(ele_tmstmpNtxt) > 2) and (len(ele_tmstmpNtxt) != 4):
                assert tkNm2tmstmp2lstAtmcDscs[tk_nm][ky]["startNend_clipName"][0] == ele_tmstmpNtxt[3][0]
                assert tkNm2tmstmp2lstAtmcDscs[tk_nm][ky]["startNend_clipName"][1] == ele_tmstmpNtxt[3][1]

                assert tkNm2tmstmp2lstAtmcDscs[tk_nm][ky]["startNend_frameIdx"][0] == ele_tmstmpNtxt[4][0]
                assert tkNm2tmstmp2lstAtmcDscs[tk_nm][ky]["startNend_frameIdx"][1] == ele_tmstmpNtxt[4][1]

                assert ky[0] == ele_tmstmpNtxt[5][0]
                assert ky[1] == ele_tmstmpNtxt[5][1]

            gt_tkNm_2_tmstmpNtxt[tk_nm].append([ky1, []])
            for ele2 in tkNm2tmstmp2lstAtmcDscs[tk_nm][ky]['text']:
                assert isinstance(ele2, str)
                gt_tkNm_2_tmstmpNtxt[tk_nm][-1][-1].append(ele2)
            gt_tkNm_2_tmstmpNtxt[tk_nm][-1].append(ky)

    tk_cnt = 0
    for tk_nm, tmstmpNtxt2 in tqdm(tkNm_2_tmstmpNtxt.items()):
        if tk_nm not in gt_tkNm_2_tmstmpNtxt:
            continue
        assert len(tmstmpNtxt2) == len(gt_tkNm_2_tmstmpNtxt[tk_nm])

        for dummyEle_idx in range(len(gt_tkNm_2_tmstmpNtxt[tk_nm]) - 1):
            if gt_tkNm_2_tmstmpNtxt[tk_nm][dummyEle_idx][0] > gt_tkNm_2_tmstmpNtxt[tk_nm][dummyEle_idx + 1][0]:
                raise ValueError
        srtd_gt_tmstmpNtxt = gt_tkNm_2_tmstmpNtxt[tk_nm]    

        for dummyEle_idx in range(len(tmstmpNtxt2) - 1):
            if tmstmpNtxt2[dummyEle_idx][0] > tmstmpNtxt2[dummyEle_idx + 1][0]:
                raise ValueError
        srtd_tmstmpNtxt = tmstmpNtxt2   

        cdrNbl_prds = {}
        cdrNbl_rfs = {}
        mtr_prds = []
        mtr_rfs = []

        indvdl_mtrs = []
        indvdl_vrb_ious = []
        indvdl_nn_ious = []
        indvdl_nnChnk_ious = []
        bstExAnntn_cnt = 0
        thsVw_ntBstEx_idxs = set()
        for eleTmstmpNtxt_idx, ele_tmstmpNtxt2 in enumerate(srtd_gt_tmstmpNtxt):
            assert eleTmstmpNtxt_idx < len(tkNm2strtNendTmstmp2bstPrdScrPrVw[tk_nm]), print(len(tkNm2strtNendTmstmp2bstPrdScrPrVw[tk_nm]),
                                                                                            eleTmstmpNtxt_idx)
            if list(tkNm2strtNendTmstmp2bstPrdScrPrVw[tk_nm].values())[eleTmstmpNtxt_idx][np.argmax(list(tkNm2strtNendTmstmp2bstPrdScrPrVw[tk_nm].values())[eleTmstmpNtxt_idx])] !=\
                    list(tkNm2strtNendTmstmp2bstPrdScrPrVw[tk_nm].values())[eleTmstmpNtxt_idx][vw_idx]:
                thsVw_ntBstEx_idxs.add(bstExAnntn_cnt)
            
            cdrNbl_prds[eleTmstmpNtxt_idx] = [srtd_tmstmpNtxt[eleTmstmpNtxt_idx][1]]
            cdrNbl_rfs[eleTmstmpNtxt_idx] = srtd_gt_tmstmpNtxt[eleTmstmpNtxt_idx][1]

            mtr_prds.append(srtd_tmstmpNtxt[eleTmstmpNtxt_idx][1])
            mtr_rfs.append(srtd_gt_tmstmpNtxt[eleTmstmpNtxt_idx][1])

            mtrIndv = mtr_scrr.compute(references=mtr_rfs[-1:], predictions=mtr_prds[-1:])['meteor']

            indvdl_mtrs.append(mtrIndv)    

            assert isinstance(srtd_tmstmpNtxt[eleTmstmpNtxt_idx][1], str)
            unq_vrbs, unq_nns, unq_nnChnks, _, _, _ =\
                extract_verbNnoun_v1(srtd_tmstmpNtxt[eleTmstmpNtxt_idx][1],
                                     nlp,
                                     all_stopwords,
                                     dct_nn2lbl=dct_nn2lbl,
                                     dct_vrb2lbl=dct_vrb2lbl,
                                     compound_whitelist=noun_whitelist,
                                     verb_sense_dict=verb_sense_dict,
                                     noun_sense_dict=noun_sense_dict,)

            lst_vrb_iou = []
            lst_nn_iou = []
            lst_nnChnk_iou = []
            for gt_txt in srtd_gt_tmstmpNtxt[eleTmstmpNtxt_idx][1]:
                assert isinstance(gt_txt, str)
                gt_unq_vrbs, gt_unq_nns, gt_unq_nnChnks, _, _, _ =\
                    extract_verbNnoun_v1(gt_txt,
                                         nlp,
                                         all_stopwords,
                                         dct_nn2lbl=dct_nn2lbl,
                                         dct_vrb2lbl=dct_vrb2lbl,
                                         compound_whitelist=noun_whitelist,
                                         verb_sense_dict=verb_sense_dict,
                                         noun_sense_dict=noun_sense_dict,)

                assert isinstance(unq_vrbs, set)
                if len(unq_vrbs) > 0:
                    assert isinstance(list(unq_vrbs)[0], str)

                assert isinstance(unq_nns, set)
                if len(unq_nns) > 0:
                    assert isinstance(list(unq_nns)[0], str)

                assert isinstance(unq_nnChnks, set)
                if len(unq_nnChnks) > 0:
                    assert isinstance(list(unq_nnChnks)[0], str)

                assert isinstance(gt_unq_vrbs, set)
                if len(gt_unq_vrbs) > 0:
                    assert isinstance(list(gt_unq_vrbs)[0], str)

                assert isinstance(gt_unq_nns, set)
                if len(gt_unq_nns) > 0:
                    assert isinstance(list(gt_unq_nns)[0], str)

                assert isinstance(gt_unq_nnChnks, set)
                if len(gt_unq_nnChnks) > 0:
                    assert isinstance(list(gt_unq_nnChnks)[0], str)

                vrb_lnIntrsctn = len(unq_vrbs.intersection(gt_unq_vrbs))
                vrb_lnUnn = max(len(unq_vrbs.union(gt_unq_vrbs)), 1)
                lst_vrb_iou.append(vrb_lnIntrsctn / vrb_lnUnn)

                nn_lnIntrsctn = len(unq_nns.intersection(gt_unq_nns))
                nn_lnUnn = max(len(unq_nns.union(gt_unq_nns)), 1)
                lst_nn_iou.append(nn_lnIntrsctn / nn_lnUnn)

                nnChnk_lnIntrsctn = len(unq_nnChnks.intersection(gt_unq_nnChnks))
                nnChnk_lnUnn = max(len(unq_nnChnks.union(gt_unq_nnChnks)), 1)
                lst_nnChnk_iou.append(nnChnk_lnIntrsctn / nnChnk_lnUnn)

            indvdl_vrb_ious.append(np.max(lst_vrb_iou))
            indvdl_nn_ious.append(np.max(lst_nn_iou))
            indvdl_nnChnk_ious.append(np.max(lst_nnChnk_iou))

        cdr, cdr_al =  cdr_scrr.compute_score(cdrNbl_rfs, cdrNbl_prds)
        mtr_al = indvdl_mtrs

        assert len(indvdl_mtrs) == len(cdr_al) == len(indvdl_vrb_ious) ==\
                len(indvdl_nn_ious) == len(indvdl_nnChnk_ious) 
        
        if tk_nm not in tkNm_2_cdrPrStpPrVw:
            tkNm_2_cdrPrStpPrVw[tk_nm] = []
        for cdr_idx, cdr_vl in enumerate(cdr_al):
            if cdr_idx >= len(tkNm_2_cdrPrStpPrVw[tk_nm]):
                tkNm_2_cdrPrStpPrVw[tk_nm].append([])
            if cdr_idx in thsVw_ntBstEx_idxs:
                tkNm_2_cdrPrStpPrVw[tk_nm][cdr_idx].append(float('-inf'))
            else:
                tkNm_2_cdrPrStpPrVw[tk_nm][cdr_idx].append(cdr_vl)

        if tk_nm not in tkNm_2_mtrPrStpPrVw:
            tkNm_2_mtrPrStpPrVw[tk_nm] = []
        for mtr_idx, mtr_vl in enumerate(mtr_al):
            if mtr_idx >= len(tkNm_2_mtrPrStpPrVw[tk_nm]):
                tkNm_2_mtrPrStpPrVw[tk_nm].append([])

            if mtr_idx in thsVw_ntBstEx_idxs:
                tkNm_2_mtrPrStpPrVw[tk_nm][mtr_idx].append(float('-inf'))
            else:
                assert mtr_vl != float('-inf')
                tkNm_2_mtrPrStpPrVw[tk_nm][mtr_idx].append(mtr_vl)

        if tk_nm not in tkNm_2_vrbIouPrStpPrVw:
            tkNm_2_vrbIouPrStpPrVw[tk_nm] = []

            assert tk_nm not in tkNm_2_nnIouPrStpPrVw
            tkNm_2_nnIouPrStpPrVw[tk_nm] = []

            assert tk_nm not in tkNm_2_nnChnkIouPrStpPrVw
            tkNm_2_nnChnkIouPrStpPrVw[tk_nm] = []

        for vrbIou_idx, vrbIou_vl in enumerate(indvdl_vrb_ious):
            if vrbIou_idx >= len(tkNm_2_vrbIouPrStpPrVw[tk_nm]):
                tkNm_2_vrbIouPrStpPrVw[tk_nm].append([])

            if vrbIou_idx in thsVw_ntBstEx_idxs:
                tkNm_2_vrbIouPrStpPrVw[tk_nm][vrbIou_idx].append(float('-inf'))
            else:
                tkNm_2_vrbIouPrStpPrVw[tk_nm][vrbIou_idx].append(float(vrbIou_vl))

        for nnIou_idx, nnIou_vl in enumerate(indvdl_nn_ious):
            if nnIou_idx >= len(tkNm_2_nnIouPrStpPrVw[tk_nm]):
                tkNm_2_nnIouPrStpPrVw[tk_nm].append([])

            if nnIou_idx in thsVw_ntBstEx_idxs:
                tkNm_2_nnIouPrStpPrVw[tk_nm][nnIou_idx].append(float('-inf'))
            else:
                tkNm_2_nnIouPrStpPrVw[tk_nm][nnIou_idx].append(float(nnIou_vl))

        for nnChnkIou_idx, nnChnkIou_vl in enumerate(indvdl_nnChnk_ious):
            if nnChnkIou_idx >= len(tkNm_2_nnChnkIouPrStpPrVw[tk_nm]):
                tkNm_2_nnChnkIouPrStpPrVw[tk_nm].append([])

            if nnChnkIou_idx in thsVw_ntBstEx_idxs:
                tkNm_2_nnChnkIouPrStpPrVw[tk_nm][nnChnkIou_idx].append(float('-inf'))
            else:
                assert float(nnChnkIou_vl) != float('-inf')
                tkNm_2_nnChnkIouPrStpPrVw[tk_nm][nnChnkIou_idx].append(float(nnChnkIou_vl))

        tk_cnt +=1 
        # if tk_cnt == 2:    # 1, 2, 5, 15
        #     break

mtrc_2_tkNm_2_vlPrStpPrVw["cider"] = tkNm_2_cdrPrStpPrVw
mtrc_2_tkNm_2_vlPrStpPrVw["meteor"] = tkNm_2_mtrPrStpPrVw
mtrc_2_tkNm_2_vlPrStpPrVw["verb_iou"] = tkNm_2_vrbIouPrStpPrVw
mtrc_2_tkNm_2_vlPrStpPrVw["noun_iou"] = tkNm_2_nnIouPrStpPrVw
mtrc_2_tkNm_2_vlPrStpPrVw["nounChunk_iou"] = tkNm_2_nnChnkIouPrStpPrVw

for mtrc_nm in METRICS:
    np.random.seed(42)
    assert mtrc_nm in ["cider", "meteor", "verb_iou", "noun_iou", "nounChunk_iou",]

    assert mtrc_nm in mtrc_2_tkNm_2_vlPrStpPrVw
    tkNm_2_vlPrStpPrVw = mtrc_2_tkNm_2_vlPrStpPrVw[mtrc_nm]

    lst_mtrc_bstVwCnt = []
    lst_mtrc_bstVwPrct = []
    mtrc_bst = []
    mtrc_dlBstN2ndBstVw_abs = []
    mtrc_dlBstN2ndBstVw_rlMxMtrcCrrStp = []
    lst_bstMtrcNtkNmNstpIdxNbstVwNprdTxtNgtTxtNscndBstMtrcNscndBstVwNscndBstPrdTxt = []
    for tk_nm, mtrcPrStpPrVw in tkNm_2_vlPrStpPrVw.items():
        lst_mtrc_bstVwCnt.append([0] * len(VIEWS))
        mtrc_bst.append([])
        mtrc_dlBstN2ndBstVw_abs.append([])
        mtrc_dlBstN2ndBstVw_rlMxMtrcCrrStp.append([])
        
        for stp_idx, mtrc_prVw in enumerate(mtrcPrStpPrVw):
            srtd_mtrc_prVw_idxs = np.argsort(mtrc_prVw)[::-1]
            srtd_mtrc_prVw = np.array(mtrc_prVw)[srtd_mtrc_prVw_idxs]

            bstVw_idxs = []
            scndBstVw_idx = None
            for vw_idx2 in srtd_mtrc_prVw_idxs:
                if mtrc_prVw[vw_idx2] != float('-inf'):
                    bstVw_idxs.append(vw_idx2)

            """ old code """
            # assert len(bstVw_idxs) > 0,\
            #     print(len(bstVw_idxs), tk_nm, mtrc_nm, stp_idx, mtrc_prVw, mtrcPrStpPrVw)
            """ new code """
            if len(bstVw_idxs) == 0:
                bstVw_idxs = list(range(len(VIEWS)))

            if scndBstVw_idx is None:
                scndBstVw_idx = bstVw_idxs[0]

            bstVw_idx = np.random.choice(bstVw_idxs)

            lst_mtrc_bstVwCnt[-1][bstVw_idx] += 1


            mtrc_bst[-1].append(srtd_mtrc_prVw[0])
            
            mtrc_dlBstN2ndBstVw_abs[-1].append(mtrc_prVw[bstVw_idx] - mtrc_prVw[scndBstVw_idx])
            mtrc_dlBstN2ndBstVw_rlMxMtrcCrrStp[-1].append((mtrc_prVw[bstVw_idx] - mtrc_prVw[scndBstVw_idx])/ max(mtrc_prVw[scndBstVw_idx], 1e-19) * 100)

            assert VIEWS[bstVw_idx] in vw_2_tkNm_2_tmstmpNtxt
            assert tk_nm in vw_2_tkNm_2_tmstmpNtxt[VIEWS[bstVw_idx]]
            assert stp_idx < len(vw_2_tkNm_2_tmstmpNtxt[VIEWS[bstVw_idx]][tk_nm])
            assert isinstance(vw_2_tkNm_2_tmstmpNtxt[VIEWS[bstVw_idx]][tk_nm][stp_idx][1], str) 

            assert tk_nm in gt_tkNm_2_tmstmpNtxt
            assert stp_idx < len(gt_tkNm_2_tmstmpNtxt[tk_nm])
            
            assert scndBstVw_idx is not None, print(scndBstVw_idx)
            lst_bstMtrcNtkNmNstpIdxNbstVwNprdTxtNgtTxtNscndBstMtrcNscndBstVwNscndBstPrdTxt.append(
                (
                 srtd_mtrc_prVw[0], 
                 tk_nm,
                 )
            )

        lst_mtrc_bstVwPrct.append((np.array(lst_mtrc_bstVwCnt[-1]) / max(np.sum(lst_mtrc_bstVwCnt[-1]), 1e-19)) * 100)

    dmp_dr_fp = f"{PRED_RESULTS_ROOT_DR}/{PRED_RESULTS_SUBDIR}/captioningMetrics_files" 
    if not ospid(dmp_dr_fp):
        os.makedirs(dmp_dr_fp)

    json_dmp(lst_bstMtrcNtkNmNstpIdxNbstVwNprdTxtNgtTxtNscndBstMtrcNscndBstVwNscndBstPrdTxt, 
             f"{dmp_dr_fp}/{mtrc_nm}_outputs.json") 
