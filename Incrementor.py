"""This is a fork program of countwords.py, forked on 2019-03-30_21:11

Usage:
  Incrementor.py [--help] 
  Incrementor.py <text_source> <out_file> [--grow_fly [new] <config>] [options]

Options:
  -h --help          show this screen
  -t --tokenize      run a simple tokenizer over the input text
  -l --linewise      don't count cooccurrences across lines
  -i --increment     load the existing space in <out_file> and extend it
  -v --verbose       comment program status with command-line output
  --grow_fly         develop a fruitfly alongside the space; either from scratch or from a config
  -d=<dims>          limit space to a number of dimensions 
  -w=<window>        number of tokens in the context (to each side) [default: 5]
  -x=<checkfile>     check whether all words of <checkfile> are in <text_source>
  
OBACHT!
  # File extensions: use them for <text_source>, <ff_config>, <checkfile>,
                     DON'T use them for <out_file>!
"""

import re
from typing import Dict, Any

import numpy as np
from docopt import docopt
import nltk
from tqdm import tqdm
from os import walk

import Fruitfly
import utils
from Fruitfly import Fruitfly


class Incrementor:

    def __init__(self, corpus_dir, matrix_file,
                 corpus_tokenize=False, corpus_linewise=False, corpus_checkvoc=None,
                 matrix_incremental=True, matrix_maxdims=None, contentwords_only=False,
                 fly_new=False, fly_grow=False, fly_file=None, fly_max_pn=None,
                 verbose=False):

        self.verbose = verbose

        self.corpus_dir   = corpus_dir
        self.is_tokenize  = corpus_tokenize
        self.is_linewise  = corpus_linewise
        self.required_voc = corpus_checkvoc

        self.outspace = matrix_file+".dm" # e.g. "data/potato"
        self.outcols  = matrix_file+".cols"
        self.is_incremental = matrix_incremental
        self.max_dims = matrix_maxdims
        self.postag_simple = contentwords_only

        self.is_new_fly  = fly_new
        self.is_grow_fly = fly_grow
        self.flyfile     = fly_file
        self.fly_max_pn  = fly_max_pn


        self.words = self.read_corpus(self.corpus_dir,
                                      tokenize_corpus=self.is_tokenize,
                                      postag_simple=self.postag_simple,
                                      linewise=self.is_linewise,
                                      verbose=self.verbose)

        self.cooc, self.words_to_i, self.i_to_words, self.fruitfly = \
            self.read_incremental_parts(self.outspace,
                                        self.outcols,
                                        self.flyfile,
                                        verbose=self.verbose)

        # words that will be counted (= labels of the final matrix dimensions)
        self.freq = self.freq_dist(self.words,
                                   size_limit=self.max_dims,
                                   required_words=self.required_voc,
                                   verbose=self.verbose)

        if self.verbose: print("\tVocabulary size:",len(self.freq),
                               "\n\tTokens (or lines) for cooccurrence count:",len(self.words))


    #========== FILE READING

    @staticmethod
    def read_corpus(indir, tokenize_corpus=False, postag_simple=False, linewise=False, verbose=False):
        """
        :param indir: a directory or a single file as text resource
        :param tokenize_corpus:
        :param postag_simple:
        :param linewise:
        :param verbose:
        :return:
        """
        if verbose: print("\nreading corpus from",indir,"...")
        lines = [] # list of lists of words
        nonword = re.compile("\W+(_X)?") # to delete punctuation entries
        lc = 0 # for files with more than one line
        wc = 0 # wordcount

        filepaths = []
        if os.path.isfile(indir):
            filepaths = [indir]
        else:
            for (dirpath, dirnames, filenames) in walk(indir):
                filepaths.extend([dirpath+"/"+f for f in filenames])

        for file in filepaths:
            with open(file) as f:
                print("reading",file,"...")
                for line in f:
                    lc += 1
                    line = line.rstrip().lower()
                    if tokenize_corpus:
                        tokens = nltk.word_tokenize(line) # CLEANUP tokenizer.tokenize(line)
                    else:
                        tokens = line.split()
                    linewords = []
                    for t in tokens:
                        if postag_simple:
                            t = t[:-1]+t[-1].upper()
                        if (re.fullmatch(nonword, t) is None): # ignores punctuation
                            linewords.append(t) # adds the list as a unit to 'lines'
                        wc+=1
                        if verbose and wc%1000000 == 0:
                            print("\twords read:",wc/1000000,"million",end="\r")
                    lines.append(linewords)


        if linewise is False:
            return [w for l in lines for w in l] # flattens to a simple word list
        else:
            return(lines)

    def read_incremental_parts(self, outspace, outcols, flyfile, verbose=False): # matrix, vocabulary, fruitfly (if wanted)
        """
        Return a matrix, a vocabulary, and a Fruitfly object.
        The matrix can be newly instantiated or taken from an already 
        existing space; the vocabulary aswell.
        The fruitfly can be optionally created alongside, also either new 
        or from an already existing config file.
        """
        if self.is_incremental:
            if verbose: print("\nloading existing space...")
            unhashed_space = utils.readDM(outspace) # returns dict of word : vector
            i_to_words, words_to_i = utils.readCols(outcols)
            dimensions = sorted(words_to_i, key=words_to_i.get)
            cooc = np.stack(tuple([unhashed_space[w] for w in dimensions]))
        else:
            cooc = np.array([[]]) # cooccurrence count (only numbers)
            words_to_i = {} # vocabulary and word positions
            i_to_words = {}

        if self.is_grow_fly:
            if self.is_new_fly:
                if verbose: print("\ncreating new fruitfly...")
                fruitfly = Fruitfly.from_scratch(max_pn_size=self.fly_max_pn) # default config: (50,30000,6,5)
            else:
                if verbose: print("\nloading fruitfly...")
                fruitfly = Fruitfly.from_config(flyfile)
                self.fly_max_pn = fruitfly.max_pn_size # update the attribute
        else:
            fruitfly = None

        return cooc, words_to_i, i_to_words, fruitfly

    def freq_dist(self, wordlist, size_limit=None, required_words=None, verbose=False):
        """
        This method is used to limit the dimensionality of the count matrix, which speeds up processing.
        The obtained dictionary is used as vocabulary reference of the current corpus at several processing steps.
        For true incrementality, size_limit is None and the dictionary is computed over the currently available corpus.
        If size_limit is None, required_words has no effect on the obtained dictionary.
        :param wordlist: list of (word) tokens from the text resource
        :param size_limit: maximum length of the returned frequency distribution
        :param required_words: file path to a list with prioritized words (regardless of their frequencies)
        :param verbose: comment on workings via print statements
        :return: dict[str:int]
        """
        if verbose: print("\ncreating frequency distribution...")
        freq = {}
        #TODO make this code better. There are too many ifs and loops.
        if self.is_linewise:
            for line in tqdm(wordlist):
                for w in line:
                    if self.postag_simple and w.endswith(("_N", "_V", "_J")): # leaves out all non-content words
                        if w in freq:
                            freq[w] += 1
                        else:
                            freq[w] = 1
                    else:
                        if w in freq:
                            freq[w] += 1
                        else:
                            freq[w] = 1
        else:
            for w in tqdm(wordlist):
                if self.postag_simple and w.endswith(("_N", "_V", "_J")):
                    if w in freq:
                        freq[w] += 1
                    else:
                        freq[w] = 1
                else:
                    if w in freq:
                        freq[w] += 1
                    else:
                        freq[w] = 1

        frequency_sorted = sorted(freq, key=freq.get, reverse=True) # list of all words

        if required_words is not None:
            checklist = self.read_checklist(checklist_filepath=required_words, with_pos_tags=self.postag_simple)
            overlap = list(set(checklist).intersection(set(frequency_sorted)))
            rest_words = [w for w in frequency_sorted if w not in overlap] # words that are not required; sorted by frequency
            returnlist = overlap+rest_words
        else:
            returnlist = frequency_sorted

        if(size_limit is not None and size_limit <= len(freq)):
            return {k:freq[k] for k in returnlist[:size_limit]}
        else:
            return freq

    @staticmethod
    def read_checklist(checklist_filepath, with_pos_tags=False):
        if checklist_filepath is None:
            return []

        checklist = []

        with open(checklist_filepath, "r") as f:
            #CLEANUP
            """
            This has been generalized by making vocabulary lists out of 
            the paired lists.
            paired_lists = ["data/MEN_dataset_natural_form_full",
                            "./data/MEN_dataset_natural_form_full",
                            "incrementality_sandbox/data/sandbox_MEN_pairs",
                            "pipe/testset_MEN_pairs"]
            if checklist_filepath in paired_lists: 
                for line in f:
                    words = line.rstrip().split()[:2]
                    checklist.extend(words)
            else:
            """
            for word in f:
                word = word.rstrip()
                checklist.append(word)
                
        if with_pos_tags is False:
            pos_tag = re.compile("_.+?") # get rid of simple POS-tags
            return [re.sub(pos_tag, "", w) for w in checklist]
        else: 
            return checklist

    def check_overlap(self, checklist_filepath=None, wordlist=None, verbose=False):
        if checklist_filepath is None: checklist_filepath = self.required_voc
        if wordlist is None: wordlist = self.freq.keys()
        checklist = self.read_checklist(checklist_filepath, with_pos_tags=self.postag_simple)
        if len(checklist) == 0: # if no checking is specified, go on without checking
            if verbose:
                print("\tcheck_overlap(): nothing to check.")
            return True, []

        unshared_words = list(set(checklist).difference(set(wordlist)))

        if verbose:
            if len(unshared_words) == 0:
                print("\tComplete overlap with",checklist_filepath)
            else:
                print("\tChecked for overlap with",checklist_filepath,
                      "\n\twords missing in the corpus:",len(unshared_words),
                      "\n\texamples:",unshared_words[:10])

        return (unshared_words is True), unshared_words



    #========== CO-OCCURRENCE COUNTING

    def extend_incremental_parts_if_necessary(self, w): # matrix dimensions and fruitfly PN layer
        #global cooc, words_to_i #CLEANUP
        if w not in self.words_to_i:
            self.words_to_i[w] = len(self.words_to_i) # extend the vocabulary
            self.i_to_words[self.words_to_i[w]] = w # extend the 'backwards vocabulary'
            temp = np.zeros((len(self.words_to_i), len(self.words_to_i))) # make bigger matrix
            temp[0:self.cooc.shape[0], 0:self.cooc.shape[1]] = self.cooc # paste current matrix into the new one
            self.cooc = temp
        if self.fruitfly is not None and len(self.words_to_i) > self.fruitfly.pn_size:
            self.fruitfly.extend() # extend if needed (incl. "catching up" with vocabulary size)

    def count_start_of_text(self, words, window): # for the first couple of words
        #global cooc, words_to_i #CLEANUP
        for i in range(window):
            if words[i] in self.freq:
                for c in range(i+window+1): # iterate over the context
                    if words[c] in self.freq:
                        self.extend_incremental_parts_if_necessary(words[i])
                        self.extend_incremental_parts_if_necessary(words[c])
                        self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[c]]] += 1
                self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[i]]]-=1 # delete "self-occurrence"

    def count_middle_of_text(self, words, window): # for most of the words
        #global cooc, words_to_i #CLEANUP
        if self.is_linewise: # this loop is without tqdm, the other loop with.
            for i in range(window, len(words)-window):
                if words[i] in self.freq:
                    for c in range(i-window, i+window+1): # iterate over the context
                        if words[c] in self.freq:
                            self.extend_incremental_parts_if_necessary(words[i])
                            self.extend_incremental_parts_if_necessary(words[c])
                            self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[c]]] += 1
                    self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[i]]]-=1 # delete "self-occurrence"
        else:
            for i in tqdm(range(window, len(words)-window)):
                if words[i] in self.freq:
                    for c in range(i-window, i+window+1): # iterate over the context
                        if words[c] in self.freq:
                            self.extend_incremental_parts_if_necessary(words[i])
                            self.extend_incremental_parts_if_necessary(words[c])
                            self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[c]]] += 1
                    self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[i]]]-=1 # delete "self-occurrence"

    def count_end_of_text(self, words, window): # for the last couple of words
        #global cooc, words_to_i #CLEANUP
        for i in range(len(words)-window, len(words)):
            if words[i] in self.freq:
                for c in range(i-window, len(words)): # iterate over the context
                    if words[c] in self.freq:
                        self.extend_incremental_parts_if_necessary(words[i])
                        self.extend_incremental_parts_if_necessary(words[c])
                        self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[c]]] += 1
                self.cooc[self.words_to_i[words[i]]][self.words_to_i[words[i]]]-=1 # delete "self-occurrence"

    def count_cooccurrences(self, words=None, window=5):
        if words is None: words = self.words # to allow partial counting
        if self.verbose: print("\ncounting inner cooccurrences within",window,"words distance...")
        #if self.postag_simple: #CLEANUP
        #    words = simplify_postags(words)
        if self.is_linewise:
            for line in tqdm(words):
                if len(line) >= 2*window: # to avoid index errors
                    self.count_start_of_text(line, window)
                    self.count_middle_of_text(line, window)
                    self.count_end_of_text(line, window)
                else:
                    if self.verbose: print("\tline too short for cooccurrence counting:",line)
        else:
            self.count_start_of_text(words, window)
            self.count_middle_of_text(words, window)
            self.count_end_of_text(words, window)

        if self.verbose:
            print("\nfinished counting; matrix shape:",self.cooc.shape)
            print("vocabulary size:",len(self.words_to_i))
            print("first words in the vocabulary:\n\t",
                  [str(self.words_to_i[key])+":"+key for key in sorted(self.words_to_i, key=self.words_to_i.get)][:10])




    #========== LOGGING

    def log_matrix(self, outspace=None, outcols=None, only_these=None, verbose=False):
        if outspace is None: outspace = self.outspace
        if outcols  is None: outcols  = self.outcols
        if only_these is None: only_these  = self.words_to_i

        with open(outspace, "w") as dm_file, open(outcols, "w") as cols_file:
            if self.verbose:
                print("\nwriting vectors to",outspace,
                      "\nwriting dictionary to",outcols,"...")

            for word,i in tqdm(sorted(only_these.items(), key=lambda x: x[1])):
                cols_file.write(word+"\n")
                vectorstring = " ".join([str(v) for v in self.cooc[i]])
                dm_file.write(word+" "+vectorstring+"\n")

    def log_fly(self, flyfile=None, verbose=False): # allows to specify a destination
        if flyfile is None: flyfile = self.flyfile
        if flyfile is not None and self.fruitfly is not None:
            if verbose: print("\nlogging fruitfly to",flyfile,"...")
            self.fruitfly.log_params(filename=flyfile, timestamp=False)

    def get_setup(self):
        return {
            "verbose":self.verbose,
            "corpus_dir":self.corpus_dir,
            "is_tokenize":self.is_tokenize,
            "is_linewise":self.is_linewise,
            "required_voc":self.required_voc,
            "outspace":self.outspace,
            "outcols":self.outcols,
            "is_incremental":self.is_incremental,
            "max_dims":self.max_dims,
            "is_new_fly":self.is_new_fly,
            "is_grow_fly":self.is_grow_fly,
            "flyfile":self.flyfile,
            "fly_max_pn":self.fly_max_pn,
            "cooc":self.cooc[:10,:10],
            "words_to_i":sorted(self.words_to_i, key=self.words_to_i.get)[:10],
            "fruitfly":self.fruitfly.get_specs(),
            "words":self.words[:20],
            "freq":sorted(self.freq, key=self.freq.get, reverse=True)[:10]
        }




if __name__ == '__main__':
    arguments = docopt(__doc__)

    is_verbose = arguments["--verbose"]
    infile = arguments["<text_source>"] # e.g. "data/potato.txt"
    outfiles = arguments["<out_file>"] # e.g. "data/potato"

    tknz = arguments["--tokenize"]
    lnws = arguments["--linewise"]
    xvoc = arguments["-x"]

    incr = arguments["--increment"]
    try: dims=int(arguments["-d"])
    except TypeError: dims=None

    nfly = arguments["new"]
    grow = arguments["--grow_fly"]
    fcfg = arguments["<config>"]

    window = int(arguments["-w"]) # not part of the Incrementor object


    incrementor = Incrementor(infile, outfiles,
                              corpus_tokenize=tknz, corpus_linewise=lnws, corpus_checkvoc=xvoc,
                              matrix_incremental=incr, matrix_maxdims=dims,
                              fly_new=nfly, fly_grow=grow, fly_file=fcfg, fly_max_pn=None,
                              verbose=is_verbose)

    if is_verbose: print("\nchecking overlap...")
    all_in, unshared_words = incrementor.check_overlap(checklist_filepath=xvoc, verbose=incrementor.verbose)

    incrementor.count_cooccurrences(words=incrementor.words, window=window, verbose=incrementor.verbose)
    incrementor.log_matrix(verbose=incrementor.verbose)
    incrementor.log_fly(verbose=incrementor.verbose)

    print("done.")


"""
"""

