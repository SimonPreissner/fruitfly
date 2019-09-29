import re
import os
import time
from typing import Dict, Any

import numpy as np
from docopt import docopt
import nltk

import Fruitfly
from tqdm import tqdm
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
                                   required_words_file=self.required_voc,
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
        if indir is None: # i.e. for initialization without resources
            if verbose: print("No text resources specified. Continuing with empty corpus.")
            lines = []
        else:
            if verbose: print("\nreading text resources from",indir,"...")
            filepaths = []
            lines = [] # list of lists of words
            nonword = re.compile("\W+(_X)?") # to delete punctuation entries in simple-POS-tagged data (_N, _V, _A, _X)
            lc = 0 # for files with more than one line
            wc = 0 # wordcount

            if os.path.isfile(indir): # for a single file that is passed
                filepaths = [indir]
            else:
                for (dirpath, dirnames, filenames) in os.walk(indir):
                    filepaths.extend([dirpath+"/"+f for f in filenames])

            for file in filepaths:
                try:
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
                except FileNotFoundError as e:
                    print(e)
            if verbose: print("Finished reading.",wc,"words read.")

        if linewise is False:
            return [w for l in lines for w in l] # flattens to a simple word list
        else:
            return(lines)

    def extend_corpus(self, text_resource):
        """
        Takes a file path, reads the file's content, and extends the Incrementor object's available text
        as well as its freq_dist.
        :param text_resource: file path
        """
        new_text = self.read_corpus(text_resource,
                                   tokenize_corpus=self.is_tokenize,
                                   postag_simple=self.postag_simple,
                                   linewise=self.is_linewise,
                                   verbose=self.verbose)
        self.words.extend(new_text)
        new_freq = self.freq_dist(new_text,
                                  size_limit=self.max_dims,
                                  required_words_file=self.required_voc,
                                  required_words=self.freq.keys(), # for the new freq.keys() to comply with the old one
                                  verbose=self.verbose)
        # update freq with the new counts -- but this might result in len(freq) > max_dims
        self.freq = self.merge_freqs(self.freq,
                                     new_freq,
                                     required_words_file=self.required_voc,
                                     max_length=self.max_dims)

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

    def freq_dist(self, wordlist, size_limit=None, required_words_file=None, required_words=None, verbose=False):
        """
        This method is used to limit the dimensionality of the count matrix, which speeds up processing.
        The obtained dictionary is used as vocabulary reference of the current corpus at several processing steps.
        For true incrementality, size_limit is None and the dictionary is computed over the currently available corpus.
        If size_limit is None, required_words has no effect on the obtained dictionary.
        :param wordlist: list of (word) tokens from the text resource
        :param size_limit: maximum length of the returned frequency distribution
        :param required_words_file: file path to a list with prioritized words (regardless of their frequencies)
        :param required_words: list of words; used to pass already existing freq keys if freq needs to be extended
        :param verbose: comment on workings via print statements
        :return: dict[str:int]
        """
        if verbose: print("\ncreating frequency distribution over",len(wordlist),"tokens...")
        freq = {}
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

        if required_words_file is None and required_words is None:
            returnlist = frequency_sorted
        else:
            checklist = []
            if required_words_file is not None:
                checklist.extend(self.read_checklist(checklist_filepath=required_words_file, with_pos_tags=self.postag_simple))
            if required_words is not None: # in case that freq needs to be extended
                checklist.extend(required_words)

            overlap = list(set(checklist).intersection(set(frequency_sorted)))
            rest_words = [w for w in frequency_sorted if w not in overlap] # words that are not required; sorted by frequency
            returnlist = overlap+rest_words


        if(size_limit is not None and size_limit <= len(freq)):
            return {k:freq[k] for k in returnlist[:size_limit]}
        else:
            return freq

    def merge_freqs(self, freq1, freq2, required_words_file=None, max_length=None): #TODO test this!
        for k, v in freq2.items():
            if k in freq1:
                freq1[k] += v
            else:
                freq1[k] = v

        frequency_sorted = sorted(freq1, key=freq1.get, reverse=True)  # list of all words, sorted

        if required_words_file is None:
            returnlist = frequency_sorted
        else:
            checklist = self.read_checklist(checklist_filepath=required_words_file, with_pos_tags=self.postag_simple)
            overlap = list(set(checklist).intersection(set(frequency_sorted)))
            rest_words = [w for w in frequency_sorted if w not in overlap]  # words that are not required; sorted by frequency
            returnlist = overlap + rest_words

        if (max_length is not None and max_length <= len(freq1)):
            return {k: freq1[k] for k in returnlist[:max_length]}
        else:
            return freq1

    @staticmethod
    def read_checklist(checklist_filepath, with_pos_tags=False):
        if checklist_filepath is None: #TODO why is this coded like this? maybe try/except?
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

    def count_cooccurrences(self, words=None, window=5, timed=False):
        """
        :param words: list of tokens to be counted
        :param window: int. specifies window size to one side.
        :param timed: bool. If True, this method returns the time taken to executethe method
        :return: float. Seconds taken to execute the method
        """
        t0 = time.time()
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

        if timed is True:
            return time.time()-t0
        else:
            pass

    def reduce_count_size(self, min_count, verbose=False, timed=False):
        t0 = time.time()
        if min_count is None:
            if timed:
                return 0, time.time()-t0
            else:
                return 0
        else:
            #print("count dimensions before reducing:",self.cooc.shape)#CLEANUP
            #print("fruitfly PN layer size before reducing:", self.fruitfly.pn_size)#CLEANUP
            #connectednesses = [len(cons) for cons in self.fruitfly.pn_to_kc.values()]  # CLEANUP
            #print("avg. pn connectedness BEFORE:", round(sum(connectednesses) / self.fruitfly.pn_size, 6))  # CLEANUP
            #connectednesses = [len(cons) for cons in self.fruitfly.proj_functions.values()]  # CLEANUP
            #print("avg. kc connectedness BEFORE:", round(sum(connectednesses) / self.fruitfly.kc_size, 6))  # CLEANUP
            #print("std of kc connectedness BEFORE:", round(np.std(connectednesses, ddof=1), 6)) #CLEANUP

            counted_freq_words = set(self.words_to_i).intersection(set(self.freq)) # because freq and words_to_i might differ!
            if verbose: print("Deleting infrequent words (less than",min_count,"occurrences) from the count matrix...")
            delete_these_w = [w for w in counted_freq_words if self.freq[w]<=min_count ]
            delete_these_i = [self.words_to_i[w] for w in delete_these_w]
            # delete rows and columns from the count matrix
            self.cooc = np.delete(self.cooc, delete_these_i, axis=0)
            self.cooc = np.delete(self.cooc, delete_these_i, axis=1)
            # delete elements from the dictionary
            for w in delete_these_w:
                del(self.words_to_i[w])
            # in the index dictionary, shift words from higher dimensions to the freed-up dimensions
            self.i_to_words = {i:w for i,w in enumerate(sorted(self.words_to_i, key=self.words_to_i.get))}
            # update the index mapping in the dictionary
            self.words_to_i = {w:i for i,w in self.i_to_words.items()}
            # also reduce the FFA!
            if self.fruitfly.pn_size > self.cooc.shape[0]:
                self.fruitfly.reduce_pn_layer(delete_these_i, self.cooc.shape[0])

            if verbose: print("\t",len(delete_these_w),"words deleted. New count dimensions:",self.cooc.shape)

            #print("fruitfly.pn_size after reducing: ",self.fruitfly.pn_size)#CLEANUP
            #connectednesses = [len(cons) for cons in self.fruitfly.pn_to_kc.values()] #CLEANUP
            #print("avg. pn connectedness AFTER:",round(sum(connectednesses) / self.fruitfly.pn_size, 6)) #CLEANUP
            #connectednesses = [len(cons) for cons in self.fruitfly.proj_functions.values()] #CLEANUP
            #print("avg. kc connectedness AFTER:",round(sum(connectednesses) / self.fruitfly.kc_size, 6)) #CLEANUP
            #print("std of kc connectedness AFTER:", round(np.std(connectednesses, ddof=1), 6)) #CLEANUP

            if timed:
                return len(delete_these_w), time.time()-t0
            else:
                return len(delete_these_w)




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
        #TODO: check whether this needs to be updated
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

    is_verbose  = False if input("Be verbose while running? [y/n] ").upper() == "N" else True
    ###
    infile = input("Path to resource(s) to be processed: ")
    outfiles = input("Path/name of the output count data (without extension): ")
    incr = True if input("Work incrementally (= use count data as basis)? [y/n] ").upper() == "Y" else False
    tknz = True if input("Tokenize text before counting? [y/n] ").upper() == "Y" else False
    s = input("Window size (to each side) for counting (default: 5):")
    try:
        window = int(s) if len(s) > 0 else 5 # not part of the Incrementor object
    except TypeError:
        print("Could not convert input to int. Continuing with window size 5.")
    lnws = False if input("Count co-occurrences across line breaks? [y/n] ").upper() == "Y" else True
    s = input("Path to a word list to be checked for overlap (optional): ")
    xvoc = s if len(s) > 0 else None # e.g. "./data/MEN_natural_vocabulary"
    s = input("Maximum dimensions of the count (optional): ")
    try:
        dims = int(s) if len(s) > 0 else None
    except TypeError:
        print("Could not convert input to int. Continuing without limitation of dimensions.")
        dims=None
    ###
    grow = True if input("Maintain an FFA object alongside counting? [y/n] ").upper() == "Y" else False
    if grow:
        nfly = True if input("Make a new FFA? [y/n] ").upper() == "Y" else False
        if nfly:
            fcfg = input("File path of the new FFA's config: ")
        else:
            fcfg = input("File path of the existing FFA's config: ")
    else:
        nfly = None
        fcfg = None


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

    if is_verbose: print("done.")


