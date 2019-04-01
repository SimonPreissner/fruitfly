"""Countwords: Create or extend a co-occurrence matrix (and optionally a corresponding fruitfly).

Usage:
  countwords.py [--help] 
  countwords.py <text_source> <out_file> [--grow_fly [new] <ff_config>] [options]

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


import sys
from docopt import docopt
#=============== PARAMETER INPUT
if __name__ == '__main__':
    arguments = docopt(__doc__)

import utils
import Fruitfly
from Fruitfly import Fruitfly
import MEN
import re
import numpy as np
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer



#========== PARAMETER INPUT

increment_mode = arguments["--increment"]
tokenization_is_required = arguments["--tokenize"]
linewise_corpus = arguments["--linewise"]

grow_fly = arguments["--grow_fly"]
fly_from_scratch = arguments["new"]
fly_file = arguments["<ff_config>"]

infile = arguments["<text_source>"] # e.g. "data/potato.txt"
outfile = arguments["<out_file>"]+".dm" # e.g. "data/potato"
outcols = arguments["<out_file>"]+".cols"

try: max_dims=int(arguments["-d"]) 
except TypeError: max_dims=None
window = int(arguments["-w"])
required_voc = arguments["-x"]

verbose_wanted = arguments["--verbose"]



#========== FILE READING

def read_corpus(infile):
    lines = [] # list of lists of words
    nonword = re.compile("\W+") # to delete punctuation entries
    lc = 0 # for files with more than one line
    wc = 0 # wordcount
    with open(infile) as f:
        for line in f:
            lc += 1
            line = line.rstrip().lower()
            if tokenization_is_required:
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(line)
            else:
                tokens = line.split()
            linewords = []
            for t in tokens:
                if (re.fullmatch(nonword, t) is None): # ignore punctuation
                    linewords.append(t) # adds the list as a unit to 'lines'
                wc+=1
                if verbose_wanted and wc%100000 == 0:
                    print("\twords read:",wc/1000000,"million",end="\r")
            lines.append(linewords)


    if linewise_corpus is False:
        return [w for l in lines for w in l] # flattens to a simple word list
    else: 
        return(lines)

def read_incremental_parts(outfile, flyfile): # matrix, vocabulary, fruitfly (if wanted)
    """
    Return a matrix, a vocabulary, and a Fruitfly object.
    The matrix can be newly instantiated or taken from an already 
    existing space; the vocabulary aswell.
    The fruitfly can be optionally created alongside, also either new 
    or from an already existing config file.
    """
    if increment_mode:
        if verbose_wanted: print("\nloading existing space...")
        unhashed_space = utils.readDM(outfile) # returns dict of word : vector
        i_to_words, words_to_i = utils.readCols(outcols)
        dimensions = sorted(words_to_i, key=words_to_i.get)
        cooc = np.stack(tuple([unhashed_space[w] for w in dimensions]))
    else: 
        cooc = np.array([[]]) # cooccurrence count (only numbers)
        words_to_i = {} # vocabulary and word positions 
    
    if grow_fly:
        if fly_from_scratch:
            if verbose_wanted: print("\ncreating new fruitfly...")
            fruitfly = Fruitfly.from_scratch() # default config: 2pn,8kc,6proj,5perc
        else:
            if verbose_wanted: print("\nloading fruitfly...")
            fruitfly = Fruitfly.from_config(flyfile)
    else:
        fruitfly = None

    return cooc, words_to_i, fruitfly

def freq_dist(wordlist, size_limit=None, required_words=None):
    freq = {}
    if linewise_corpus:
        for line in tqdm(wordlist):
            for w in line:
                if w in freq:
                    freq[w] += 1
                else:
                    freq[w] = 1
    else:
        for w in tqdm(wordlist):
            if w in freq:
                freq[w] += 1
            else:
                freq[w] = 1

    frequency_sorted = sorted(freq, key=freq.get, reverse=True) # list of all words

    if required_words is not None:
        checklist = read_checklist(required_words)
        overlap = list(set(checklist).intersection(set(frequency_sorted)))
        rest_words = [w for w in frequency_sorted if w not in overlap] # words that are not required; sorted by frequency
        returnlist = overlap+rest_words 
    else: 
        returnlist = frequency_sorted

    if(size_limit is not None and size_limit <= len(freq)):
        return {k:freq[k] for k in returnlist[:size_limit]}
    else:
        return freq

def read_checklist(checklist_filepath):
    if checklist_filepath is None:
        return []

    checklist = []

    with open(checklist_filepath, "r") as f:
        #TODO generalize this so that it takes any text file
        paired_lists = ["data/MEN_dataset_natural_form_full",
                        "incrementality_sandbox/data/sandbox_MEN_pairs"]
        if checklist_filepath in paired_lists: 
            for line in f:
                words = line.rstrip().split()[:2]
                checklist.extend(words)
        else:
            for word in f:
                word = word.rstrip()
                checklist.append(word)
        
    pos_tag = re.compile("_.+?") # get rid of simple POS-tags
    return [re.sub(pos_tag, "", w) for w in checklist]

def check_overlap(wordlist, checklist_filepath):
    checklist = read_checklist(checklist_filepath)
    if len(checklist) == 0: # if no checking is specified, go on without checking
        if verbose_wanted: 
            print("\tcheck_overlap(): nothing to check.")
        return True, []

    unshared_words = list(set(checklist).difference(set(wordlist)))

    if verbose_wanted:
        if len(unshared_words) == 0:
            print("\tComplete overlap with",checklist_filepath)
        else:
            print("\tChecked for overlap with",checklist_filepath,\
                  "\n\twords missing in the corpus:",len(unshared_words),\
                  "\n\texamples:",unshared_words[:10])

    return (unshared_words is True), unshared_words



#========== CO-OCCURRENCE COUNTING

def extend_incremental_parts_if_necessary(w): # matrix dimensions and fruitfly PN layer
    global cooc, words_to_i

    if w not in words_to_i:
        words_to_i[w] = len(words_to_i) # extend the vocabulary
        temp = np.zeros((len(words_to_i), len(words_to_i))) # make bigger matrix
        temp[0:cooc.shape[0], 0:cooc.shape[1]] = cooc # paste current matrix into the new one
        cooc = temp
    if fruitfly is not None and len(words_to_i) > fruitfly.pn_size:
        fruitfly.extend() # extend if needed (incl. "catching up" with vocabulary size)

def count_start_of_text(words): # for the first couple of words
    global cooc, words_to_i
    for i in range(window): 
        if words[i] in freq:
            for c in range(i+window+1): # iterate over the context
                if words[c] in freq:
                    extend_incremental_parts_if_necessary(words[i])
                    extend_incremental_parts_if_necessary(words[c])
                    cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1
            cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1 # delete "self-occurrence"

def count_middle_of_text(words): # for most of the words
    global cooc, words_to_i
    if linewise_corpus: # this loop is without tqdm, the other loop with.
        for i in range(window, len(words)-window): 
            if words[i] in freq: 
                for c in range(i-window, i+window+1): # iterate over the context
                    if words[c] in freq:
                        extend_incremental_parts_if_necessary(words[i])
                        extend_incremental_parts_if_necessary(words[c])
                        cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 
                cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1 # delete "self-occurrence"
    else:
        for i in tqdm(range(window, len(words)-window)): 
            if words[i] in freq:
                for c in range(i-window, i+window+1): # iterate over the context
                    if words[c] in freq:
                        extend_incremental_parts_if_necessary(words[i])
                        extend_incremental_parts_if_necessary(words[c])
                        cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 
                cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1 # delete "self-occurrence"

def count_end_of_text(words): # for the last couple of words
    global cooc, words_to_i    
    for i in range(len(words)-window, len(words)):
        if words[i] in freq:
            for c in range(i-window, len(words)): # iterate over the context
                if words[c] in freq:
                    extend_incremental_parts_if_necessary(words[i])
                    extend_incremental_parts_if_necessary(words[c])
                    cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 
            cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1 # delete "self-occurrence"



#========== EXECUTIVE CODE

cooc, words_to_i, fruitfly = read_incremental_parts(outfile, fly_file)

if verbose_wanted: print("\nreading corpus...")
words = read_corpus(infile)

if verbose_wanted: print("\ncreating frequency distribution...")
# words that will be counted (= labels of the final matrix dimensions)
freq = freq_dist(words, size_limit=max_dims, required_words=required_voc)

if verbose_wanted: print("\tVocabulary size:",len(freq),\
                       "\n\tTokens (or lines) for cooccurrence count:",len(words))

if verbose_wanted: print("\nchecking overlap...")
all_in, unshared_words = check_overlap(freq.keys(), required_voc)


if verbose_wanted: print("\ncounting cooccurrences within",window,"words distance...")
if linewise_corpus:
    for line in tqdm(words):
        if len(line) >= 2*window: # to avoid index errors
            count_start_of_text(line)
            count_middle_of_text(line)
            count_end_of_text(line)
        else:
            if verbose_wanted: print("\tline too short for cooccurrence counting:",line)
else:
    count_start_of_text(words)
    count_middle_of_text(words)
    count_end_of_text(words)

if verbose_wanted:
    print("\nfinished counting; matrix shape:",cooc.shape)
    print("vocabulary size:",len(words_to_i))
    print("first words in the vocabulary:\n\t",\
           [str(words_to_i[key])+":"+key for key in sorted(words_to_i, key=words_to_i.get)][:10])



#========== OUTPUT

with open(outfile, "w") as dm_file, open(outcols, "w") as cols_file:
    if verbose_wanted:
        print("\nwriting vectors to",outfile,\
              "\nwriting dictionary to",outcols,"...")
    counter = 0
    for word,i in tqdm(sorted(words_to_i.items(), key=lambda x: x[1])):
        cols_file.write(word+"\n")
        vectorstring = " ".join([str(v) for v in cooc[i]])
        dm_file.write(word+" "+vectorstring+"\n")
        counter += 1

if fly_file is not None and fruitfly is not None: 
    if verbose_wanted: print("\nlogging fruitfly to",fly_file,"...")
    fruitfly.log_params(filename=fly_file, timestamp=False)  

print("done.")
sys.exit() #BREAKPOINT

