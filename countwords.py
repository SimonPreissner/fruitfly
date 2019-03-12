"""Countwords: Create or extend a co-occurrence matrix (and optionally a corresponding fruitfly).

Usage:
  countwords.py [--help] 
  countwords.py [-i] <text_source> [-t] [-l] <out_file> [-f <ff_config>] [-d <dims>] [-w <window>] [-x <checkfile>] [-v]
  countwords.py <text_source> <out_file> [-d <dims>] [-v]

Options:
  -h --help        show this screen
  -i --increment   use out_file as text_source (= extend the space)
  -t --tokenize    run a simple tokenizer over the input text
  -l --linewise    don't count cooccurrences across lines
  -f=<ff_config>   develop a fruit fly alongside the space
  -d=<dims>        limit space to a number of dimensions
  -w=<window>      number of tokens in the context (to each side) [default: 5]
  -x=<checkfile>   check for full word overlap of the space with 'checkfile' 
  -v --verbose     comment program status with command-line output
  
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
import MEN
import re
import numpy as np
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer


#========== PARAMETER INPUT

increment_mode = arguments["--increment"]
tokenization_is_required = arguments["--tokenize"]
linewise_corpus = arguments["--linewise"]
infile = arguments["<text_source>"] # e.g. "data/potato.txt"
outfile = arguments["<out_file>"]+".dm" # e.g. "data/potato"
outcols = arguments["<out_file>"]+".cols"

fly_file = arguments["-f"]
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

    if lc > 1 and linewise_corpus is False:
        return [w for l in lines for w in l] # flattens to a simple word list
    else: 
        return(lines)

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
        #overlap = [w for w in frequency_sorted if w in checklist] # overlap of the vocabulary with required words
        rest_words = [w for w in frequency_sorted if w not in overlap] # words that are not required; sorted by frequency
        returnlist = overlap+rest_words 
        """
        if verbose_wanted:
            print("required words that are in the corpus:", overlap)
            print("first rest words:", rest_words[:30])
            print("first returned words and their frequencies:")
            for w in returnlist[:50]:
                print(returnlist.index(w),"\t",freq[w],"\t",w)
        """
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
        
    pos_tag = re.compile("_.+?") #if it's POS-tagged, this will get rid of that
    return [re.sub(pos_tag, "", w) for w in checklist] # add [:10] for test purposes ONLY!

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
            print("\ŧChecked for overlap with",checklist_filepath,\
                  "\n\twords missing in the corpus:",len(unshared_words),\
                  "\n\texamples:",unshared_words[:10])

    return (unshared_words is True), unshared_words

#========== CO-OCCURRENCE COUNTING

def extend_matrix_if_necessary(w):
    global cooc, words_to_i
    if w not in words_to_i:
        words_to_i[w] = len(words_to_i) # extend the vocabulary
        temp = np.zeros((len(words_to_i), len(words_to_i))) # make bigger matrix
        temp[0:cooc.shape[0], 0:cooc.shape[1]] = cooc # paste current matrix into the new one
        cooc = temp

        #fruitfly.extend_pn() #TODO add input node to the pn_layer

def count_start_of_text(words): # for the first couple of words
    global cooc, words_to_i
    for i in range(window): 
        if words[i] in freq:
            for c in range(i+window+1): # iterate over the context
                if words[c] in freq:
                    extend_matrix_if_necessary(words[i])
                    extend_matrix_if_necessary(words[c])
                    cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 # increment cooccurrence
            cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1

def count_middle_of_text(words): # for most of the words
    global cooc, words_to_i
    if linewise_corpus: #this is without tqdm, the other loop with.
        for i in range(window, len(words)-window): 
            if words[i] in freq:
                for c in range(i-window, i+window+1): 
                    if words[c] in freq:
                        extend_matrix_if_necessary(words[i])
                        extend_matrix_if_necessary(words[c])
                        cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 
                cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1
    else:
        for i in tqdm(range(window, len(words)-window)): 
            if words[i] in freq:
                for c in range(i-window, i+window+1): 
                    if words[c] in freq:
                        extend_matrix_if_necessary(words[i])
                        extend_matrix_if_necessary(words[c])
                        cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 
                cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1

def count_end_of_text(words): # for the last couple of words
    global cooc, words_to_i    
    for i in range(len(words)-window, len(words)): 
        if words[i] in freq:
            for c in range(i-window, len(words)):
                if words[c] in freq:
                    extend_matrix_if_necessary(words[i])
                    extend_matrix_if_necessary(words[c])
                    cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 
            cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1



#========== EXECUTIVE CODE

if increment_mode:
    if verbose_wanted: print("\nloading existing space...")
    unhashed_space = utils.readDM(outfile) # returns dict of word : vector
    cooc = np.stack([v for k,v in unhashed_space.items()])
    i_to_words, words_to_i = utils.readCols(outcols)
    if fly_file is not None:
        if verbose_wanted: print("\nloading fruitfly...")
        fruitfly = Fruitfly.from_config(fly_file) #TODO test from_config()
else: 
    cooc = np.array([[]]) # cooccurrence count (only numbers)
    words_to_i = {} # vocabulary and word positions 
    fruitfly = None

if verbose_wanted: print("\nreading corpus...")
words = read_corpus(infile)

if verbose_wanted: print("\ncreating frequency distribution...")
freq = freq_dist(words, size_limit=max_dims, required_words=required_voc)
if verbose_wanted: print("\tVocabulary size:",len(freq))
if verbose_wanted: print("\tTokens (/lines) for cooccurrence count:",len(words))

if verbose_wanted: print("\nchecking overlap...")
all_in, unshared_words = check_overlap(freq.keys(), required_voc)



#CLEANUP
# for now, the matrix extension is done beforehand
#if verbose_wanted: print("creating empty matrix...")
#wordset = set(words)
#for w in freq.keys(): # This limits the matrix to the required size
#        cooc, words_to_i = extend_matrix_if_necessary(cooc, words_to_i, w)

if verbose_wanted: print("\ncounting cooccurrences...")
if linewise_corpus:
    for line in tqdm(words):
        if len(line) >= 2*window: # to avoid index errors
            count_start_of_text(line)
            count_middle_of_text(line)
            count_end_of_text(line)
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

#outfile=rawtext[:-3]+"dm" # change the (3-letter) file ending
with open(outfile, "w") as dm_file, open(outcols, "w") as cols_file:
    if verbose_wanted:
        print("\nwriting vectors to",outfile,\
            "\n\tand dictionary to",outcols,"...")
    counter = 0
    for word,i in tqdm(sorted(words_to_i.items(), key=lambda x: x[1])):
        cols_file.write(word+"\n")
        vectorstring = " ".join([str(v) for v in cooc[i]])
        dm_file.write(word+" "+vectorstring+"\n")
        counter += 1

print("done.")


#TODO update the README when the incremental part is implemented
