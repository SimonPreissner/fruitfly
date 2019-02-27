import sys
import utils
import Fruitfly
import MEN
import re
import numpy as np
from nltk.tokenize import RegexpTokenizer

"""
This script compiles data from text into cooccurrence matrices.
"""


if len(sys.argv) < 3:
    print("USAGE: python3 countwords.py [infile] [outfiles] -t -dim [k] -window [n] -check [file] -v\n\
          [infile]: raw input of text\n\
          [outfile]: output files of vectors WITHOUT file extension (produces a .dm and a .cols file)\n\
          -t: optionally run an nltk tokenizer on the input file\n\
          -dim: optionally limit dimensions to the [k] most frequent words\n\
          -window: select the scope for cooccurrence counting ([n] words to each side); default is 5\
          -check: see whether all words of [file] are in the corpus\n\
          -v: optionally run with command line output\n")
    sys.exit() 


#========== PARAMETER INPUT

infile = sys.argv[1] # e.g. "data/potato.txt"
outfile = sys.argv[2]+".dm" # e.g. "data/potato"
outcols = sys.argv[2]+".cols"
tokenization_is_required = ("-t" in sys.argv)
verbose_wanted = ("-v" in sys.argv)

if ("-dim" in sys.argv):
    max_dims = int(sys.argv[sys.argv.index("-dim")+1]) # take the number after '-dim' as value
else:
    max_dims = None

if ("-check" in sys.argv):
    required_voc = sys.argv[sys.argv.index("-check")+1] # 
else:
    required_voc = None
if ("-window" in sys.argv):
    window = int(sys.argv[sys.argv.index("-window")+1]) # take the number after '-dim' as value
else:
    window = 5


#========== FILE READING

def read_corpus(infile):
    words = []
    nonword = re.compile("\W+") # to delete punctuation entries
    with open(infile) as f:
        for text in f:
            text = text.lower() # lowercase everything
            if(tokenization_is_required):
                tokenizer = RegexpTokenizer(r'\w+')
                text = " ".join(tokenizer.tokenize(text)) # format the tokenized words into a space-separated string
            tokens = text.rstrip().split()
            for t in tokens:
                if (re.fullmatch(nonword, t) is None): # get rid of tokens that are punctuation etc.
                    words.append(t) # if there are multiple lines, extend() takes care of it
                    if len(words)%1000000 == 0:
                        print("words read:",len(words))
    return(words)

def freq_dist(wordlist, size_limit=None, required_words=None):
    freq = {}
    for w in wordlist:
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
        if verbose_wanted is True:
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
    pos_tag = re.compile("_.+?") #if it's POS-tagged, this will get rid of that

    with open(checklist_filepath, "r") as f:
        #TODO generalize this so that it takes any text file
        if checklist_filepath == "data/MEN_dataset_natural_form_full": 
            if verbose_wanted is True:
                print ("checking overlap with the gold standard:",checklist_filepath,"...")
            for line in f:
                words = line.rstrip().split()[:2]
                checklist.extend(words)
        for word in f:
            word = word.rstrip()
            word = re.sub(pos_tag, "",word)
            checklist.append(word)
    return checklist # add [:10] for test purposes ONLY!

def check_overlap(wordlist, checklist_filepath):
    checklist = read_checklist(checklist_filepath)
    if checklist is False: # if no checking is specified, go on without checking
        if verbose_wanted is True:
            print("check_overlap(): nothing to check.")
        return True, []

    unshared_words = list(set(checklist).difference(set(wordlist)))

    if verbose_wanted is True:
        if unshared_words is True:
            print("Complete overlap with",checklist_filepath)
        else:
            print("Checked for overlap with",checklist_filepath,\
                  "- some of the",len(unshared_words),"words missing in the corpus:\n",\
                  unshared_words[:min(int(np.ceil(len(unshared_words)/10)), 25)])

    return (unshared_words is True), unshared_words




#========== COOCCURRENCE COUNTING

def extend_matrix_if_necessary(cooc, words_to_i, word):
    if word not in words_to_i:
        words_to_i[word] = len(words_to_i) # extend the vocabulary
        temp = np.zeros((len(words_to_i), len(words_to_i))) # make bigger matrix
        temp[0:cooc.shape[0], 0:cooc.shape[1]] = cooc # paste current matrix into the new one
        cooc = temp
        #fruitfly.extend_pn() \#TODO add input node to the pn_layer
        return cooc, words_to_i
    else:
        return cooc, words_to_i



cooc = np.array([[]]) # cooccurrence count (only numbers)
words_to_i = {} # vocabulary and word positions 

words = read_corpus(infile)
freq = freq_dist(words, size_limit=max_dims, required_words=required_voc)
if verbose_wanted is True:
    print("finished creating frequency distribution with",len(freq),"entries.")
all_in, unshared_words = check_overlap(freq.keys(), required_voc)


# for now, the matrix extension is done beforehand
if verbose_wanted is True:
    print("creating empty matrix...")
for w in set(words):
    if w in freq: # This limits the matrix to 
        cooc, words_to_i = extend_matrix_if_necessary(cooc, words_to_i, w)


if verbose_wanted is True:
    print("counting cooccurrences...")
# all the "in freq" checking is in order to only count the most frequent words
for i in range(window): # for the first couple of words
    if words[i] in freq:
        #cooc = extend_matrix_if_necessary(cooc, words_to_i, words[i])
        for c in range(i+window+1):
            if words[c] in freq:
                cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 # increment cooccurrence
        cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1 # don't count cooccurrence with yourself!

for i in range(window, len(words)-window): # for most of the words
    if verbose_wanted is True and i%1000000 == 0:
        print("words already processed:",i)
    if words[i] in freq:
        #cooc = extend_matrix_if_necessary(cooc, words_to_i, words[i])
        for c in range(i-window, i+window+1): 
            if words[c] in freq:
                cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 
        cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1

for i in range(len(words)-window, len(words)): # for the last couple of words
    if words[i] in freq:
        #cooc = extend_matrix_if_necessary(cooc, words_to_i, words[i])
        for c in range(i-window, len(words)):
            if words[c] in freq:
                cooc[words_to_i[words[i]]][words_to_i[words[c]]] += 1 
        cooc[words_to_i[words[i]]][words_to_i[words[i]]]-=1

if verbose_wanted is True:
    print("finished counting cooccurrences; matrix shape:",cooc.shape)
    print("vocabulary size:",len(words_to_i))
    print("first words in the vocabulary:",\
           [str(words_to_i[key])+":"+key for key in sorted(words_to_i, key=words_to_i.get)][:25])




#========== OUTPUT

#outfile=rawtext[:-3]+"dm" # change the (3-letter) file ending
with open(outfile, "w") as dm_file, open(outcols, "w") as cols_file:
    if verbose_wanted is True:
        print("writing vectors to",outfile,"and dictionary to",outcols,"...")
    counter = 0
    for word,i in sorted(words_to_i.items(), key=lambda x: x[1]):
        cols_file.write(word+"\n")
        vectorstring = " ".join([str(v) for v in cooc[i]])
        dm_file.write(word+" "+vectorstring+"\n")
        #dm_file.write(word+" "+np.array_str(cooc[i], max_line_width=100000000)[1:-1]+"\n")
        if verbose_wanted is True and counter%100==0:
            print(counter,"word vectors written...")
        counter += 1



"""
print("so far, everything works")
sys.exit()
"""


#TODO update the README when the incremental part is implemented
