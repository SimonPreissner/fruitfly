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
    print("USAGE: python3 countwords.py [infile] [outfiles] -t -v -dim [k] -check [file] -window [n]\n\
    	  [infile]: raw input of text\n\
    	  [outfile]: output file of vectors (produces a .dm and a .cols file)\n\
    	  -t: optionally run an nltk tokenizer\n\
    	  -v: optionally run with command line output\n\
    	  -dim: optionally limit dimensions to the [k] most frequent words\n\
    	  -check: see whether all words of [file] are in the corpus\n\
    	  -window: select the scope for cooccurrence counting ([n] words to each side); default is 5")
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
	return(words)

def freq_dist(wordlist, size_limit=None):
	freq = {}
	for w in wordlist:
		if w in freq:
			freq[w] += 1
		else:
			freq[w] = 1

	if(size_limit is not None and size_limit <= len(freq)):
		frequency_sorted = sorted(freq, key=freq.get, reverse=True) # list of all words
		return {k:freq[k] for k in frequency_sorted[:size_limit]}
	else:
		return freq

def check_overlap(wordlist, checklist):
	if(checklist is None): # if no checking is specified, go on without checking
		if verbose_wanted is True:
			print("check_overlap(): nothing to check.")
		return(True, [])

	all_in = True
	unshared_words = []
	pos_tag = re.compile("_.+?") #if it's POS-tagged, this will get rid of that
	            
	with open(checklist, "r") as f:
	    for word in f:
	        word = word.rstrip()
	        word = re.sub(pos_tag, "",word)
	        if (word not in words):
	            unshared_words.append(word)
	            all_in = False
	if verbose_wanted is True:
		if all_in is True:
			print("all required words are in the corpus")
		else:
			print("Some of the",len(unshared_words),"words that are not in the corpus:\n",\
				  unshared_words[:min(round(len(unshared_words)/10), 100)])

	return all_in, unshared_words




#========== COOCCURRENCE COUNTING

def extend_matrix_if_necessary(cooc, words_to_i, word):
	if word not in words_to_i:
		words_to_i[word] = len(words_to_i) # extend the vocabulary
		temp = np.zeros((len(words_to_i), len(words_to_i))) # make bigger matrix
		temp[0:cooc.shape[0], 0:cooc.shape[1]] = cooc # paste current matrix into the new one
		cooc = temp
		#fruitfly.extend_pn() \#TODO add input node to the pn_layer
		return cooc
	else:
		return cooc


cooc = np.array([[]]) # cooccurrence count (only numbers)
words_to_i = {} # vocabulary and word positions 

words = read_corpus(infile)
freq = freq_dist(words, size_limit=max_dims)
all_in, unshared_words = check_overlap(words, required_voc)


# for now, the matrix extension is done beforehand
if verbose_wanted is True:
	print("creating empty matrix...")
for w in words:
	if w in freq:
		cooc = extend_matrix_if_necessary(cooc, words_to_i, w)


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
	for word,i in words_to_i.items():
		cols_file.write(word+"\n")
		dm_file.write(word+" "+np.array_str(cooc[i], max_line_width=100000000)[1:-1]+"\n")



"""
print("so far, everything works")
sys.exit()
"""
