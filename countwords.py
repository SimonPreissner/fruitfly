import sys
import utils
import Fruitfly
import MEN
import numpy as np
from nltk.tokenize import RegexpTokenizer

"""
This script compiles data from text into cooccurrence matrices.
"""

\#TODO parameter input

#if len(sys.argv) < 5 or sys.argv[1] not in ("bnc","wiki","rawiki"):
#    print("USAGE: python3 countwords.py [-u] [infile] [outfile]")
#    sys.exit() 
#
#if sys.argv[1] == "bnc":
#    data = "data/BNC-MEN.dm"
#    column_labels = "data/BNC-MEN.cols"

rawtext = "data/potato.txt"
with open(rawtext) as f:
	rawlines=f.readlines()
f.close()

cooc = np.array([[]]) # cooccurrence count (only numbers)
words_to_i = {} # vocabulary and word positions 


\#TODO get config sorted out
#fruitfly = Fruitfly.Fruitfly(pn_size, kc_factor*pn_size, projections, hash_size)


tokenizer = RegexpTokenizer(r'\w+')
lines = [tokenizer.tokenize(l) for l in rawlines] # tokenized sentences
#print(lines)

# fill and extend the cooccurrence matrix
\#TODO solve the line[i] thingy with enumerate()?
for line in lines:
	for i in range(len(line)): # count cooccurrence with previous word 
		if line[i] not in words_to_i:

			words_to_i[line[i]] = len(words_to_i) # extend the vocabulary
			
			temp = np.zeros((len(words_to_i), len(words_to_i))) # make bigger matrix
			temp[0:cooc.shape[0], 0:cooc.shape[1]] = cooc # paste current matrix into the new one
			cooc = temp

\#    		fruitfly.extend_pn() 					# add input node to the pn_layer

		if i == 0: # to avoid range problems
			pass
		else:
			cooc[words_to_i[line[i]]][words_to_i[line[i-1]]] += 1 # increment cooccurrence


print(cooc)
print("cooc shape:",cooc.shape)
print([str(words_to_i[key])+" : "+key for key in sorted(words_to_i, key=words_to_i.get)])
print("dict length:",len(words_to_i))



outfile=rawtext[:-3]+"dm" # change the (3-letter) file ending
with open(outfile, "w") as f:
	for word,i in words_to_i.items():
		f.write(word+" "+np.array_str(cooc[i])[1:-1]+"\n")
		# print(np.array_str(cooc[i])[1:-1])
f.close()