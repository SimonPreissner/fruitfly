import re

"""
This script was used to extract from the ukwac_100m_oneline corpus
those passages that contain the words in the sandbox test corpus.
The product of this script will be used as corpus within the sandbox.
"""

original_corpus = "../../ukwac_100m/ukwac_100m_oneline.txt"
checklist_filepath = "data/sandbox_MEN_pairs"
outfile = "data/sandbox_corpus.txt"

words = []
nonword = re.compile("\W+") # to delete punctuation entries
with open(original_corpus) as f:
    for text in f:
        text = text.lower() # lowercase everything
        tokens = text.rstrip().split()
        for t in tokens:
            if (re.fullmatch(nonword, t) is None): # get rid of tokens that are punctuation etc.
                words.append(t) # if there are multiple lines, extend() takes care of it
                if len(words)%1000000 == 0:
                    print("words read:",len(words))


checklist = []
with open(checklist_filepath, "r") as f:
    print ("checking overlap with the gold standard:",checklist_filepath,"...")
    for line in f:
        words = line.rstrip().split()[:2]
        checklist.extend(words)
checkset = set(checklist)

for w in words[10:-10]:
	if w in checkset:
		#TODO continue here with extraction of the passages



