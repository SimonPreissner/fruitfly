import numpy as np

"""
This script was used to check whether every word from the gold standard
is in the ukwac_100m corpus.
"""


wacfile = "../ukwac_100m/ukwac_100m_oneline.txt"
menfile = "data/MEN_dataset_natural_form_full"

wordlist_WAC = []
checklist = []
unshared_words = []

with open(wacfile, "r") as f:
    print("reading",wacfile,"...")
    for line in f:
        line = line.rstrip() 
        wordlist_WAC.extend(line.split())
    print("Finished reading. Tokens in the corpus:",len(wordlist_WAC))
            
with open(menfile, "r") as f:
    for line in f:
        words = line.rstrip().split()[:2]
        checklist.extend(words)

wordset_WAC = set(wordlist_WAC)
checkset = set(checklist)
print("Types in the corpus:   ", len(wordset_WAC))
print("Types in the checklist:", len(checkset))

unshared_words = list(checkset.difference(wordset_WAC))

if unshared_words is False:
    print("All required words are in the corpus.")
else:
    print(unshared_words[:min(int(np.ceil(len(unshared_words)/10)), 25)])
    print("These are some of the",len(unshared_words),"words are not in the corpus.")

            
