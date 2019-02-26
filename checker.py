import re
import numpy as np

"""
This script was used to check whether every word from the gold standard
is in the ukwac_100m corpus.
"""


wacfile = "../ukwac_100m/ukwac_100m_oneline.txt"
#wacfile = "data/ukwac_100m_oneline_demo.txt"
menfile = "data/MEN_dataset_natural_form_full"
#menfile = "data/BNC-MEN.cols"

wordlist_WAC = []
checklist = []
unshared_words = []

reg = re.compile("_.+?")


with open(wacfile, "r") as f:
    for line in f:
        line = line.rstrip() 
        wordlist_WAC.extend(line.split())
        print("items already read:",len(wordlist_WAC))
            
with open(menfile, "r") as f:
    if menfile == "data/MEN_dataset_natural_form_full": 
        for line in f:
            words = line.rstrip().split()[:2]
            checklist.extend(words)
    else:
        for word in f:
            word = word.rstrip()
            word = re.sub(reg, "",word)
            checklist.appent(word)

unshared_words = list(set(checklist).difference(set(wordlist_WAC)))

if unshared_words is True:
    print("all required words are in the corpus.")
else:
    print(unshared_words[:min(int(np.ceil(len(unshared_words)/10)), 25)])
    print("These are some of the",len(unshared_words),"words are not in the corpus.")

print("types in the corpus:   ", len(set(wordlist_WAC)))
print("types in the checklist:", len(set(checklist)))
            
