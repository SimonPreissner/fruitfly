import re

"""
This script was used to check whether every word from the 
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

unshared_words = list(set(wordlist_WAC).intersection(set(checklist)))

if unshared_words is True:
    print("all MEN words are in the data set.")
else:
    print(unshared_words)
    print("These",len(unshared_words),"MEN words are not in the dataset.")
            
