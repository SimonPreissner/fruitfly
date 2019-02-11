import re

wacfile = "/mnt/8tera/corpora/ukwac/ukwac_100m.txt"
#wacfile = "data/ukwac_100m_demo.txt"
menfile = "data/BNC-MEN.cols"

all_in = True

wordlist_WAC = []
unshared_words = []

reg = re.compile("_.+?")


with open(wacfile, "r") as f:
    for word in f:
        word = word.rstrip() 
        wordlist_WAC.append(word)
        print("items already read:",len(wordlist_WAC))
            
with open(menfile, "r") as f:
    for word in f:
        word = word.rstrip()
        word = re.sub(reg, "",word)
        if (word not in wordlist_WAC):
            unshared_words.append(word)
            all_in = False

if(all_in):
    print("all MEN words are in the data set.")
else:
    print(unshared_words)
    print("These",len(unshared_words),"MEN words are not in the dataset.")
            
