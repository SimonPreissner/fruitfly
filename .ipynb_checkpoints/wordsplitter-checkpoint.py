import re

infile = "/mnt/8tera/corpora/ukwac/ukwac_preproc_tokenised.txt"
outfile = "data/ukwac_100m.txt"

limit = 100000000 # 100 million
wordlist = []

reg = re.compile("_.+?\s")


with open(infile, "r") as f:
    for chunk in f: 
        if (len(wordlist) < limit):
            chunk = chunk.rstrip()
            print(len(wordlist))
            
            raw_words = re.sub(reg, " ",chunk)
            raw_lc_words = raw_words.lower()
            words = raw_lc_words.split()
            
            wordlist.extend(words)
        else:
            print(len(wordlist))
            break
            
with open(outfile, "w") as f:
    for i in range(limit):
        f.write(wordlist[i]+"\n")
        print("items written:",i)