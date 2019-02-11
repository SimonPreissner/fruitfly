import re

infile = "/mnt/8tera/corpora/ukwac/ukwac_preproc_tokenised.txt"
outfile = "data/kuwac_100m.txt"

reg = re.compile("_.+?\s")

limit = 1000#00000 # 100 million
wordcount = 0
wordlist = []

with open(infile, "r") as f:
    for chunk in iter(lambda: f.read(4096), b''):
        print("reading batch...")
        while (len(wordlist) < limit):
            raw_words = re.sub(reg, " ",chunk)
            raw_lc_words = raw_words.lower()
            words = raw_lc_words.split()
            
            wordlist.append(words)
            
with open(outfile, "w") as f:
    for i in range(limit):
        f.write(wordlist[i])