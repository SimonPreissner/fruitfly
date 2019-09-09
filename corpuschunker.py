import math
import os

corpus = "/mnt/8tera/corpora/enwiki-20181120/enwiki-20181120-pages-meta-current10.txt-p2336425p3046511"
outdir = "./data/chunks_wiki/"
w_per_file = 100000 # 1m
w_limit = 1000000 # 10m

if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)

wordlist = []
with open(corpus, "r") as f:
    print("reading from",corpus,"...")
    for line in f:
        wordlist.extend(line.rstrip().split())
        print("words read:",len(wordlist), end="\r")
        if len(wordlist) >= w_limit:
            break


for wc in range (0, len(wordlist), w_per_file):
    nr= math.ceil(wc/w_per_file)
    outfile = outdir+"text_"+str(nr)+".txt"
    with open(outfile, "w") as f:
        print("writing to",outfile,"...")
        f.write(" ".join(wordlist[wc:wc+w_per_file]))



"""
corpus = 
"../ukwac_100m/ukwac_100m.txt"
"/mnt/8tera/corpora/enwiki-20181120/enwiki-20181120-pages-meta-current10.txt-p2336425p3046511"

outdir = 
"./data/chunks/"
"./data/chunks_wiki/"

"""