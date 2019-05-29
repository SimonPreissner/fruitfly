import math

corpus = "../ukwac_100m/ukwac_100m.txt" # alternative: sandbox_corpus_multiline.txt
outdir = "data/chunks/"
w_per_file = 1000000 # 1m
w_limit = 10000000 # 10m

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

