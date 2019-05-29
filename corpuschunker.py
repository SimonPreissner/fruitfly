import math

corpus = "data/pride.txt" # alternative: sandbox_corpus_multiline.txt
outdir = "data/chunks/"
w_per_file = 100000

wordlist = []
with open(corpus, "r") as f:
    print("reading from",corpus,"...")
    for line in f:
        wordlist.extend(line.rstrip().split())
        print("word count:",len(wordlist))

for wc in range (0, len(wordlist), w_per_file):
    nr= math.ceil(wc/w_per_file)
    outfile = outdir+"text_"+str(nr)+".txt"
    with open(outfile, "w") as f:
        print("writing to",outfile,"...")
        f.write(" ".join(wordlist[wc:wc+w_per_file]))

