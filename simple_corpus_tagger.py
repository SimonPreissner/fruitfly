"""
This script POS-tags a corpus and simplifies the tags to represent
only 4 classes: Noun (N), Verb (V), Adjective (J), Other (X)
"""

import nltk
from tqdm import tqdm

postags = {"N" :["NN", "NNS", "NNP", "NNPS"],
           "V" :["VB", "VBD", "VBG", "VBN", "VBZ", "VBP"],
           "J" :["JJ", "JJR", "JJS"]}

corpusfile =    "/mnt/8tera/shareclic/fruitfly/ukwac_100m.txt" #"data/potato.txt" #CLEANUP
tagged_corpus = "/mnt/8tera/shareclic/fruitfly/ukwac_100m_tok-tagged.txt" #"test/postag_potato.txt" #CLEANUP

def simplify_postags(tagged_words):
    simplified = []
    for w, t in tagged_words:
        if t in postags["N"]:
            simplified.append("_".join([w, "N"]))
        elif t in postags["V"]:
            simplified.append("_".join([w, "V"]))
        elif t in postags["J"]:
            simplified.append("_".join([w, "J"]))
        else:
            simplified.append("_".join([w, "X"]))
    return simplified

lines = []
with open(corpusfile, "r") as f:
    print("reading and tokenizing corpus from",corpusfile,"...")
    for line in tqdm(f):
        words = nltk.word_tokenize(line.rstrip())
        lines.append(words)

with open(tagged_corpus, "w") as f:
    print("POS-tagging and writing corpus to",tagged_corpus,"...")
    for line in tqdm(lines):
        tagged = nltk.pos_tag(line) # list of tuples (word, tag)
        simply_tagged = simplify_postags(tagged)
        f.write(" ".join(simply_tagged)+"\n")

print("done.")



