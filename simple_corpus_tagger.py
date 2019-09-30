"""
This script POS-tags a corpus and simplifies the tags to represent
only 4 classes: Noun (N), Verb (V), Adjective (J), Other (X)
"""

import os
import nltk
import utils
from tqdm import tqdm

#corpusfile =    "../ukwac_100m/ukwac_100m_oneline.txt" #"data/potato.txt" #CLEANUP
indir = "data/chunks_small" #"data/pride.txt"
outdir = "data/chunks_small_pos" #"test/pride_postagged.txt"
#tagged_corpus = "/mnt/8tera/shareclic/fruitfly/ukwac_100m_tok-tagged.txt" #"test/postag_potato.txt" #CLEANUP

filepaths = {}
if os.path.isfile(indir):  # for a single file that is passed
    filepaths[indir] = os.walk(indir)[0]+"/"+os.walk(indir)[2]
else:
    for (dirpath, dirnames, filenames) in os.walk(indir):
        for fn in filenames:
            filepaths[fn] =(dirpath + "/" + fn)

for filename, filepath in filepaths.items():
    lines = []
    with open(filepath, "r") as f:
        print("reading and tokenizing corpus from",filepath,"...")
        for line in tqdm(f):
            words = nltk.word_tokenize(line.rstrip())
            lines.append(words)
    with open(outdir+"/"+filename, "w") as f:
        print("POS-tagging and writing corpus to",filepath,"...")
        for line in tqdm(lines):
            tagged = nltk.pos_tag(line) # list of tuples (word, tag)
            simply_tagged = utils.simplify_postags(tagged)
            f.write(" ".join(simply_tagged)+"\n")

print("done.")



