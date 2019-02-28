import re

"""
This script was used to extract from the ukwac_100m_oneline corpus
those passages that contain the words in the sandbox test corpus.
The product of this script will be used as corpus within the sandbox.
"""

#========== FUNCTIONS

def read_corpus(filepath):
    words = []
    nonword = re.compile("\W+") # to delete punctuation entries
    with open(filepath) as f:
        for text in f:
            tokens = text.rstrip().lower().split()
            for t in tokens:
                if (re.fullmatch(nonword, t) is None): # ignore punct.
                    words.append(t)
                    if len(words)%1000000 == 0:
                        print("words read:",len(words))
    return words

def make_checkset(filepath):
    checklist = [] # the contexts of these words will become the corpus
    with open(filepath, "r") as f:
        for line in f:
            words = line.rstrip().split()[:2]
            checklist.extend(words)
    return set(checklist)

def write_to(filepath, corpus, multiline=False):
    print("writing to",filepath,"...")
    with open(filepath, "w") as f:
        if multiline is False:
            tokens = [w for l in corpus for w in l] # "flattens" to a list
            outstring = " ".join(tokens)
            print("writing",len(tokens)," words to",filepath,"...")
        else:
            lines = [" ".join(l) for l in corpus]
            outstring = "\n".join(lines)
            print("writing",len(lines),"lines to",filepath,"...")

        f.write(outstring)



#========== EXECUTIVE CODE

original_corpus = "../../ukwac_100m/ukwac_100m_oneline.txt"
#original_corpus = "../data/pride.txt" # for test purposes
checklist_filepath = "data/sandbox_MEN_pairs"
outfile = "data/sandbox_corpus.txt"


window = 10 # size of context to one side 
multi_line_corpus_wanted = False

source_words = read_corpus(original_corpus)
source_words_size = len(source_words)
checkset = make_checkset(checklist_filepath)

corpus = []
print("searching for occurences in the source corpus ...")
for i in range(window, source_words_size-window): # avoid index problems
    if source_words[i] in checkset:
        corpus.append(source_words[i-window:i+window]) # list of lists
    if i%(source_words_size/500) == 0:
        print(round(float(i*500)/source_words_size, 2),"percent done")


write_to(outfile, corpus, multi_line_corpus_wanted)




