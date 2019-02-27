import re

"""
This script was used to compile the ukwac_100m corpus that was used to 
compile spaces of various dimensionalities.
"""

infile =      "/mnt/8tera/corpora/ukwac/ukwac_preproc_tokenised.txt"
outfile =     "/mnt/8tera/corpora/ukwac/ukwac_100m.txt"
onelinefile = "/mnt/8tera/corpora/ukwac/ukwac_100m_oneline.txt"

output_as_a_single_line = True
limit = 100000000 # 100 million

wordlist = []
reg = re.compile("_.+?\s") # to get rid of POS-tags


with open(infile, "r") as f:
    for chunk in f: 
        if (len(wordlist) < limit):
            chunk = chunk.rstrip()
            #print(len(wordlist))
            
            raw_words = re.sub(reg, " ",chunk)
            raw_lc_words = raw_words.lower()
            words = raw_lc_words.split()
            
            wordlist.extend(words)
            
        else:
            print("words in total:", len(wordlist))
            break

if(output_as_a_single_line is True):
    outfile = onelinefile
    sep = " "
else: 
    sep = "\n"
    
print("writing to file now...")
with open(outfile, "w") as f:
    f.write(sep.join(wordlist[:limit]))