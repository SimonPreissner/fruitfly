"""
This script was used to select 20 MEN-data set pairs as test set for
a sandbox (to explore incrementality)
"""

menfile = "../data/MEN_dataset_natural_form_full"

wordlist = []
pairs = []
with open(menfile, "r") as f:
    for line in f:
        words = line.rstrip().split()[:2]
        pairs.append(words)
        wordlist.extend(words)

freq = {}
for w in wordlist:
	if w in freq:
		freq[w] += 1
	else: 
		freq[w] = 1

most_frequent = sorted(freq, key=freq.get, reverse=True)[:20]
print("number of words:",len(freq))
print("most frequent words:")
for w in most_frequent:
	print(freq[w],"\t",w)


# do a cross-search for the most frequent words: do they form pairs?
most_frequent_pairs = []
for p in pairs:
	if p[0] in most_frequent and p[1] in most_frequent:
		most_frequent_pairs.append(p)

mfp_sorted = sorted(most_frequent_pairs)
print("\npairs with these words:")
for p in mfp_sorted:
	print(p[0],"\t",p[1])