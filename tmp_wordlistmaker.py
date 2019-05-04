#infile = "data/MEN_dataset_natural_form_full"
#outfile = "data/MEN_natural_vocabulary"

#infile = "data/MEN_dataset_lemma_form_full"
#outfile = "data/MEN_lemma_vocabulary"

#infile = "incrementality_sandbox/data/sandbox_MEN_pairs"
#outfile = "incrementality_sandbox/data/sandbox_MEN_vocabulary"

infile = "pipe/testset_MEN_pairs"
outfile = "pipe/testset_MEN_vocabulary"


words = []
with open(infile, "r") as f:
	for line in f:
		line = line.rstrip()
		words.extend(line.split()[:2])

voc = set(words)
print(voc)
print(len(voc))

with open(outfile, "w") as f:
	for w in voc:
		f.write(w+"\n")

print("done.")


