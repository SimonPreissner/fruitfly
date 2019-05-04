import os
import time
import utils
import MEN
import numpy as np
from Fruitfly import Fruitfly
from Incrementor import Incrementor



"""
# TEST_01
# make a fruitfly, save it, expand it and save each expansion state.

log_dest = "test_cfgs/"
if not os.path.isdir(log_dest):
    os.makedirs(log_dest, exist_ok=True)

fliege = Fruitfly.from_scratch(flattening="log", pn_size=10, kc_size=100, proj_size=6, hash_percent=5)
print("initial fly:",fliege.show_off())
fliege.log_params(filename=log_dest+"run-0.txt", timestamp=False)

for i in range(5):
	print("RUN",i+1)
	fliege.extend()
	fliege.show_off()
	fliege.log_params(filename=log_dest+"run-"+str(i+1)+".txt", timestamp=False)
"""



"""
# TEST_02
# load fruitfly, expand it, save it 

log_dest = "test_cfgs/"
if not os.path.isdir(log_dest):
    os.makedirs(log_dest, exist_ok=True)

fliege = Fruitfly.from_config(log_dest+"run-0.txt")
print("\tFliege 1:",fliege.show_off())

fliege.extend()
print("\tFliege 2:",fliege.show_off())
fliege.log_params(filename=log_dest+"run-01.txt", timestamp=False)
"""



"""
# TEST_03
# extend fruitfly several times, see how the connectedness of the PNs
# varies (e.g., compute avg. and variance every 100th extension)

log_dest = "test_cfgs/"
if not os.path.isdir(log_dest):
    os.makedirs(log_dest, exist_ok=True)

fliege = Fruitfly.from_scratch(flattening="log", pn_size=2, kc_size=10000, proj_size=6, hash_percent=5)

loopstart = time.time()
while fliege.pn_size < 20:
	t1 = time.time()
	fliege.extend()
	t2 = time.time()

	if fliege.pn_size%1 == 0:
		connections = fliege.pn_to_kc
		con_dist = [len(con) for pn,con in connections.items()]
		#con_dic = {pn:len(con) for pn,con in connections.items()}

		average  = round(np.average(con_dist),4) # should decrease from 600 to 24
		variance = round(np.var(con_dist,ddof=1),4) # should converge to 0
		std_dev  = round(np.std(con_dist,ddof=1),4)

		print("extend time:",round(t2-t1,6),"pn_size:",str(fliege.pn_size)+"   \tstats (avg, var, std):  ", average," ", variance," ", std_dev)
		with open("test_cfgs/extension_stats_cp.tsv","a") as f:
			f.write(str(round(t2-t1,6))+"\t"+str(fliege.pn_size)+"\t"+str(average)+"\t"+str(variance)+"\t"+str(std_dev)+"\n")

print("total runtime:",round(time.time()-loopstart,5),"seconds)")
"""



"""
# TEST_04
# this tests the utility methods concerning saving and loading a hashed space.
# writeDH(), readDH(), and sparsifyDH()

unhashed_space = utils.readDM("data/BNC-MEN.dm") # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols("data/BNC-MEN.cols") # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances

fruitfly = Fruitfly.from_scratch("log", 4000, 1000, 6, 5)
space_hashed, t_flight = fruitfly.fly(unhashed_space) # a dict of word : binary_vector (= after "flying")
print("coin_N\n\n",space_hashed["coin_N"][:100],"\n") # TARGET

utils.writeDH(space_hashed, "testwrite.dh")
loaded_hashes = utils.readDH("testwrite.dh")
print(loaded_hashes["coin_N"][:100]) # outputs a dense representation

sparse_space = utils.sparsifyDH(loaded_hashes, 1000) # is the same as TARGET
print(sparse_space["coin_N"][:100])
"""


"""
# TEST_05
# this tests Fruitfly.fit_space()
# 


# TEST_05-1
firstfly = Fruitfly.from_scratch("log", pn_size=10, kc_size=200, proj_size=6, hash_percent=5, max_pn_size=30)
firstfly.log_params(filename="data/testfly.cfg",timestamp=False)
print("TEST_05-1 passed!")

# TEST_05-2
fruitfly = Fruitfly.from_config("data/testfly.cfg")
fruitfly.show_off()
print("TEST_05-2 passed!")


unhashed_space = utils.readDM("data/potato.dm") # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols("data/potato.cols") # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances

# TEST_05-3
(space_hashed, space_dic, space_ind), t_flight = fruitfly.fly(unhashed_space, cols_to_i)

#print(space_hashed)
#print(space_dic)
print("TEST_05-3 passed!")

# TEST_05-4
for w in sorted(space_dic, key=space_dic.get): # prints sorted by index
	print(w,"--",space_dic[w])
for i in sorted(space_ind, key=space_ind.get): # prints sorted alphabetically
	print(i,"--",space_ind[i])

utils.writeDH(space_hashed, "data/testhashed.dh")
print("TEST_05-4 passed!")

# check dimensionality (30): passed
# check shape: (m,m)? (30,30) : passed
# check space_dic and space_ind : passed
"""



"""
# TEST_06
# this tests expanding the matrix over the PN_limit of the fruitfly, 
# afterwards hashing the space -- or rather, the PN_limit most frequent words.

firstfly = Fruitfly.from_scratch("log", pn_size=10, kc_size=200, proj_size=6, hash_percent=5, max_pn_size=30)
firstfly.log_params(filename="data/testfly.cfg",timestamp=False)

#===== incremental work
incro = Incrementor("data/pride.txt", "data/testmatrix", \
        corpus_tokenize=True, corpus_linewise=False, corpus_checkvoc=None, \
        matrix_incremental=False, matrix_maxdims=None, \
        fly_new=False, fly_grow=True, fly_file="data/testfly.cfg", fly_max_pn=None, \
        verbose=True)

incro.count_cooccurrences()
incro.log_matrix()

#===== flying work
unhashed_space = utils.readDM(incro.outspace)
i_to_words, words_to_i = utils.readCols(incro.outcols) # optional; later, you could also use incro.words_to_i

(hashed_space, space_dic, space_ind), t_flight = \
	incro.fruitfly.fly(unhashed_space, words_to_i)

#===== evaluation work
spb,tsb = MEN.compute_men_spearman(unhashed_space, "data/MEN_dataset_natural_form_full")
spa,tsa = MEN.compute_men_spearman(hashed_space, "data/MEN_dataset_natural_form_full")
sp_diff = spa-spb

print("before: {0}\t({1} items)\nafter: {2}\t({3} items)\n\
	difference: {4}".format(spb, tsb, spa, tsa, sp_diff)) 

# tsb and tsa should be different: passed

"""






