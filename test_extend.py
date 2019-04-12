import os
import time
import utils
import numpy as np
from Fruitfly import Fruitfly



"""
# TEST_01
# make a fruitfly, save it, expand it and save each expansion state.

log_dest = "test_cfgs/"
if not os.path.isdir(log_dest):
    os.makedirs(log_dest, exist_ok=True)

fliege = Fruitfly.from_scratch(pn_size=10, kc_size=100, proj_size=6, hash_percent=5)
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

fliege = Fruitfly.from_scratch(pn_size=2, kc_size=10000, proj_size=6, hash_percent=5)

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



# TEST_04
# this tests the utilitymethods concerning saving and loading a hashed space.
# writeDH(), readDH(), and sparsifyDH()

unhashed_space = utils.readDM("data/BNC-MEN.dm") # returns dict of word : word_vector
i_to_cols, cols_to_i = utils.readCols("data/BNC-MEN.cols") # returns both-ways dicts of the vocabulary (word:pos_in_dict); important for maintenances

fruitfly = Fruitfly.from_scratch(4000, 1000, 6, 5)
space_hashed, t_flight = fruitfly.fly(unhashed_space, "log") # a dict of word : binary_vector (= after "flying")
print("coin_N\n\n",space_hashed["coin_N"][:100],"\n") # TARGET

utils.writeDH(space_hashed, "testwrite.dh")
loaded_hashes = utils.readDH("testwrite.dh")
print(loaded_hashes["coin_N"][:100]) # outputs a dense representation

sparse_space = utils.sparsifyDH(loaded_hashes, 1000) # is the same as TARGET
print(sparse_space["coin_N"][:100])
"""
"""


