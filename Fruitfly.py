import sys
import time # for logging
import MEN
import utils
from utils import timeit
from tqdm import tqdm
from math import ceil
import numpy as np

class Fruitfly:
    """
    This class contains all the architecture and methods of the FFA:
    - PN layer, KC layer, projection connections
    - input flattening, projection, hashing
    - output and configuration logging
    - dynamic growth of PN layer and connections
    It does not implement any cooccurrence counting or evaluation.
    """


#========== CONSTRUCTORS AND SETUP

    def __init__(self, pn_size, kc_size, proj_size, hash_percent, old_proj=None):
        '''Create layers and random projections. For initialization, use one of the classmethods'''
        self.pn_size   = pn_size
        self.kc_size   = kc_size
        self.kc_factor = kc_size/pn_size
        self.proj_size = proj_size
        self.hash_percent = hash_percent

        self.pn_layer = np.zeros(self.pn_size) # input layer
        self.kc_layer = np.zeros(self.kc_size) 

        # arrays of PNs that are connected to any one KC 
        self.proj_functions = old_proj if old_proj is not None else self.create_projections() 
        self.pn_to_kc = self.forward_connections([i for i in range(self.pn_size)])

    @classmethod
    def from_config(cls, filename):
        """ load parameters from a file in the log/configs folder """
        try:
            with open(filename, "r") as f:
                lines = f.readlines()

            specs = {p[0]:int(p[1]) for p in [l.split() for l in lines[:1]+lines[2:5]]}

            connections = {}
            for line in lines[5:]:
                values = line.split()
                connections[int(values[0])] = [int(v) for v in values[1:]]

            return cls(specs["pn_size"],specs["kc_size"],specs["proj_size"],specs["hash_perc"], connections)
        except FileNotFoundError as e:
            print("FileNotFoundError in Fruitfly.from_config()!\n"\
                  "\tcontinuing with a fruitfly from scratch (50,30000,6,5)!")
            return from_scratch()

    @classmethod
    def from_scratch(cls, pn_size=50, kc_size=30000, proj_size=6, hash_percent=5):
        """ This is a workaround for issues with the default constructor """
        return cls(pn_size, kc_size, proj_size, hash_percent)

    def create_projections(self):
        proj_functions = {}
        print("\nCreating new projections...")

        for cell in range(self.kc_size):
            # uniform random choice + maximally 1 connection per PN-KC pair
            activated_pns = list(set(np.random.randint(self.pn_size, size=self.proj_size)))
            proj_functions[cell] = activated_pns

        return proj_functions

    def forward_connections(self, pn_indices): 
        pn_indices = [pn_indices] if type(pn_indices) != list else pn_indices

        pn_to_kc = {pn:[] for pn in pn_indices} # { pn_index : [connected KCs] }
        for kc,connections in self.proj_functions.items():
            for pn in pn_indices: # only for the PNs given to the method!
                if pn in connections:
                    pn_to_kc[pn].append(kc)

        return pn_to_kc



#========== STRINGS AND LOGGING
            
    def show_off(self):
        """ for command line output """
        statement = "pn_size: "  +str(self.pn_size)+"\t"+\
                    "kc_factor: "+str(self.kc_factor)+"\t"+\
                    "kc_size: "  +str(self.kc_size)+"\t"+\
                    "proj_size: "+str(self.proj_size)+"\t"+\
                    "hash-perc: "+str(self.hash_percent)
        return statement

    def get_specs(self):
        """ for in-code usage """
        return {"pn_size":self.pn_size, 
                "kc_factor":self.kc_factor, 
                "kc_size":self.kc_size,
                "proj_size":self.proj_size, 
                "hash_percent":self.hash_percent}

    def log_params(self, filename="log/configs/ff_config.cfg", timestamp=True):
        """ writes parameters and projection connections to a specified file"""
        if timestamp is True: 
            filename = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())+"_"+filename
        connections = ""
        print("\nLogging fruitfly config to",filename,"...")
        for kc,pns in tqdm(self.proj_functions.items()):
            connections+=(str(kc)+" "+" ".join([str(pn) for pn in pns])+"\n")
        with open(filename, "w") as logfile:
            logfile.write("pn_size "  +str(self.pn_size)+"\n"+\
                          "kc_factor "+str(self.kc_factor)+"\n"+\
                          "kc_size "  +str(self.kc_size)+"\n"+\
                          "proj_size "+str(self.proj_size)+"\n"+\
                          "hash_perc "+str(self.hash_percent)+"\n")
            logfile.write(connections)
        logfile.close()

    def important_words_for(self, word_hash, i_to_cols, n=None):
        """ 
        For every PN that is connected to an activated KC of the given 
        hash, count the number of connections from that PN to connected
        KCs.
        """
        important_words = {} # dict of word:count_of_connections
        for i in range(len(word_hash)):
            if word_hash[i] == 1:
                activated_pns = self.proj_functions[i] # retrieve transitions of an activated KC
                for pn in activated_pns: # count which word helped how many times to lead to 'word_hash'
                    w = i_to_cols[pn] # retrieve word from PN index
                    if w in important_words:
                        important_words[w]+=1 
                    else:
                        important_words[w]=1
        count_ranked = sorted(important_words, key=important_words.get, reverse=True)
        if n is None: 
            return count_ranked # only print the most important words
        else:
            return count_ranked[:n]



#========== INCREMENTALITY

    def extend(self): #TODO 'num_proj' needed?
        """ add a PN to the fruitfly and connect it"""
        self.pn_size+=1
        self.pn_layer = np.append(self.pn_layer, [0])
        self.kc_factor = self.kc_size/self.pn_size 

        """number of connections from the new PN = avg. PN connectedness"""
        new_avg_pn_con = int(sum([len(p) for k,p in self.proj_functions.items()])/(self.pn_size))

        """weight KCs with inverse of their respective connectednesses"""
        weighted_KCs = {}
        for cell in self.proj_functions:
            weighted_KCs[cell] = 1.0/(1+len(self.proj_functions[cell]))
            weighted_KCs[cell] = weighted_KCs[cell]*np.random.rand()
        winners = sorted(weighted_KCs, key=weighted_KCs.get, reverse=True)[:new_avg_pn_con]

        """fully connected winner KCs experience connection switching"""
        for kc in winners: # add PN to connections of the winner KCs
            if len(self.proj_functions[kc]) == self.proj_size: # full KC
                pn_con = {pn:len(self.pn_to_kc[pn]) for pn in self.proj_functions[kc]}
                # the most connected PN gets robbed
                robbed_pn = sorted(pn_con, key=pn_con.get, reverse=True)[0]
                # replace PN indices in peroj_functions
                self.proj_functions[kc][self.proj_functions[kc].index(robbed_pn)] = self.pn_size-1
                # update pn_to_kc
                del self.pn_to_kc[robbed_pn][self.pn_to_kc[robbed_pn].index(kc)] 
        
            else: # make new connections
                self.proj_functions[kc].append(self.pn_size-1)

        self.pn_to_kc.update(self.forward_connections([self.pn_size-1]))



#========== FFA APPLICATION

    def flatten(self, frequency_vector, method=None): 
        """ 
        make extremely frequent words (especially stopwords) less important 
        before they hit the projection algorithm
        """
        flat_vector = np.zeros(len(frequency_vector))
        if (method == "log"):
            for i, freq in enumerate(frequency_vector):
                flat_vector[i] = np.log(1.0+freq) # '1.0+' for values < 1
        elif (method == "log2"):
            for i, freq in enumerate(frequency_vector):
                flat_vector[i] = np.log2(1.0+freq)
        elif (method == "log10"):
            for i, freq in enumerate(frequency_vector):
                flat_vector[i] = np.log10(1.0+freq)
        else: 
            print("No valid flattening method specified. Continuing without flattening.")
            return frequency_vector
        return flat_vector

    def projection(self):
        """ for each KC, sum up the values of the PNs that have a connection to this KC """
        kc_layer = np.zeros(self.kc_size)
        for cell in range(self.kc_size):
            activated_pns = self.proj_functions[cell] # PNs connected to the KC
            for pn in activated_pns:
                kc_layer[cell]+=self.pn_layer[pn] # sum activation of the PNs for the KC
        return kc_layer

    def hash_kenyon(self):
        """ choose the most activated KCs, set them to 1 and the rest to 0 """
        kc_activations = np.zeros(self.kc_size)
        top = int(ceil(self.hash_percent * (self.kc_size) / 100)) # number of winners (highest activation)
        activated_kcs = np.argpartition(self.kc_layer, -top)[-top:]
        for cell in activated_kcs:
            kc_activations[cell] = 1 # assign 1 to the winners
        return kc_activations

    @timeit
    def fly(self, unhashed_space, flattening=None):
        """
        Hash each element of the input space. Apply flattening before input, 
        afterwards project, hash, and return the complete hashed space.
        """
        space_hashed = {} # a dict of word : binary_vector (= after "flying")
        print("\nStarting flying...")
        for w in tqdm(unhashed_space): # iterate through space, word by word 
            self.pn_layer = self.flatten(unhashed_space[w], flattening)
            self.kc_layer = self.projection()
            space_hashed[w] = self.hash_kenyon() # same dimensionality as 'kc_layer'
        return space_hashed


