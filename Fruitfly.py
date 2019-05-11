import time  # for logging

import numpy as np
from math import ceil
from tqdm import tqdm

from utils import timeit


# noinspection SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection
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

    def __init__(self, flattening, pn_size, kc_size, proj_size, hash_percent, max_pn_size=None, old_proj=None):
        """Create layers and random projections. For initialization, use one of the class methods"""
        self.flattening = flattening
        if self.flattening not in ["log", "log2", "log10"]:
            print("No valid flattening method for the FFA. Continuing without flattening.")
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.kc_factor = kc_size/pn_size
        self.proj_size = proj_size
        self.hash_percent = hash_percent

        self.pn_layer = np.zeros(self.pn_size) # input layer
        self.kc_layer = np.zeros(self.kc_size) 

        self.max_pn_size = max_pn_size

        # arrays of PNs that are connected to any one KC 
        self.proj_functions = old_proj if old_proj is not None else self.create_projections() 
        self.pn_to_kc = self.forward_connections([i for i in range(self.pn_size)])

    @classmethod
    def from_config(cls, filename):
        """ load parameters from a file in the log/configs folder """
        try:
            with open(filename, "r") as f:
                lines = f.readlines()
            specs = {}

            lnr = 0
            con_ind = 0 # will be set to the line number of the first connection
            paramline = True
            while paramline:
                params = lines[lnr].rstrip().split()
                try:
                    int(params[0]) # only connection lines have int-able first elements
                    paramline = False
                    con_ind = lnr
                except ValueError: # non-int-able first elements are from paramlines
                    try:
                        specs[params[0]]=int(params[1]) # converts to int if possible
                    except ValueError:
                        specs[params[0]]=params[1] # leaves string parameters as strings
                    lnr+=1

            if "max_pn_size" not in specs or specs["max_pn_size"] == "None":
                specs["max_pn_size"] = None

            connections = {}
            for line in lines[con_ind:]:
                values = line.split()
                connections[int(values[0])] = [int(v) for v in values[1:]]

            return cls(specs["flattening"], specs["pn_size"], specs["kc_size"],
                       specs["proj_size"], specs["hash_perc"],
                       max_pn_size=specs["max_pn_size"], old_proj=connections)
        except FileNotFoundError:
            print("FileNotFoundError in Fruitfly.from_config()!\n"
                  "\tcontinuing with a fruitfly from scratch (log,50,30000,6,5)!")
            return Fruitfly.from_scratch()

    @classmethod
    def from_scratch(cls, flattening="log", pn_size=50, kc_size=30000, proj_size=6, hash_percent=5, max_pn_size=None):
        """ This is a workaround for issues with the default constructor """
        return cls(flattening, pn_size, kc_size, proj_size, hash_percent, max_pn_size=max_pn_size)

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
        statement = "flattening: " +str(self.flattening)+"\t"+\
                    "pn_size: "    +str(self.pn_size)+"\t"+\
                    "kc_factor: "  +str(self.kc_factor)+"\t"+\
                    "kc_size: "    +str(self.kc_size)+"\t"+\
                    "proj_size: "  +str(self.proj_size)+"\t"+\
                    "hash-perc: "  +str(self.hash_percent)+"\t"+\
                    "max_pn_size: "+str(self.max_pn_size)
        return statement

    def get_specs(self):
        """ for in-code usage """
        return {"flattening":self.flattening,
                "pn_size":self.pn_size, 
                "kc_factor":self.kc_factor, 
                "kc_size":self.kc_size,
                "proj_size":self.proj_size, 
                "hash_percent":self.hash_percent,
                "max_pn_size":self.max_pn_size}

    def log_params(self, filename="log/configs/ff_config.cfg", timestamp=True):
        """ writes parameters and projection connections to a specified file"""
        if timestamp is True: 
            filename = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())+"_"+filename
        connections = ""
        print("\nLogging fruitfly config to",filename,"...")
        for kc,pns in tqdm(self.proj_functions.items()):
            connections+=(str(kc)+" "+" ".join([str(pn) for pn in pns])+"\n")
        with open(filename, "w") as logfile:
            logfile.write("flattening " +str(self.flattening)+"\n"+
                          "pn_size "    +str(self.pn_size)+"\n"+
                          "kc_factor "  +str(self.kc_factor)+"\n"+
                          "kc_size "    +str(self.kc_size)+"\n"+
                          "proj_size "  +str(self.proj_size)+"\n"+
                          "hash_perc "  +str(self.hash_percent)+"\n"+
                          "max_pn_size "+str(self.max_pn_size)+"\n")
            logfile.write(connections)

    def important_words_for(self, word_hash, pn_dic, n=None):
        """ 
        For every PN that is connected to an activated KC of the given 
        hash, count the number of connections from that PN to connected
        KCs.
        """
        if len(pn_dic) != self.pn_size:
            print("WARNING: in Fruitfly.important_words_for(): \
            vocabulary doesn't match PN layer!", end=" ")
            print("Make sure to call this method with the vocabulary obtained from flying! \
            Continuing with 'wrong' vocabulary")
        important_words = {} # dict of word:count_of_connections
        for i in range(len(word_hash)):
            if int(word_hash[i]) == 1:
                activated_pns = self.proj_functions[i] # retrieve transitions of an activated KC
                for pn in activated_pns: # count which word helped how many times to lead to 'word_hash'
                    w = pn_dic[pn]  # retrieve word from PN index
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

    def extend(self):
        """ don't extend if there's a limit that has been reached"""
        if self.max_pn_size is not None and self.pn_size == self.max_pn_size:
            return None
        """ add a PN to the fruitfly and connect it"""
        self.pn_size+=1
        self.pn_layer = np.append(self.pn_layer, [0])
        self.kc_factor = self.kc_size/self.pn_size 

        """number of connections from the new PN = avg. PN connectedness"""
        new_avg_pn_con = int(sum([len(p) for k,p in self.proj_functions.items()])/self.pn_size)

        """weight KCs with inverse of their respective connectednesses"""
        weighted_kcs = {}
        for cell in self.proj_functions:
            weighted_kcs[cell] = 1.0/(1+len(self.proj_functions[cell]))
            weighted_kcs[cell] = weighted_kcs[cell]*np.random.rand()
        winners = sorted(weighted_kcs, key=weighted_kcs.get, reverse=True)[:new_avg_pn_con]

        """fully connected winner KCs experience connection switching"""
        for kc in winners: # add PN to connections of the winner KCs
            if len(self.proj_functions[kc]) == self.proj_size: # full KC
                pn_con = {pn:len(self.pn_to_kc[pn]) for pn in self.proj_functions[kc]}
                # the most connected PN gets robbed
                robbed_pn = sorted(pn_con, key=pn_con.get, reverse=True)[0]
                # replace PN indices in proj_functions
                self.proj_functions[kc][self.proj_functions[kc].index(robbed_pn)] = self.pn_size-1
                # update pn_to_kc
                del self.pn_to_kc[robbed_pn][self.pn_to_kc[robbed_pn].index(kc)] 
        
            else: # make new connections
                self.proj_functions[kc].append(self.pn_size-1)

        self.pn_to_kc.update(self.forward_connections([self.pn_size-1]))

    def fit_space(self, unhashed_space, words_to_i):
        if self.max_pn_size is not None and len(words_to_i)<=self.max_pn_size:
            print("no space fitting needed.")#CLEANUP
            return unhashed_space, words_to_i, {v:k for k,v in words_to_i.items()}

        """ extract the most frequent words"""
        freq = {w:sum(vec) for w,vec in unhashed_space.items()}
        new_keys = sorted(freq, key=freq.get, reverse=True)[:self.max_pn_size]
        #print("fit_space() -- new_keys: {0} ({1})".format(new_keys, len(new_keys)))#CLEANUP
        """ delete dimensions of words that are not frequent enough"""

        fitted_space = {w:vec for w,vec in unhashed_space.items() if w in new_keys} # reduce rows
        #print("fit_space() -- fitted_space number of vectors:",len(fitted_space))#CLEANUP

        old_dims = [i for w,i in words_to_i.items() if w not in new_keys]

        for w,vec in fitted_space.items():
            fitted_space[w] = np.delete(vec,old_dims)

        #unhashed_space = np.delete(unhashed_space, old_dims, 0) # rows #CLEANUP
        #unhashed_space = np.delete(unhashed_space, old_dims, 1) # columns #CLEANUP

        new_keys.sort() # sort words alphabetically (htis sorts the space)
        new_dic = {k:new_keys.index(k) for k in new_keys} # word:index
        new_ind = {v:k for k,v in new_dic.items()} # index:word
        #print("fit_space() -- fitted_space vector length:",len(fitted_space[new_ind[0]]))#CLEANUP


        #print("fit_space() -- new_ind: {0} ({1})".format(new_ind, len(new_ind)))#CLEANUP

        return fitted_space, new_dic, new_ind


#========== FFA APPLICATION

    def flatten(self, frequency_vector):
        """ 
        make extremely frequent words (especially stopwords) less important 
        before they hit the projection algorithm
        """
        flat_vector = np.zeros(len(frequency_vector))

        if self.flattening == "log":
            for i, freq in enumerate(frequency_vector):
                flat_vector[i] = np.log(1.0+freq) # '1.0+' for values < 1
        elif self.flattening == "log2":
            for i, freq in enumerate(frequency_vector):
                flat_vector[i] = np.log2(1.0+freq)
        elif self.flattening == "log10":
            for i, freq in enumerate(frequency_vector):
                flat_vector[i] = np.log10(1.0+freq)
        else: 
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
        top = int(ceil(self.hash_percent * self.kc_size / 100)) # number of winners (highest activation)
        activated_kcs = np.argpartition(self.kc_layer, -top)[-top:]
        for cell in activated_kcs:
            kc_activations[cell] = 1 # assign 1 to the winners
        return kc_activations

    @timeit
    def fly(self, unhashed_space, words_to_i):
        """
        Hash each element of the input space. 
        Fit input space to the Fruitfly's PN layer (if necessary) by
        choosing the most frequent words as dimensions. Afterwards, apply 
        flattening before input, afterwards project, hash, and return the 
        complete hashed space.
        """

        # choose most frequent word to hash
        fitted_space, flight_dic, flight_ind = self.fit_space(unhashed_space, words_to_i)

        space_hashed = {} # a dict of word : binary_vector (= after "flying")
        print("\nStarting flying...")
        for w in tqdm(fitted_space): # iterate through space, word by word 
            self.pn_layer = self.flatten(fitted_space[w])
            self.kc_layer = self.projection()
            space_hashed[w] = self.hash_kenyon() # same dimensionality as 'kc_layer'
        return space_hashed, flight_dic, flight_ind


