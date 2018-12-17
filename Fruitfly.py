import sys
import utils
import MEN
import numpy as np

class Fruitfly:
    """
    This class contains all the architecture, but not (yet) the methods; 
    those are in 'projection.py', It sets up the layers, the connections,
    has functionalities to flatten the input and show its parts, but 
    does not handle the data input, the hashing itself, or the evaluations.
    """

    def __init__(self, pn_size, kc_size, proj_size, hash_percent):
        '''Create layers and random projections'''
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.proj_size = proj_size
        self.hash_percent = hash_percent

        self.pn_layer = np.zeros(pn_size) # input layer
        self.kc_layer = np.zeros(kc_size)
        self.proj_functions = self.create_projections() # contains arrays of connections to any given kenyon cell

    def create_projections(self):
        #print("Creating",KC_size,"random projections...")
        proj_functions = {}
        for cell in range(self.kc_size):
            activated_pns = np.random.randint(self.pn_size, size=self.proj_size)
            proj_functions[cell] = activated_pns
        return proj_functions




    def show_off(self):
        """ for command line output """
        statement = "pn_size: "+str(self.pn_size)+"\t"+\
                    "kc_size: "+str(self.kc_size)+"\t"+\
                    "proj_size: "+str(self.proj_size)+"\t"+\
                    "hash-perc: "+str(self.hash_percent)
        return statement

    def get_specs(self):
        """ for in-code usage """
        return {"pn_size":self.pn_size, 
                "kc_size":self.kc_size, 
                "proj_size":self.proj_size, 
                "hash_percent":self.hash_percent}

    def show_projections(self, w, hw, i_to_cols):
        important_words = {} # dict of word:number
        for i in range(len(hw)):
            if hw[i] == 1:
                activated_pns = self.proj_functions[i] # retrieve transitions of an activated KC
                #print(w,[i_to_cols[pn] for pn in activated_pns])
                for pn in activated_pns: # count which word helped how many times to lead to 'hw'
                    w = i_to_cols[pn] # retrieve word from PN index
                    if w in important_words:
                        important_words[w]+=1
                    else:
                        important_words[w]=1
        print(w,"BEST PNS", sorted(important_words, key=important_words.get, reverse=True)[:self.proj_size]) # only print the most important words




    def flatten(self, frequency_vector, method): 
        """ 
        make extremely frequent words (especially stopwords) less important 
        before they hit the projection algorithm
        """
        if method == "log":
            for i, freq in enumerate(frequency_vector):
                frequency_vector[i] = np.log(1.0+freq) # add 1 to make sure that no value is below 1
        elif method == "sigmoid":
            for i, freq in enumerate(frequency_vector):
                frequency_vector[i] = 1.0 / (1 + np.exp(-freq))
        elif method == "softmax":
            exp_vector = np.exp(frequency_vector)
            return exp_vector/exp_vector.sum(0)
        else: 
            print("No flattening method specified. Continuing without flattening.")
        return frequency_vector

    def projection(self):
        """ for each KC, sum up the values of the PNs that have a connection to this KC """
        kc_layer = np.zeros(self.kc_size)
        for cell in range(self.kc_size):
            activated_pns = self.proj_functions[cell] # array of the connected cells
            for pn in activated_pns:
                kc_layer[cell]+=self.pn_layer[pn] # sum the activation values of the pn nodes in the kc
                #kc_layer[cell]+=np.log(1+3000*pn_layer[pn]) # direct flattening with intuitively chosen factor
        return kc_layer

    def hash_kenyon(self):
        """ choose the most activated KCs, set them to 1 and the rest to 0 """
        kc_activations = np.zeros(self.kc_size)
        top = int(self.hash_percent * self.kc_size / 100) # number of winners (highest activation)
        activated_kcs = np.argpartition(self.kc_layer, -top)[-top:]
        for cell in activated_kcs:
            kc_activations[cell] = 1 # assign 1 to the winners
        return kc_activations

    def fly(self, unhashed_space, flattening):
        """
        Hash each element of the input space. Apply flattening before input, 
        afterwards project, hash, and return the complete hashed space.
        """
        space_hashed = {} # a dict of word : binary_vector (= after "flying")
        for w in unhashed_space: # iterate through dictionary 
            self.pn_layer = self.flatten(unhashed_space[w], flattening)# flatten before hitting the PNs
            self.kc_layer = self.projection()
            space_hashed[w] = self.hash_kenyon() # same dimensionality as 'kc_layer'
        return space_hashed