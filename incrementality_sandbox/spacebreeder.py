"""
THE DOCOPT SECTION IS UNDER CONSTRUCTION

Spacebreeder: apply a full incrementality pipeline 

Usage:
  spacebreeder.py <texts> <space> <fly_file> ([setup] | [options])

Options:
  -h --help          Show this screen.
  -v --verbose       Output most important dimensions per word.
  -t=<teststeps>     test after a certain number of incremental runs
  
  --eval=<testset>  

OBACHT!
  # Use the file names for <space> and <testset> WITHOUT file extension!
from docopt import docopt
  
"""

import sys
import utils
import time
import Incrementor
from Incrementor import Incrementor
import Fruitfly
#from Fruitfly import Fruitfly
import MEN
import numpy as np


texts_location = "data/chunks"

matrix_location = "spaces/unhashed/"
fly_location = "log/configs/"
space_location = "spaces/hashed/" # for the hashed spaces

testinterval = 10
testfile = "data/sandbox_MEN_pairs"
allow_disconnection = True

"""
parameters that might be convenient:
tokenize = True
linewise = True
max_dims = None
verbose  = True
"""



"""
runtime_zero = time.time()
open directory
for file in directory:

  if this is the first run: 
    incremental = False
    fly_new = True
  else: 
    incremental = True
    fly_grow = True

  incrementor = Incrementor(all the cool parameters)
  incrementor.count_cooccurrences()

  if run%testinterval == 0:
    incrementor.log(...)
    unhashed_space = utils.readDM(...)

    hashed_space = incrementor.fruitfly.fly(unhashed_space "log")
    log(hashed_space, ...) #TODO implement this in utils?
    incrementor.fruitfly.log_params(...)

    spb = ...
    spa = ...
    log the following: [spb spa spdiff runtimes filenames]

    incrementor.fruitfly.selective_disconnect(<on the sheet>)
"""