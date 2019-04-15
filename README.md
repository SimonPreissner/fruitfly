# Implementing the fruit fly's similarity hashing

This is an implementation of the random indexing method described by Dasgupta et al (2017) in [A neural algorithm for a fundamental computing problem](http://science.sciencemag.org/content/358/6364/793/tab-figures-data).

The aim of this project is to create an architecture for distributional semantics that grows in parallel to the data that is available. The incrementality that we aim for will enable us to create distributional semantic spaces from growing text corpora at any stage, independently from their current size. 

### Description of the data

The data/ directory contains several semantic spaces:

- one from the British National Corpus, containing lemmas expressed in 4000 dimensions. (BNC-MEN)
- one from a subset of Wikipedia, containing words expressed in 1000 dimensions. (wiki_all)
- one from a subset of UKWAC, containing words expressed in 1000 dimensions. (ukwac_1k_GS-checked)

The cells in the BNC and the Wikipedia space are normalised co-occurrence frequencies *without any additional weighting* (PMI, for instance); the cells in the UKWAC space are raw frequency counts. The UKWAC space contains only words that are in the test set (see below).

The directory also contains test pairs from the [MEN similarity dataset](https://staff.fnwi.uva.nl/e.bruni/MEN), both in lemmatised and natural forms.

### Running the fruit fly code with projection.py
To create a Fruitfly, let it run over a space, and evaluate it, use this script.
For detailed usage description, run it with the `--help` option:
```
python3 projection.py --help
```
A standard run might look like this (hash the raw count wikipedia vectors and test on he BNC space. Flatten the inputs with logarithm and use a fruit fly with 8000 Kenyon cells (=KCs), 6 projections leading to each KC, and 5% hashing (or, 'reduction')): 
```
python3 projection.py data/wiki_abs-freq data/MEN_dataset_natural_form_full -f log -k 8000 -p 6 -r 5
```

The program returns the Spearman correlation with the MEN similarity data, as calculated a) from the raw frequency space; and b) after running the fly's random projections, as well as the difference between the two.
Except for input and test space, all parameters are optional and set to default values.

If you want to set a baseline and evaluate a semantic space without "flying", use the `eval-only` option (fruit fly parameters are not important here):
```
python3 projection.py data/BNC-MEN eval-only
```
Finally, with the -v flag ("verbose"), the script prints out the projection neurons that are most responsible for the activation in the Kenyon layer.
```
    python3 projection.py data/BNC-MEN data/MEN_dataset_lemma_form_full -v
```

### The algorithm itself: Fruitfly.py
The Fruitfly class implements the Fruitfly algorithm and utilities for logging and inspecting any one fruitfly instance. 
 
`Fruitfly.from_scratch()` instantiates a new fruitfly,  
`Fruitfly.log_params()`  outputs a fruitfly to a .cfg file, and with  
`Fruitfly.from_config()`, you can re-use the exact same net (including all projection connections) later on.  
`Fruitfly.fly()` implements the actual application of the algorithm (hashing a distributional space).  
`Fruitfly.extend()` implements the incremental feature of this project's variant of the Fruitfly algorithm.  


### Tuning hyperparameters with hyperopt.py
To perform a grid search on ranges of the fruitfly's hyperparameters, use hyperopt.py; For detailed usage description, run it without parameters:
```
python3 hyperopt.py
```
You will need to specify a corpus (the same options as in projection.py are available). Hyperopt.py will create files with the test result, one for each run, and a final summary. It's recommended to specify the directory for these files.
Other parameters you can provide:
- flattening options (as simple arguments)
- factor of expansion from PN layer to KC layer
- number of connecitons going to each KC
- percentage of KCs to be selected as "winners"

All parameters have default ranges, so if you run 
```
python3 hyperopt.py bnc -logto data/potato log log2 -kc 2 8 2 -hash 5 15 3
```
the script will perform a grid search for parameters on the BNC space with 2 parameters for flattening, 4 for the KC-factor, 5 for the hashing, and the default ranges for projections (again, to see the default ranges, run the script without parameters). You will find a new directory called `potato` with logs of all runs.  

There is also an option to run in verbose mode, by adding the flag `-v`  

Be careful with large numbers for the ranges of -kc and -proj! These parameters can quickly inflate the performance time.


### Compiling semantic spaces with countwords.py
Part of the project is to be able to integrate new data into an already-processed corpus. For now, `countwords.py` takes a text coprus and compiles a co-occurrence matrix of words. Run it with the `--help` option for details.  
The most important parameters are 
- `-d`, which limits the dimensionality of the resulting space
- `-w`, which specifies the range (before and after a word) that is used for co-occurrence counting 
- `-x`, which checks the word overlap of a specified file with the resulting space

For example, `space_1k_dims.dm` (and `.cols`) can be compiled from `ukwac_100m.txt` (which is not available in this repository) with the following command:
```
python3 countwords.py ukwac_100m.txt space_1k_dims -d 1000 -w 5 -x data/MEN_dataset_natural_form_full
```


### The fruitfly's work: storing dense, locality-sensitive hashes
Applying the FFA to a cooccurrence cound is called "flying". This happens with `Fruitfly.fly()`, which returns a dictionary of words to their binary hashes. 
In order to store these hashes in a dense format and re-convert stored hashes into the binary form, there are three functions in the `utils` package:
- `writeDH()`: converts the binary hashes to dense hashes and writes them to a .dh file.
- `readDH()`: reads a .dh file and returns the dense hashes from it.
- `sparsifyDH()`: returns the sparse representation of dense hashes.



### Going incremental
Countwords.py also implements incrementality for both cooccurrence counting and the FFA: if specified, a Fruitfly "grows" alongside counting by creating nodes and connections whenever a new word is observed. 

Incrementality is specified by setting the `-i` flag, which uses the specified output space file as input. Fruitflies can be stored as well. 

Initially, you will want to set up a space (and if you want to grow a fruitfly, also a fruitfly) with a command similar to this:
```
python3 countwords.py textsource.txt spacefile --grow-fly new ff_config.txt -v 
```
Note that `spacefile` has no file extension, but the other files (`textsource.txt` and `ff_config.txt`) have. Also, there is no incrementality flag. The `-v` flag lets you observe the progress while the program is running.

In order to build on the space and the fruitfly, you can use
```
python3 countwords.py newtextsource.txt spacefile --grow-fly ff_config.txt -i -v
```

Happy flying!
