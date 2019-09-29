"""
This is the full pipeline of the incremental Fruitfly.
It reads a corpus, grows a Fruitfly alongside,
then applies the FFA to the counts,
optionally runs Word2Vec on the same corpus,
evaluates the results, and logs them.
"""

import os
import time

import math
import numpy as np
from tqdm import tqdm

import MEN
import utils
from Fruitfly import Fruitfly
from Incrementor import Incrementor


global errorlog, corpus_dir, w2v_exe_file, testset_file, overlap_file,\
       pipedir, results_summary_file, results_location, w2v_results_file,\
       matrix_filename, fly_location, space_location, vip_words_location,\
       w2v_corpus_file, w2v_space_location, number_of_vip_words, test_interval_in_words,\
       pns, kcs, con, red, flat, max_pns, tokenize, postag_simple, verbose, window #TODO is this really needed? is it dangerous?



errorlog = "spacebreeder_errorlog.txt"

#========== PARAMETER INPUT
#try:

print("=== Resources and Output Directories ===")
s = input("Path to text resources (default: data/chunks):") #TODO for final version, set the path to the chunks directory
corpus_dir = s if len(s) > 0 else "data/chunks" # "/mnt/8tera/shareclic/fruitfly/ukwac_100m_tok-tagged.txt" # "../ukwac_100m/ukwac_100m.txt" # "test/pride_postagged.txt" #CLEANUP
s = input("Path to word2vec code (optional):")
w2v_exe_file = s if len(s) > 0 else None # "./../share/word2vec" #TODO clean up this path (i.e. no if/else)
testset_file = input("Path to test set:")# "data/MEN_dataset_lemma_form_full" # "./data/MEN_dataset_natural_form_full" #TODO set correct path
s = input("Path to a word list to be checked for overlap (optional):")
overlap_file = s if len(s) > 0 else None #"./data/MEN_lemma_vocabulary" # "./data/MEN_natural_vocabulary" #TODO clean up this path (i.e. no if/else)
s = input("Path to the resulting directories (default: ../fly_out):")
pipedir = s if len(s) > 0 else "/mnt/8tera/shareclic/fruitfly/pipe_postagged/" #TODO set correct default path

results_summary_file = pipedir+"/"+"summary.tsv"
results_location     = pipedir+"/"+"ffa/results/stats/"
w2v_results_file     = pipedir+"/"+"w2v/results.txt"

matrix_filename      = pipedir+"/"+"ffa/cooc"
fly_location         = pipedir+"/"+"ffa/configs/"
space_location       = pipedir+"/"+"ffa/spaces/"
vip_words_location   = pipedir+"/"+"ffa/results/words/"

w2v_corpus_file      = pipedir+"/"+"w2v/resource.txt"
w2v_space_location   = pipedir+"/"+"w2v/"

for f in [fly_location, space_location, results_location, vip_words_location, w2v_space_location]:
    if not os.path.isdir(f):
        os.makedirs(f, exist_ok=True)

print("=== Fruitfly Parameters ===")
d = True if input("use default parameters (50, 40000, 20, 8, log, 10000) [y/n]?").upper() == "Y" else False
if d is False:
    pns = int(input("Initial number of PNs:"))
    kcs = int(input("Number of KCs:"))
    con = int(input("Connections per KC:"))
    red = int(input("WTA-percentage (whole number):"))
    flat = input("Flattening function (log/log2/log10):")
    max_pns = int(input("Maximum number of PNs:"))
else:
    # Initial Fruitfly parameters (taken from the ukwac_100m gridsearch on 10k dims)
    pns, kcs, con, red, flat, max_pns = 50, 40000, 20, 8, "log", 10000

print("=== Other Parameters ===")
s = input("Window size (to each side) for counting (default: 5):")
window = int(s) if len(s) > 0 else 5  # one-directional window size for counting (+/- 5 words)
s = input("Periodic deletion of infrequent words from the count (optional) -- minimum occurrences:")
min_occs_in_text = int(s) if (len(s) > 0 and int(s) > 1) else None # this will keep the count matrix' dimensions smaller
s = input("Maximum vocabulary size for the count (skip this for true incrementality):") #TODO test the case of not skipped
max_dims = int(s) if len(s) > 0 else None
s = input("Test interval in words (default: 1000000; file-wise testing with 'f'):") #TODO change so that 'None' means file-wise testing
#TODO test with file-wise testing
#TODO test with slice size larger than file size
try:
    test_interval_in_words = int(s) if len(s) > 0 else 1000000
    test_filewise = False
except ValueError:
    test_interval_in_words = None
    test_filewise = True # this is redundant, but more readable than handling test_interval_in_words==None

s = input("Number of important words to be extracted (default: 50):")
number_of_vip_words = int(s) if len(s) > 0 else 50

tokenize = False if input("Tokenize the input text? [y/n]").upper() == "N" else True
postag_simple = True if input("Only count nouns/verbs/adjectives? [y/n]").upper() == "Y" else False #TODO test this (True) # if true, it will only count nouns, verbs, and adjectives.
#linewise = False # DON'T SET THIS TO True! It will break the loop. #CLEANUP
verbose  = False if input("Be verbose while running? [y/n]").upper() == "N" else True

print("=== Let's go! ===")
#except Exception as e:
#    with open(errorlog, "a") as f:
#        f.write(str(e)[:500]+"\n")
#    print("An error occured. Check",errorlog,"for further information.")

#========== SETUP RESOURCE PATHS, INITIAL FRUIT FLY, AND INCREMENTOR
#try:
""" make a list of file paths to be passed one by one to an Incrementor object """
corpus_files = []
if os.path.isfile(corpus_dir):
    corpus_files = [corpus_dir]
else:
    for (dirpath, dirnames, filenames) in os.walk(corpus_dir):
        corpus_files.extend([dirpath + "/" + f for f in filenames])

""" This step is taken because Incrementor can't do custom initialization """
initial_fly_file = fly_location+"fly_run_0.cfg"
first_fly = Fruitfly.from_scratch(flattening=flat, pn_size=pns, kc_size=kcs, proj_size=con, hash_percent=red, max_pn_size=max_pns)
first_fly.log_params(filename=initial_fly_file, timestamp=False)

"""
The constructor of Incrementor 
- reads in text files from a directory (if specified)
- makes a frequency distribution
- sets up a Fruitfly object from an already-logged Fruitfly
- sets up a cooccurence matrix, with a limit if specified
matrix_incremental=False because it would need to load an existing matrix;
however, in this setup, it can increment all the way from an initial one.
linewise=False because it would destroy the whole loop. Therefore, it is
also commented out in the parameter section above.
"""

""" Initialization without corpus. File paths are passed to the Incrementor via the loop. """
breeder = Incrementor(None, matrix_filename,
                      corpus_tokenize=tokenize, corpus_linewise=False, corpus_checkvoc=overlap_file,
                      matrix_incremental=False, matrix_maxdims=max_dims, contentwords_only=postag_simple,
                      fly_new=False, fly_grow=True, fly_file=initial_fly_file,
                      verbose=verbose) #TODO rename breeder
#except Exception as e:
#    with open(errorlog, "a") as f:
#        f.write(str(e)[:500]+"\n")
#    print("An error occured. Check", errorlog, "for further information.")



#========== FUNCTIONS AND METHODS

def new_paths(run, verbose=False):
    """
        :param run: int
        :param verbose: bool
        :return: str, str, str, str
        """
    #if verbose: print("\n\nNEW RUN OF THE PIPELINE:", run, "\n") #CLEANUP
    a = fly_location + "fly_run_" + str(run) + ".cfg"  # for logging
    b = space_location + "space_run_" + str(run) + ".dh"
    c = results_location + "stats_run_" + str(run) + ".txt"
    d = vip_words_location + "words_run_" + str(run) + ".txt"
    return a,b,c,d

def count_test_log(count_these):
    # TODO maybe improve on the verbose option within this whole method?
    global t_thisrun, f, t, e
    # only counts cooccurrences of words within the freq_dist (which can be limited by matrix_maxdims)
    t_count = breeder.count_cooccurrences(words=count_these, window=window, timed=True)
    # delete words from the count matrix which are very infrequent
    nr_of_del_dims, t_cooc_del = breeder.reduce_count_size(min_occs_in_text, verbose=verbose, timed=True)
    # TODO make min_occs_in_text an attribute of Incrementor?
    # TODO log the number of deleted words and the time for deletion
    is_x, x_diff = breeder.check_overlap(checklist_filepath=overlap_file, wordlist=count_these)
    breeder.log_matrix()
    """ Read in the count model and apply the FFA """
    unhashed_space = utils.readDM(breeder.outspace)
    # print("length of unhashed_space:", len(unhashed_space))  # CLEANUP
    i_to_words, words_to_i = utils.readCols(breeder.outcols)
    # print("length of words_to_i obtained from unhashed_space: {0}".format(len(words_to_i)))  # CLEANUP
    # only select words that will be needed for evaluation:
    if overlap_file is None:
        fly_these = unhashed_space  # in this case, fly() is applied to the whole of unhashed_space
    else:
        words_for_flight = breeder.read_checklist(checklist_filepath=overlap_file,
                                                  with_pos_tags=breeder.postag_simple)
        fly_these = {w: unhashed_space[w] for w in words_for_flight if w in unhashed_space}
    # print("length of words_to_i:", len(breeder.words_to_i))  # CLEANUP
    # print("length of fly_these:", len(fly_these))  # CLEANUP
    # space_dic and space_ind are the words_to_i and i_to_words of the cropped vectors (done in Fruitfly.fit_space())
    hashed_space, space_dic, space_ind, t_flight = breeder.fruitfly.fly(fly_these, words_to_i, timed=True)
    # print("length of space_dic:", len(space_dic))  # space_dic = {word:i}; space_ind = {i:word} # or hash unhashed_space in stead of fly_these? #CLEANUP
    """ Evaluate the hashed space """
    spb, tsb = MEN.compute_men_spearman(unhashed_space, testset_file)
    spa, tsa = MEN.compute_men_spearman(hashed_space, testset_file)
    sp_diff = spa - spb
    connectednesses = [len(cons) for cons in breeder.fruitfly.pn_to_kc.values()]
    avg_PN_con = round(sum(connectednesses) / breeder.fruitfly.pn_size, 6)
    var_PN_con = round(np.var(connectednesses, ddof=1), 6)
    std_PN_con = round(np.std(connectednesses, ddof=1), 6)
    t_thisrun = time.time() - t_thisrun
    """ Log the hashed space, the fruitfly, and the evaluation """
    utils.writeDH(hashed_space, space_file)  # space_file is updated above
    # log most important words for each word
    print("Logging most important words to", vip_words_file, "...")
    with open(vip_words_file, "w") as f:
        for w in tqdm(hashed_space):
            vip_words = breeder.fruitfly.important_words_for(hashed_space[w], space_ind, n=number_of_vip_words)
            vip_words_string = ", ".join(vip_words)
            f.write("{0} --> {1}\n".format(w, vip_words_string))
    # log the whole fruitfly
    breeder.log_fly()  # breeder.flyfile is updated at the very beginning of the run
    # log the results
    with open(results_file, "w") as f:
        result_string = "RESULTS FOR RUN " + str(run) + ":\n" + \
                        "\nsp_corr_diff:       " + str(sp_diff) + \
                        "\nsp_corr_after:      " + str(spa) + \
                        "\nsp_corr_before:     " + str(spb) + \
                        "\nitems_tested_after: " + str(tsa) + \
                        "\nitems_tested_before:" + str(tsb) + "\n"
        fc = breeder.fruitfly.get_specs()  # fly config
        t = (fc["flattening"], fc["pn_size"], fc["kc_size"], fc["proj_size"], fc["hash_percent"])
        fruitfly_string = "\nFFA_CONFIG: ({0}, {1}, {2}, {3}, {4})".format(t[0], t[1], t[2], t[3], t[4]) + \
                          "\navg_PN_connections: " + str(avg_PN_con) + \
                          "\nvar_PN_connections: " + str(var_PN_con) + \
                          "\nstd_PN_connections: " + str(std_PN_con) + "\n"
        time_string = "\nTIME TAKEN: " + \
                      "\ncounting: " + str(t_count) + \
                      "\nflying:   " + str(t_flight) + \
                      "\ntotal:    " + str(t_thisrun) + "\n"
        file_string = "\nRELATED FILES:" \
                      "\nfruitfly:        " + breeder.flyfile + \
                      "\nhashed_space:    " + space_file + \
                      "\nimportant_words: " + vip_words_file + "\n"
        f.write(result_string + fruitfly_string + time_string + file_string)
    # keep internal log
    performance_summary[run] = [len(count_these), fc["pn_size"], tsb, spb, spa, sp_diff, t_thisrun]
    # TODO extract the whole Word2Vec part to one or multiple method(s)
    """ Train and evaluate a Word2Vec model """
    if w2v_exe_file is not None:  # TODO check for errors in the else-case (missing stats to log?)
        print("\nRUNNING WORD-2-VEC\n")
        # add the current text slice to a file that can be used by w2v
        with open(w2v_corpus_file, "a") as f:
            f.write(" ".join(count_these) + " ")

        # choose the minimum word count for the w2v run: it's the lowest count in the FFA's PN layer
        occs = sorted([sum(vec) for vec in breeder.cooc])[:breeder.fruitfly.max_pn_size]
        w2v_min_count = math.floor(occs[0] / (window * 2))  # selects the smallest number

        w2v_space_file = w2v_space_location + "space.txt"
        w2v_vocab_file = w2v_space_location + "space.vocab"
        print("training w2v with minimum count", w2v_min_count, "...")

        # run the w2v code
        #try:
        t_w2v = time.time()
        os.system(w2v_exe_file + " " + \
                  "-train " + w2v_corpus_file + \
                  "-output " + w2v_space_file + \
                  "-size 300 " + \
                  "-window " + str(window) + \
                  "-sample 1e-3 " + \
                  "-negative 10 " + \
                  "-iter 1 " + \
                  "-min-count " + str(w2v_min_count) + \
                  "-save-vocab " + w2v_vocab_file)
        t_train = time.time() - t_w2v
        #except Exception as e:
        #    with open(errorlog, "a") as f:
        #        f.write(str(e)[:500]+"\n")
        #    print("An error occured while running Word2Vec. Check", errorlog, "for further information.")

        # evaluate the w2v model
        w2v_space = utils.readDM(w2v_space_file)
        spcorr, pairs = MEN.compute_men_spearman(w2v_space, testset_file)
        with open(w2v_results_file, "a+") as f:
            f.write("RUN " + run + ":" + \
                    "\tSP_CORR: " + spcorr + \
                    "\tTEST_PAIRS: " + pairs + \
                    "\tTRAIN_TIME: " + t_train + "\n")
        # keep internal log
        performance_summary[run].extend([spcorr, pairs, t_train])


#========== LOOP
#try: #TODO make the try:except blocks smaller
runtime_zero = time.time()
performance_summary = {} # for a final summary
wc = 0 # word count; used for slicing
run = 0

for (file_n, file) in enumerate(corpus_files):
    breeder.extend_corpus(file) # read in a new file (but no counting yet)

    if test_interval_in_words is not None:
        """ Count and test in word intervals """
        if file_n+1 == len(corpus_files) and len(breeder.words()) < wc+test_interval_in_words:
            """ Last run with the remaining words of breeder.words """
            t_thisrun = time.time()  # ends before logging
            run += 1
            if verbose: print("\n\nStarting the last run ...")
            breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run, verbose=verbose)
            count_these = breeder.words[wc:len(breeder.words)] # take the whole rest of breeder.words()
            if verbose: print("Size of the final corpus slice:", len(count_these))
            count_test_log(count_these) # this is the actual loop content

        while len(breeder.words) >= wc+test_interval_in_words:
            """ Loop as long as there are enough words for a whole slice """
            t_thisrun = time.time()  # ends before logging
            run = int(wc / test_interval_in_words) + 1
            if verbose: print("\n\nStarting run",run,"...")
            breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run, verbose=verbose)
            count_these = breeder.words[wc: wc + test_interval_in_words]
            if verbose: print("Size of corpus slice:", len(count_these))
            count_test_log(count_these) # this is the actual loop content
            wc += test_interval_in_words # last statement of the while loop

    elif test_filewise is True:
        """ Count and test once per file """
        t_thisrun = time.time()  # ends before logging
        run += 1
        if verbose: print("\n\nStarting run", run, "with file", file, "...")
        breeder.flyfile, space_file, results_file, vip_words_file = new_paths(run, verbose=verbose)
        count_these = breeder.words[wc:len(breeder.words)] # count the whole newly read-in document
        if verbose: print("Size of text resource in this run:", len(count_these))
        count_test_log(count_these)  # this is the actual loop content
        wc = len(breeder.words)  #




#TODO make this prettier!
if verbose: print("\n\nFinished all runs.")
# make a summary file
with open(results_summary_file,"w") as f:
    f.write("run\tnew_data_in_words\tPN_size\ttestset\tspearman_before\tspearman_after\tsp_diff\ttime-taken\tw2v-score\tw2v_testset\n\n")
    for k,t in performance_summary.items():
        f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n"\
            .format(k,t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8]))


print("done.")
#except Exception as e: #TODO is this try:except obsolete?
#    with open(errorlog, "a") as f:
#        f.write(str(e)[:500]+"\n")
        

