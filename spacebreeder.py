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
       pns, kcs, con, red, flat, max_pns, tokenize, postag_simple, verbose, window



errorlog = "spacebreeder_errorlog.txt"

#========== PARAMETER INPUT
try:
    

    print("=== Resources and Output Directories ===")
    s = input("Path to text resources (default: data/chunks):") #TODO for final version, set the path to the chunks directory
    corpus_dir = s if len(s) > 0 else "data/chunks" # "/mnt/8tera/shareclic/fruitfly/ukwac_100m_tok-tagged.txt" # "../ukwac_100m/ukwac_100m.txt" # "test/pride_postagged.txt" #CLEANUP
    s = input("Path to word2vec code (optional):")
    w2v_exe_file = s if len(s) > 0 else "./../share/word2vec" #TODO clean up this path (i.e. no if/else)
    testset_file = input("Path to test set:")# "data/MEN_dataset_lemma_form_full" # "./data/MEN_dataset_natural_form_full" #TODO set correct path
    s = input("Path to a word list to be checked for overlap (optional):")
    overlap_file = s if len(s) > 0 else "./data/MEN_lemma_vocabulary" # "./data/MEN_natural_vocabulary" #TODO clean up this path (i.e. no if/else)
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
    d = True if input("use default parameters (50, 40000, 20, 8, log, 10000) [y/n]?").upper == "Y" else False
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
    s = input("Maximal vocabulary size for the count (skip this for true incrementality):")
    max_dims = int(s) if len(s) > 0 else None
    s = input("Test interval in words (default: 1000000):")
    test_interval_in_words = int(s) if len(s) > 0 else 1000000
    s = input("Number of important words to be extracted (default: 50):")
    number_of_vip_words = int(s) if len(s) > 0 else 50

    tokenize = False if input("Tokenize the input text? [y/n]").upper() == "N" else True
    postag_simple = True if input("Only count nouns/verbs/adjectives? [y/n]").upper() == "Y" else False # if true, it will only count nouns, verbs, and adjectives.
    #linewise = False # DON'T SET THIS TO True! It will break the loop. #CLEANUP
    verbose  = False if input("Be verbose while running? [y/n]").upper() == "N" else True

    print("=== Let's go! ===")
except Exception as e:
    with open(errorlog, "a") as f:
        f.write(str(e)[:500])
    print("An error occured. Check",errorlog,"for further information.")

#========== SETUP INITIAL FRUIT FLY AND INCREMENTOR
try:

    """ This step is taken because Incrementor can't do custom initialization """
    initial_fly_file = fly_location+"fly_run_0.cfg"
    first_fly = Fruitfly.from_scratch(flattening=flat, pn_size=pns, kc_size=kcs, proj_size=con, hash_percent=red, max_pn_size=max_pns)
    first_fly.log_params(filename=initial_fly_file, timestamp=False)

    """
    This 
    - reads the whole corpus
    - makes a freq_dist of the whole corpus (with a limited number, if given)
    - sets up a Fruitfly
    - sets up a cooccurence matrix
    matrix_incremental=False because it would need to load an existing matrix;
    however, in this setup, it can increment all the way from an initial one.
    linewise=False because it would destroy the whole loop. Therefore, it is
    also commented out in the parameter section above.
    """
    breeder = Incrementor(corpus_dir, matrix_filename,
                          corpus_tokenize=tokenize, corpus_linewise=False, corpus_checkvoc=overlap_file,
                          matrix_incremental=False, matrix_maxdims=max_dims, contentwords_only=postag_simple,
                          fly_new=False, fly_grow=True, fly_file=initial_fly_file,
                          verbose=verbose)
except Exception as e:
    with open(errorlog, "a") as f:
        f.write(str(e)[:500])
    print("An error occured. Check", errorlog, "for further information.")

#========== LOOP
try:
    runtime_zero = time.time()
    performance_summary = {} # for a final summary

    for i in range(0, len(breeder.words), test_interval_in_words): #TODO change to iterate over resource files in a directory
        t_thisrun = time.time() # ends before logging
        run = int(i/test_interval_in_words)+1
        print("\n\nNEW RUN OF THE PIPELINE:",run,"\n")

        breeder.flyfile = fly_location+"fly_run_"+str(run)+".cfg" # for logging 
        space_file = space_location+"space_run_"+str(run)+".dh" 
        results_file = results_location+"stats_run_"+str(run)+".txt" 
        vip_words_file = vip_words_location+"words_run_"+str(run)+".txt" 


        #========== COUNT AND EXPAND, THEN LOG FOR FLYING
        try: # take a slice from the corpus
            count_these = breeder.words[i:i+test_interval_in_words]
            print("Size of corpus slice for run",run,":",len(count_these))
        except IndexError as e:
            count_these = breeder.words[i:len(breeder.words)]
            print("This batch of words only has",len(count_these),"items")

        # only counts cooccurrences of words within the freq_dist (which can be limited by matrix_maxdims)
        breeder.count_cooccurrences(words=count_these, window=window)#[1] # [1] selects the stats tuple
        is_x, x_diff = breeder.check_overlap(checklist_filepath=overlap_file, wordlist=count_these)

        # only log the cooccurrence counts that will be evaluated (to speed things up)
        #words_for_log = breeder.read_checklist(checklist_filepath=overlap_file, with_pos_tags=breeder.postag_simple) # disable this for full logging #CLEANUP?
        #print("length of checklist (= words_for_log):",len(words_for_log)) #CLEANUP
        #log_these = {w:breeder.words_to_i[w] for w in words_for_log if w in breeder.words_to_i} # disable this for full logging
        #print("length of words_to_i:",len(breeder.words_to_i))
        #print("length of log_these:",len(log_these))
        #breeder.log_matrix(only_these=log_these)#[1] # no update of the filepath needed # run without optional params for full logging
        breeder.log_matrix()

        #========== FLY AND EVALUATE    
        unhashed_space = utils.readDM(breeder.outspace)
        print("length of unhashed_space:",len(unhashed_space)) #CLEANUP
        i_to_words, words_to_i = utils.readCols(breeder.outcols)
        print("length of words_to_i obtained from unhashed_space: {0}".format(len(words_to_i))) #CLEANUP

        # only select words that will be needed for evaluation:
        if overlap_file is None:
            fly_these = unhashed_space # in this case, fly() is applied to the whole of unhashed_space
        else:
            words_for_flight = breeder.read_checklist(checklist_filepath=overlap_file,
                                                      with_pos_tags=breeder.postag_simple)
            fly_these = {w:unhashed_space[w] for w in words_for_flight if w in unhashed_space}
        print("length of words_to_i:",len(breeder.words_to_i)) #CLEANUP
        print("length of fly_these:",len(fly_these)) #CLEANUP
        
        
        # this is where the magic happens
        # space_dic and space_ind are the words_to_i and i_to_words of the cropped vectors (done in Fruitfly.fit_space())
        hashed_space, space_dic, space_ind, t_flight = breeder.fruitfly.fly(fly_these, words_to_i, timed=True)
        print("length of space_dic:",len(space_dic)) # space_dic = {word:i}; space_ind = {i:word} # or hash unhashed_space in stead of fly_these?

        spb,tsb = MEN.compute_men_spearman(unhashed_space, testset_file) 
        spa,tsa = MEN.compute_men_spearman(hashed_space, testset_file)
        sp_diff = spa-spb


        connectednesses = [len(cons) for cons in breeder.fruitfly.pn_to_kc.values()]
        avg_PN_con = round(sum(connectednesses)/breeder.fruitfly.pn_size,6)
        var_PN_con = round(np.var(connectednesses, ddof=1),6)
        std_PN_con = round(np.std(connectednesses, ddof=1),6)

        t_thisrun = time.time()-t_thisrun

        #if allow_disconnection:
        #    #breeder.fruitfly.selective_disconnect(<in the notes>) #TODO implement!


        #========== LOGGING (SPACE, FLY, PERFORMANCE)
        utils.writeDH(hashed_space, space_file) # space_file is updated above

        # log most important words for each word
        print("Logging most important words to",vip_words_file,"...")
        with open(vip_words_file, "w") as f:
            for w in tqdm(hashed_space):
                vip_words = breeder.fruitfly.important_words_for(
                    hashed_space[w], space_ind, n=number_of_vip_words)
                vip_words_string = ", ".join(vip_words)
                f.write("{0} --> {1}\n".format(w, vip_words_string))

        # log the whole fruitfly
        breeder.log_fly()#[1] # flyfile is updated above

        # log the results
        with open(results_file, "w") as f:
            result_string = "RESULTS FOR RUN "+str(run)+":\n\n"+\
                            "SP_DIFF:   {0}\nSP_AFTER:  {1}\nSP_BEFORE: {2}\nTESTED_AFT.: {3}\nTESTED_BEF.: {4}\n\n"\
                            .format(sp_diff, spa, spb, tsa, tsb)
            fly_specs = breeder.fruitfly.get_specs()
            t = (fly_specs["flattening"], fly_specs["pn_size"], fly_specs["kc_size"], fly_specs["proj_size"], fly_specs["hash_percent"])
            fruitfly_string = "FLY CONFIG: ({0}, {1}, {2}, {3}, {4})\n"\
                              .format(t[0],t[1],t[2],t[3],t[4])+\
                              "AVG PN CON.:  {0}\nVAR PN CON.: {1}\nSTD PN CON.: {2}\n\n"\
                              .format(avg_PN_con, var_PN_con, std_PN_con)
            time_string = "TIME TAKEN: \nFLYING: {0}\nTOTAL: {1}\n\n".format(t_flight, t_thisrun)
            file_string = "RELATED FILES:\nFRUITFLY: {0}\nSPACE: {1}\nVIP WORDS: {2}\n"\
                          .format(breeder.flyfile, space_file, vip_words_file)

            f.write(result_string \
                    +fruitfly_string \
                    +time_string \
                    +file_string)

        # keep internal log
        performance_summary[run] = [len(count_these),fly_specs["pn_size"], tsb, spb, spa, sp_diff, t_thisrun]





        #========== TRAIN WORD_2_VEC
        if (len(w2v_exe_file) > 0): # TODO check for errors in the else-case (missing stats to log?)
            print("\nRUNNING WORD-2-VEC\n")
            # add the current text slice to a file that can be used by w2v
            with open(w2v_corpus_file, "a") as f:
                f.write(" ".join(count_these))
                f.write(" ")

            # choose the minimum word count for the w2v run
            occs = sorted([sum(vec) for vec in breeder.cooc]) # has only 10k dimensions
            w2v_min_count = math.floor(occs[0]/(window*2)) # selects the lowest number

            w2v_space_file = w2v_space_location+"space.txt"
            w2v_vocab_file = w2v_space_location+"space.vocab"
            print("training w2v with minimum count",w2v_min_count,"...")

            # run the w2v code
            try:
                os.system("{0} -train {1} -output {2} -size 300 -window {3} -sample 1e-3 -negative 10 -iter 1 -min-count {4} -save-vocab {5}"\
                    .format(w2v_exe_file, w2v_corpus_file, w2v_space_file, window, w2v_min_count, w2v_vocab_file))
            except Exception as w2v_error:
                print("OBACHT!!! An error occured while running word2vec. Look at the errorlog.")
                with open(errorlog, "a") as f:
                    f.write(str(w2v_error))

            #========== EVALUATE WORD_2_VEC
            w2v_space = utils.readDM(w2v_space_file)

            spcorr,pairs = MEN.compute_men_spearman(w2v_space, testset_file)

            with open(w2v_results_file, "a+") as f:
                f.write("RUN {0}:\tSP_CORR: {1}\tTEST_PAIRS: {2}\n".format(run, spcorr, pairs))

            #keep internal log
            performance_summary[run].extend([spcorr,pairs])




    print("\n\nFINISHED ALL RUNS. ALMOST DONE.")
    # make a summary file
    with open(results_summary_file,"w") as f:
        f.write("run\tnew_data_in_words\tPN_size\ttestset\tspearman_before\tspearman_after\tsp_diff\ttime-taken\tw2v-score\tw2v_testset\n\n")
        for k,t in performance_summary.items():
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n"\
                .format(k,t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8]))


    print("done.")
except Exception as e:
    with open(errorlog, "a") as f:
        f.write(str(e)[:500])
        

