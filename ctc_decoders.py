import numpy as np
from tqdm import tqdm
from time import time
import json, pickle, os, string, tqdm, kenlm, json
from collections import defaultdict, Counter
from itertools import groupby
import Levenshtein as Lev


## TO CALCULATE WORD ERROR RATE AND CHARACTER ERROR RATE

#s1 = True text
#s2 = predicted text

def wer_(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    
    return Lev.distance(''.join(w1), ''.join(w2))

def cer_(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')

    return Lev.distance(s1, s2)




#When using the above implementation, use the code belove to calculate the wer in percentatge: 
#pred = list of ouput prediction of model (it is the text) # example [" MY NAME IS HEMANT", " I AM A GOD"]
# total_wer = 0
# for x in range(len(pred)):
#     transcript, reference = data_[x][1], pred[x]
#     wer_inst = wer(transcript, reference)
#     total_wer += float(wer_inst)
# print("WER is : ",total_wer/len(pred),"%")


## GREDDY DECODING

def ctc_best_path(out,labels):
    '''
    Iplements best path decoding as shown by Graves
    Arguments : 
    out : output array that has to be decode
    labels : labels for the corresponding output position
    '''
    out = [labels[i] for i in np.argmax(out, axis=1) if i!=labels[-1]]
    o = ""
    for i,j in groupby(out):
        o = o + i
    return o.replace("_","")


## CTC BEAM SEARCH IMPLEMENTATION

def sort_beam(ptot,k):
    "To sort the beams at any given time step"
    if len(ptot) < k:
        return [i for i in ptot.keys()]
    else:
        dict_ = sorted(dict((v,k) for k,v in ptot.items()).items(),reverse=True)[:k]
        return [i[1] for i in dict_]

#using WORD Language Model (LM)
def ctc_beam_search(out,labels, prune=0.0001, k=20, lm=None,alpha=0.3,beta=12):
    "implements CTC Prefix Search Decoding Algo13.043478260869565%'rithm as shown by Graves"
    '''
    out = ctc output
    labels = string of labels
    prune = prune the ctc output
    k=beam-width
    lm=word age model used
    alpha,beta = hyper-parameters
    '''

    bc_i = 0 # blank/special charatcter index 
    F = out.shape[1]
    out = np.vstack((np.zeros(F), out))
    steps = out.shape[0]
    
    pb, pnb = defaultdict(Counter), defaultdict(Counter)
    pb[0][''], pnb[0][''] = 1, 0
    prev_beams = ['']
    for t in range(1,steps):
        pruned_alphabet = [labels[i] for i in np.where(out[t] > prune)[0]]
        for b in prev_beams:
            for c_t in pruned_alphabet:
                index = labels.index(c_t)
                #Collapsing case (copy case as the last character in the beam)
                if c_t == "_": #Extending with a blank
                    pb[t][b] += out[t][index]*(pb[t-1][b] + pnb[t-1][b])   
                else:
                    i_plus = b + c_t
                    if len(b) > 0 and c_t == b[-1]: #Extending with the same character as the last one
                        pnb[t][b] += out[t][index]*pnb[t-1][b]
                        pnb[t][i_plus] += out[t][index]*pb[t-1][b]
                    #expanding the beam (extend case as the last character is different)
                    elif c_t == " " and len(b.replace(' ', '')) > 0 : # LM constraints
                        prob = [i[0] for i in lm.full_scores(i_plus,eos=False,bos=False)][-1]
                        lm_p = (10**prob)**alpha
                        pnb[t][i_plus] += lm_p*out[t][index]*(pb[t-1][b] + pnb[t-1][b])
                    else:
                        pnb[t][i_plus] += out[t][index]*(pb[t-1][b] + pnb[t-1][b])
                        
                    if i_plus not in prev_beams:
                        pb[t][i_plus] += out[t][index] * (pb[t - 1][i_plus] + pnb[t - 1][i_plus])
                        pnb[t][i_plus] += out[t][index] * pnb[t - 1][i_plus]

        ptot = pb[t] + pnb[t]
        for i in ptot.keys():
            ptot[i] = ptot[i]*(len(i)+1)**beta
        prev_beams = sort_beam(ptot,k)
    return prev_beams[0]





#using CHARACTER LM
def ctc_beam_search_clm(out,labels, prune=0.001, k=20, lm=None,alpha=0.3,beta=12):
    "implements CTC Prefix Search Decoding Algorithm as shown by Graves"
    
    '''
    out = ctc output
    labels = string of labels
    prune = prune the ctc output
    k=beam-width
    lm=charac language model used
    alpha,beta = hyper-parameters
    '''
    
    bc_i = 0 # blank/special charatcter index 
    F = out.shape[1]
    out = np.vstack((np.zeros(F), out))
    steps = out.shape[0]
    
    pb, pnb = defaultdict(Counter), defaultdict(Counter)
    pb[0][''], pnb[0][''] = 1, 0
    prev_beams = ['']
    for t in range(1,steps):
        pruned_alphabet = [labels[i] for i in np.where(out[t] > prune)[0]]
        for b in prev_beams:
            for c_t in pruned_alphabet:
                index = labels.index(c_t)
                #Collapsing case (copy case as the last character in the beam)
                if c_t == "_": #Extending with a blank
                    pb[t][b] += out[t][index]*(pb[t-1][b] + pnb[t-1][b])  
                else:  # LM constraints
                    i_plus = b + c_t
                     #Extending with the same character as the last one
                    if len(b) > 0 and c_t == b[-1]:
                        pnb[t][b] += out[t][index]*pnb[t-1][b]
                        pnb[t][i_plus] += out[t][index]*pb[t-1][b]
                    #expanding the beam (extend case as the last character is different)
                    elif len(b.replace(' ', '')) > 0 :
                        prob = [i[0] for i in lm.full_scores(i_plus,eos=False,bos=False)][-1]
                        lm_p = 1#(10**prob)**alpha
                        pnb[t][i_plus] += lm_p*out[t][index]*(pb[t-1][b] + pnb[t-1][b])
                    else:
                        pnb[t][i_plus] += out[t][index]*(pb[t-1][b] + pnb[t-1][b])
                        
                    if i_plus not in prev_beams:
                        pb[t][i_plus] += out[t][index] * (pb[t - 1][i_plus] + pnb[t - 1][i_plus])
                        pnb[t][i_plus] += out[t][index] * pnb[t - 1][i_plus]
                        
        ptot = pb[t] + pnb[t]
        for i in ptot.keys():
            ptot[i] = ptot[i]*(len(i)+1)**beta
        prev_beams = sort_beam(ptot,k)
    return prev_beams[0]
