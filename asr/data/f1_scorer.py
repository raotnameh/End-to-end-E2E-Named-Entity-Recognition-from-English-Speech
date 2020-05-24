import numpy as np
from tqdm import tqdm
import glob, os

start = [ "|", "$", "{"]
stop = [']']

def rea(file):
    with open(file, 'r') as f:
        dummy = f.read()
    l = ''
    w = ''
    for i in dummy:
        if i in start:
            l+= i
            w= '-'
        elif w == '-' and len(l) != 0:
            l+= i
        if i in stop:
            l+= ','
            w = ''
    return l[:-1].split(',')

def step(true_,pred_):
    true = [rea(true_)]
    pred = [rea(pred_)]

    t = []
    for i in true:
        for j in i:
            t.append(j)

    t_ = []
    for i in t:
        s,st =0, 0
        for j in i:
            if j in start:
                s+=1
            if j in stop:
                st+=1
        if s ==st:
            t_.append(i)    
    t = t_

    p = []
    for i in pred:
        for j in i:
            if j:
                p.append(j)

    p_ = []
    for i in p:
        s,st =0, 0
        for j in i:
            if j in start:
                s+=1
            if j in stop:
                st+=1
        if s ==st:
            p_.append(i)    
    p = p_

    return len(set(p).intersection(t)) , len(set(p) - set(t)), len(set(t) - set(p))
    
    
def main(pred_, true_, show=True):

    tp,fp, fn = 0,0,0
    for i in range(len(pred_)):
        assert os.path.basename(true_[i]) ==  os.path.basename(pred_[i])
        a,b,c = step(true_[i], pred_[i])
        tp+=a
        fp+=b
        fn+=c
    return tp, fp, fn
    
    if show:
        print_(tp,fp,fn)
    else: 
        pre = np.round(tp/(tp+fp),decimals=3)
        rec = np.round(tp/(tp+fn),decimals=3)
        f1 = np.round(2*((pre*rec)/(pre+rec)),decimals=3)
        return pre, rec, f1

def print_(tp,fp,fn):
    pre = np.round(tp/(tp+fp),decimals=3)
    rec = np.round(tp/(tp+fn),decimals=3)
    f1 = np.round(2*((pre*rec)/(pre+rec)),decimals=3)

    print(f"prec is {pre}\n recall is {rec}\n f1 score is {f1}")
