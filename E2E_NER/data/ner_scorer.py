import numpy as np
from tqdm .auto import tqdm
import glob, os, argparse
from IPython.display import display as ipd
import IPython

parser = argparse.ArgumentParser(description='to calculate the scores for the NER task from speech')
parser.add_argument('--true-path',default = '/home/hemant/true_files', help='full path to the folder')
parser.add_argument('--pred-path',default = '/home/hemant/pred_files', help='full path to the folder')
args = parser.parse_args()

true_path =  args.true_path + '/*.txt'
pred_path = args.pred_path + '/*.txt'

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



def main(true, pred):
    t = []
    for i in true:
        for j in i:
            if j: t.append(j)

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
            if j: p.append(j)

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
    return t,p


def score(tp, fp, fn):
    if tp == 0:
        pre, rec, f1 = 0, 0, 0
    else:
        pre = np.round(tp/(tp+fp),decimals=3)
        rec = np.round(tp/(tp+fn),decimals=3)
        f1 = np.round(2*((pre*rec)/(pre+rec)),decimals=3)
    return pre, rec, f1


true_ = glob.glob(true_path)
pred_ = glob.glob(pred_path)
prf = []
tpfp = []
assert len(true_) == len(pred_)
for j in tqdm(["|", "{", "$"]):
    start = [j]
    print()
    print(f"score for the {start[0]} tag")
    tp, fp, fn = 0, 0, 0
    t_ = []

    for i in range(len(true_)):

        true = [rea(true_[i])]
        pred = [rea(pred_[i])]
        t, p = main(true,pred)
        t = [i for i in t]
        p = [i for i in p]

        tp+=  len(set(p).intersection(t))
        fp+= len(set(p) - set(t))
        fn+=  len(set(t) - set(p))

    pre, rec,f1 = score(tp, fp, fn)
    prf.append([pre,rec,f1])
    tpfp.append([tp,fp,fn])
    print(f"prec is {pre}\n recall is {rec}\n f1 score is {f1}")


#macro
print("Macro score")
score_ = np.round((prf[0][0] + prf[1][0] + prf[0][0] ) /3,3),np.round((prf[0][1] + prf[1][1] + prf[2][1] ) /3,3), np.round((prf[0][2] + prf[1][2] + prf[2][2] ) /3,3) 
print(f"Prec, recall and f1 score are: {score_[0]}, {score_[1]},and {score_[2]}")

#micro
print("Micro scores")
score_ = score(tpfp[0][0] + tpfp[1][0] + tpfp[2][0], tpfp[0][1] +tpfp[1][1] + tpfp[2][1], tpfp[0][2] + tpfp[1][2] + tpfp[2][2])
print(f"Prec, recall and f1 score are: {score_[0]}, {score_[1]},and {score_[2]}")
