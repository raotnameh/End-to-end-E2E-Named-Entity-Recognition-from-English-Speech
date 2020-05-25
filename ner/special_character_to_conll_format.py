# Input - HE WOULD ALWAYS SHIRK MAKING A CHOICE HIS AUNT |HELEN] SAID TO HIM

# Output - (CoNLL-2003 format data with BIO tags)
# he O
# would O
# always O
# shirk O
# making O
# a O
# choice O
# his O
# aunt O
# helen B-PER
# said O
# to O
# him O


import pandas as pd
import numpy as np
import re
import os
import argparse

parser = argparse.ArgumentParser(description='Save files from special characters to CoNLL-200 format')
parser.add_argument('--input', '-i',
                        help='Name of the input folder')
parser.add_argument('--output', '-o',
                        help='Name of the output folder')

args = parser.parse_args()
input_folder=args.input
output_folder=args.output

files=os.listdir(input_folder)

for file in files:
    f=open(input_folder+file, "r")
    sentence=f.read()
    if len(sentence.split(" "))==1:
        continue
    split_sentence=sentence.split(" ")
    conll_dataset=pd.DataFrame()
    conll_dataset["Words"]=split_sentence
    conll_dataset["labels"]="O"
    count_spec=0
    count_spec_bra=0
    count_spec+=sentence.count("|")
    count_spec+=sentence.count("$")
    count_spec+=sentence.count("{")
    count_spec+=sentence.count("(")
    count_spec_bra+=sentence.count("]")
    count_words=0

    for i,row in conll_dataset.iterrows():
        count_words+=1
        if "|" in row["Words"]:
            row["labels"]="B-PER"
            if "]" in row["Words"]:
                continue
            elif "]" in " ".join(split_sentence[count_words:]):
                index=1
                while "]" not in conll_dataset["Words"].iloc[i+index]:
                    conll_dataset["labels"].iloc[i+index]="I-PER"
                    index+=1
                conll_dataset["labels"].iloc[i+index]="I-PER"
            else:
                row["labels"]="O"
        
        if "$" in row["Words"]:
            row["labels"]="B-LOC"
            if "]" in row["Words"]:
                continue
            elif "]" in " ".join(split_sentence[count_words:]):
                index=1
                while "]" not in conll_dataset["Words"].iloc[i+index]:
                    conll_dataset["labels"].iloc[i+index]="I-LOC"
                    index+=1
                conll_dataset["labels"].iloc[i+index]="I-LOC"
            else:
                row["labels"]="O"
            
        if "{" in row["Words"]:
            row["labels"]="B-ORG"
            if "]" in row["Words"]:
                continue
            elif "]" in " ".join(split_sentence[count_words:]):
                index=1
                while "]" not in conll_dataset["Words"].iloc[i+index]:
                    conll_dataset["labels"].iloc[i+index]="I-ORG"
                    index+=1
                conll_dataset["labels"].iloc[i+index]="I-ORG"
            else:
                row["labels"]="O"
            
        if "(" in row["Words"]:
            row["labels"]="B-MISC"
            if "]" in row["Words"]:
                continue
            elif "]" in " ".join(split_sentence[count_words:]):
                index=1
                while "]" not in conll_dataset["Words"].iloc[i+index]:
                    conll_dataset["labels"].iloc[i+index]="I-MISC"
                    index+=1
                conll_dataset["labels"].iloc[i+index]="I-MISC"
            else:
                row["labels"]="O"

    for i, row in conll_dataset.iterrows():           
        if "]" in row["Words"]:
            row["Words"]=(re.sub(r'\]','', row["Words"]))
        if "(" in row['Words']:
            row["Words"]=(re.sub(r'/(','', row["Words"]))
        if "{" in row['Words']:
            row["Words"]=(re.sub(r'/{','', row["Words"]))
        if "|" in row['Words']:
            row["Words"]=(re.sub(r'[|]','', row["Words"]))
        if "$" in row['Words']:
            row["Words"]=(re.sub(r'[$]','', row["Words"]))
        conll_dataset["Words"].iloc[i]=conll_dataset["Words"].iloc[i].lower()
            
    np.savetxt(output_folder+file, conll_dataset.values, fmt='%s', delimiter=' ')