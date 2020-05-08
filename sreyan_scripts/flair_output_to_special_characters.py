# Input format - {"text": "AND HERE IS FULL SCALE IT'S A TINY BONE AND IN THE MIDDLE IS THE MINISTER OF ETHIOPIAN TOURISM WHO CAME TO VISIT THE NATIONAL MUSEUM OF ETHIOPIA", 
# "labels": [[117, 144, "LOC"], [77, 86, "MISC"]]}

# Output format - AND HERE IS FULL SCALE IT'S A TINY BONE AND IN THE MIDDLE IS THE 
# MINISTER OF ETHIOPIAN TOURISM WHO CAME TO VISIT THE $NATIONAL MUSEUM OF ETHIOPIA]

# If you want MISC tag to be taken into consideration too, just uncomment lines 71, 72, 73 and 75
# Additionally, if you want spaces between your word and special characters, just add a space to "|" (eg.-"| ")

import os
import argparse

parser = argparse.ArgumentParser(description='Save files with special characters')
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
    entry=eval(f.read())
    sentence=entry['text']
    print(sentence)
    index=0
    per = 0
    loc = 0
    org = 0
    misc = 0
    all_labels=entry["labels"]
    # edge case to handle senetences where there are no labels
    if len(list(entry["labels"])) == 0:
        f=open(output_folder+file, "w")
        print(sentence.upper())
        f.write(sentence.upper())
        f.close()
        continue
    all_labels.sort(key = lambda x: x[0])

    for tag_list in all_labels:
            
        if "PER" in tag_list:
            per+=1
            list_sentence=list(sentence)
            list_sentence.insert(tag_list[0]+index,"|")
            list_sentence.insert(tag_list[1]+1+index,"]")
            sentence=list_sentence
            index+=2
                
        if "LOC" in tag_list:
            loc+=1
            list_sentence=list(sentence)
            list_sentence.insert(tag_list[0]+index,"$")
            list_sentence.insert(tag_list[1]+1+index,"]")
            sentence=list_sentence
            index+=2
                
        if "ORG" in tag_list:
            org+=1
            list_sentence=list(sentence)
            list_sentence.insert(tag_list[0]+index,"{")
            list_sentence.insert(tag_list[1]+1+index,"]")
            sentence=list_sentence
            index+=2
            
        if "MISC" in tag_list:
       #     misc+=1
            list_sentence=list(sentence)
       #     list_sentence.insert(tag_list[0]+index,"(")
       #     list_sentence.insert(tag_list[1]+1+index,"]")
            sentence=list_sentence
       #     index+=2
                

            
    f=open(output_folder+file, "w")
    print("".join(list_sentence).upper())
    f.write("".join(list_sentence).upper())
    f.close()