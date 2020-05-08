# Program to count named entities for a folder with a sentences with special characters

# Input - HE WOULD ALWAYS SHIRK MAKING A CHOICE HIS AUNT |HELEN] SAID TO HIM

import argparse
import os

parser = argparse.ArgumentParser(description='Get count of named enitites')
parser.add_argument('--foldername', '-f',
                        help='Input the name of the folder')

args = parser.parse_args()
input_folder=args.foldername

files=os.listdir(input_folder)

person_tag=0
location_tag=0
misc_tag=0
org_tag=0

for file in files:
    f=open(input_folder + file, "r")
    sentence=f.read()
    person_tag+=sentence.count("|")
    location_tag+=sentence.count("$")
    org_tag+=sentence.count("{")
    misc_tag+=sentence.count("(")
    
    
print("The number of person tags is ",person_tag)
print("The number of location tags is ",location_tag)
print("The number of miscellaneous tags is ",misc_tag)
print("The number of oragnisation tags is ",org_tag)

