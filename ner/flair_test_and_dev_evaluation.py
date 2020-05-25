import flair
import os
import argparse

parser = argparse.ArgumentParser(description='Save files with special characters')
parser.add_argument('--input', '-i',
                        help='Name of the input folder containing dev and test files')
parser.add_argument('--model', '-m',
                        help='Name of the model file')
parser.add_argument('--gpu', '-g',
                        help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')

args = parser.parse_args()
input_folder=args.input
model_file=args.model
gpu_type=args.gpu


flair.device = torch.device(gpu_type)
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger

#Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text',1: 'ner'}

corpus: ColumnCorpus = ColumnCorpus(input_folder, column_format={0: 'text',1: 'ner'})

tagger = SequenceTagger.load(model_file)
print("Dev set results")
result, _ = tagger.evaluate(corpus.dev)
print(result.detailed_results)
print("Test set results")
result, _ = tagger.evaluate(corpus.test)
print(result.detailed_results)