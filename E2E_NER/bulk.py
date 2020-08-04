import pickle
import json
import os.path
from data.data_loader import SpectrogramParser
import torch
from decoder import GreedyDecoder
import argparse
import numpy as np

from pathlib import Path

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
import warnings
import glob
from opts import add_decoder_args, add_inference_args
from utils import load_model
import torch.nn.functional as F

import concurrent.futures

warnings.simplefilter('ignore')

arg_parser = argparse.ArgumentParser(
        description='DeepSpeech transcription')
arg_parser = add_inference_args(arg_parser)
arg_parser.add_argument('--path', default=' ',
                        help='Audio file to predict on', required=True)
arg_parser.add_argument('--gpu', dest='gpu', type=str, help='GPU to be used', required=True)
arg_parser.add_argument('--batch-size', default=5, type=int, help='Batch size for testing')
arg_parser.add_argument('--m1', type=float,default=0.0, help='start of the percentage')
arg_parser.add_argument('--m2', type=float, default=1.0,help='end of the percentage')
arg_parser.add_argument('--workers', default=48, type=int, help='num of workers to use')
arg_parser.add_argument('--save', default='', type=str, help='where to save the files')
arg_parser = add_decoder_args(arg_parser)
args = arg_parser.parse_args()


device = torch.device("cuda" if args.cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
model = load_model(device, args.model_path, args.half)
spect_parser = SpectrogramParser(model.audio_conf, normalize=True)


def transcribe(audio_path, use_half=False):
    spect = spect_parser.parse_audio(audio_path)
    spect = spect.view(1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(2)]).int()

    return spect, input_sizes

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def save_(p_,data_):
    with open(f"{args.save}/{os.path.basename(p_).split('.')[0]}.txt", "w") as f:
        f.write(data_)

if __name__ == '__main__':
    if args.decoder == "beam":
        print(f"using beam decoder")
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.workers, blank_index=model.labels.index('_'))
    else:
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
        
    
    print(f'Reading files from {args.path}')
    directory_files = [i for i in glob.glob(args.path + '*.wav')]
    print(f"{len(directory_files)} number of Files have been read")

    t_ = len(directory_files)

    trans = [os.path.basename(i).split('.')[0] + '.wav' for i in glob.glob( args.save+ '*.txt')]
    directory_files = [i for i in directory_files if os.path.basename(i) not in trans]

    assert len(directory_files) == t_ - len(trans)

    print(f"{len(directory_files)} updated number of Files.")

    steps = []
    for x in batch(np.arange(0,len(directory_files)), args.batch_size):
        steps.append([x[0],x[-1]])
    directory_files = [directory_files[i[0]:i[1]+1] for i in steps]

    x1 = int(len(directory_files))
     
    print(f"starting main process")
    for batch in tqdm(directory_files[int(x1*args.m1):int(x1*args.m2)]):
        a_ = []
        length = 0
        try:
            for path in batch:
                spect, input_sizes = transcribe(audio_path=path,
                                    use_half=args.half)
                if length < input_sizes.item(): 
                    length = input_sizes.item() 
                                    
                a_.append([spect,input_sizes,path])
        except: continue
        a_ = sorted(a_, key = lambda x: x[1].item(), reverse=True)
        input_sizes = torch.tensor([i[1] for i in a_])
        batch_temp = [i[2] for i in a_]
    
        spect = torch.stack([F.pad(input=i[0], pad=(0,length-i[0].shape[-1]), mode='constant', value=0) for i in a_])
        
        out, output_sizes  = model(spect, input_sizes)
        decoded_output, scores, = decoder.decode(out, output_sizes)

        
        [save_(batch_temp[m],decoded_output[m][0]) for m in range(out.shape[0])]

        # print('done')

    print("finished")
