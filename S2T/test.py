import os
from ctc_decoders import *
import argparse


import numpy as np
import torch
from tqdm import tqdm

from data.data_loader import SpectrogramParser

from opts import add_decoder_args, add_inference_args
from utils import load_model

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--gpu-rank', default=0,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--prune', default=0.00001, type=float, help='prune value')
parser = add_decoder_args(parser)


labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "

if __name__ == '__main__':
	args = parser.parse_args()
	torch.set_grad_enabled(False)
	device = torch.device("cuda" if args.cuda else "cpu")
	model = load_model(device, args.model_path, args.half)
	spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

	torch.cuda.set_device(int(args.gpu_rank))

	if args.lm_path: lm = kenlm.LanguageModel(args.lm_path)
	with open(args.test_manifest,"r") as f:
		csv = f.readlines()

	total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
	for i in tqdm(csv):
		
		audio_path, reference_path = i.split(",")

		spect = spect_parser.parse_audio(audio_path).contiguous()
		spect = spect.view(1, 1, spect.size(0), spect.size(1))
		spect = spect.to(device)

		input_sizes = torch.IntTensor([spect.size(3)]).int()
		out, output_sizes = model(spect, input_sizes)
		out = out.cpu().detach().numpy()[0]

		if args.decoder == "greedy": transcript = ctc_best_path(out,labels)
		elif args.decoder == "beam_w": transcript = ctc_beam_search(out,labels,args.prune,args.beam_width,lm=lm,alpha=args.alpha,beta=args.beta)
		elif args.decoder == "beam_c": transcript = ctc_beam_search_clm(out,labels,args.prune,args.beam_width,lm=lm,alpha=args.alpha,beta=args.beta)

		with open(reference_path.replace("\n",""),"r") as f:
			reference = f.readline()

		# print(transcript,"\n",reference)
		# exit()
		# break
		wer_inst = wer_(transcript, reference)
		cer_inst = cer_(transcript, reference)
		total_wer += wer_inst
		total_cer += cer_inst
		num_tokens += len(reference.split())
		num_chars += len(reference)

	wer = (float(total_wer) / num_tokens)*100
	cer = (float(total_cer) / num_chars)*100
	print('Test Summary \t'
		'Average WER {wer:.3f}\t'
		'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))

