import argparse
import json
import sys
from multiprocessing import Pool

import numpy as np
import torch

from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder, BeamCTCDecoder
from model import DeepSpeech
from utils import load_model
from opts import add_decoder_args

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Path to model file created by training')
parser.add_argument('--logits', default="", type=str, help='Path to logits from test.py')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--num-workers', default=16, type=int, help='Number of parallel decodes to run')
parser.add_argument('--output-path', default="tune_results_noise.json", help="Where to save tuning results")
parser.add_argument('--lm-alpha-from', default=0, type=float, help='Language model weight start tuning')
parser.add_argument('--lm-alpha-to', default=4, type=float, help='Language model weight end tuning')
parser.add_argument('--lm-beta-from', default=0.5, type=float,
                       help='Language model word bonus (all words) start tuning')
parser.add_argument('--lm-beta-to', default=4, type=float,
                       help='Language model word bonus (all words) end tuning')
parser.add_argument('--lm-num-alphas', default=40, type=float, help='Number of alpha candidates for tuning')
parser.add_argument('--lm-num-betas', default=10, type=float, help='Number of beta candidates for tuning')
parser = add_decoder_args(parser)
args = parser.parse_args()


def decode_dataset(logits, test_dataset, batch_size, lm_alpha, lm_beta, mesh_x, mesh_y, labels, grid_index,device):
    print("Beginning decode for {}, {}".format(lm_alpha, lm_beta))
    test_loader = AudioDataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    decoder = BeamCTCDecoder(labels, beam_width=args.beam_width, cutoff_top_n=args.cutoff_top_n,
                             blank_index=labels.index('_'), lm_path=args.lm_path,
                             alpha=lm_alpha, beta=lm_beta, num_processes=1)
    total_cer, total_wer = 0, 0
    for i, (data) in enumerate(test_loader):
        inputs, targets, input_percentages, target_sizes = data

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out = torch.from_numpy(logits[i][0]).to(device)
        sizes = torch.from_numpy(logits[i][1]).to(device)

        decoded_output, _, = decoder.decode(out, sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference) / float(len(reference.split()))
            cer_inst = decoder.cer(transcript, reference) / float(len(reference))
            wer += wer_inst
            cer += cer_inst
        total_cer += cer
        total_wer += wer

    wer = total_wer / len(test_loader.dataset)
    cer = total_cer / len(test_loader.dataset)
    print(grid_index, mesh_x, mesh_y, lm_alpha, lm_beta, wer, cer)

    return [grid_index, mesh_x, mesh_y, lm_alpha, lm_beta, wer, cer]


if __name__ == '__main__':
    if args.lm_path is None:
        print("error: LM must be provided for tuning")
        sys.exit(1)
    torch.backends.cudnn.enabled=False
    
    device = torch.device("cuda" )
    torch.cuda.set_device(int(1))
    model = load_model(device, args.model_path, "cuda")
    #model = DeepSpeech.load_model(args.model_path)

    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest,
                                      labels=model.labels, normalize=True)

    logits = np.load(args.logits)
    batch_size = logits[0][0].shape[0]

    p = Pool(args.num_workers)

    cand_alphas = np.linspace(args.lm_alpha_from, args.lm_alpha_to, args.lm_num_alphas)
    cand_betas = np.linspace(args.lm_beta_from, args.lm_beta_to, args.lm_num_betas)
    params_grid = []
    for x, alpha in enumerate(cand_alphas):
        for y, beta in enumerate(cand_betas):
            params_grid.append((alpha, beta, x, y))
    device = torch.device("cuda:0")
    futures = []
    for index, (alpha, beta, x, y) in enumerate(params_grid):
        print("Scheduling decode for a={}, b={} ({},{}).".format(alpha, beta, x, y))
        futures.append(decode_dataset(logits, test_dataset, batch_size, alpha, beta, x, y, model.labels, index,device))
                         
    for f in futures:
        print("Result calculated:", f)
    print("Saving tuning results to: {}".format(args.output_path))
    with open(args.output_path, "w") as fh:
        json.dump(futures, fh)
