import pickle
import numpy as np
import json
import os.path
from data.data_loader import SpectrogramParser
import torch
from decoder import GreedyDecoder
import argparse

from pathlib import Path
import sox
from tqdm import tqdm
import warnings

from opts import add_decoder_args, add_inference_args
from utils import load_model

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def decode_results(decoded_output, decoded_offsets,path,siz):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            dur = sox.file_info.info(str(path))['duration']
            div = dur/siz
            if args.offsets:
                result['offsets'] = ((decoded_offsets[b][pi])*div).tolist()

            
            results['output'].append(result)
    return results
def ensem(out, out_e):
    dummy = torch.zeros_like(out)

    for b in range(len(out)):
        for r in range(out.shape[2]):
            if r <2:
                dummy[b,:,r] = out_e[b,:,r]
            elif r >=2 and r <28:
                dummy[b,:,r] = out_e[b,:,2]/26.0
            elif r ==28:
                dummy[b,:,r] = out_e[b,:,3]
            elif r ==29:
                dummy[b,:,r] = out_e[b,:,4]
            elif r ==30:
                dummy[b,:,r] = out_e[b,:,5]
            elif r ==31:
                dummy[b,:,r] = out_e[b,:,6]
            elif r ==32:
                dummy[b,:,r] = out_e[b,:,7]
            elif r ==33:
                dummy[b,:,r] = out_e[b,:,8]
    
    return (out+dummy*0.1)/1.1


def transcribe(audio_path, spect_parser, model, model_e, decoder, device, use_half):
    spect = spect_parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    out_e, e_s = model_e(spect, input_sizes)
    out = ensem(out, out)
    
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets, out.shape[1]



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='DeepSpeech transcription')
    arg_parser = add_inference_args(arg_parser)
    arg_parser.add_argument('--audio-path', default='audio.wav',
                            help='Audio file to predict on')
    arg_parser.add_argument('--path', default=' ',
                            help='Audio file to predict on')
    arg_parser.add_argument('--output_path', default=' ',
                            help='Transcription output folder')
    arg_parser.add_argument('--offsets', dest='offsets',
                            action='store_true', help='Returns time offset information')
    arg_parser.add_argument('--gpu', dest='gpu', type=str, help='GPU to be used')
    arg_parser = add_decoder_args(arg_parser)
    args = arg_parser.parse_args()
    print(args)
    device = torch.device("cuda" if args.cuda else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = load_model(device, args.model_path, args.half)
    model1 = load_model(device,'/home/hemant/asr_wm/models/star_dummy.pth',args.half)
    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(
            model.labels, blank_index=model.labels.index('_'))

    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)
    print(f'Reading files from {args.path}')
    directory_files = Path(args.path).glob('**/*.wav')
    print('Files have been read')
    for path in tqdm(directory_files):
        file_path = Path(args.path).joinpath(path)
        file_name = file_path.stem
        output_path = Path(args.output_path).joinpath(f'{file_name}.txt')
        decoded_output, decoded_offsets, size = transcribe(audio_path=file_path,
                                                         spect_parser=spect_parser,
                                                         model=model,
                                                         model_e=model1,
                                                         decoder=decoder,
                                                         device=device,
                                                         use_half=args.half)

        # print(json.dumps(decode_results(decoded_output, decoded_offsets)))  #['output'][0]['transcription'])))
        
        
        with open(output_path, "w") as fp:
            json.dump(decode_results(decoded_output, decoded_offsets,file_path,size)['output'][0], fp)

    print("finished")
