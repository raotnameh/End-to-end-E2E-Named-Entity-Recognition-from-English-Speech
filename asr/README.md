*<sub>This code is heavily based on the **[SeanNaren's](https://github.com/SeanNaren/deepspeech.pytorch)** Deepspeech2 (DS2) pytorch implementation. We highly recommend to go take a look at his repo for deeper understanding and any issue.</sub>*

### Building it from Source

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Install this fork for Warp-CTC bindings:
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc; mkdir build; cd build; cmake ..; make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding && python setup.py install
```

Install NVIDIA apex:
```
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
```

If you want decoding to support beam search with an optional language model, install ctcdecode:
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

Finally clone this repo and run this within the repo:
```
pip install -r requirements.txt
```

## Training

### Datasets
#### Custom Dataset
To create a custom dataset you must create a CSV file containing the locations of the training data. This has to be in the format of:
```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```
The first path is to the audio file, and the second path is to a text file containing the transcript on one line. This can then be used as stated below.

#### Merging multiple manifest files

To create bigger manifest files (to train/test on multiple datasets at once) we can merge manifest files together like below using different csvs.
```
cat file_1.csv file_2.cav > combined.csv
```

### Training a Model
For training an E2E NER use star_labels.json, and for training a standard ASR use true_labels.json
```
python train.py --train-manifest data/ner/train.csv --val-manifest data/ner/dev.csv --cuda --rnn-type gru --hidden-layers 5 --momentum 0.95 --weights models/without_space.pth --opt-level O0 --loss-scale 1.0 --hidden-size 1024 --epochs 50 --lr 0.0051 --gpu-rank 4 --batch-size 32 --labels star_labels.json
```

Use `python train.py -h` for more parameters and options.

Different Optimization levels are available. More information on the Nvidia Apex API can be seen [here](https://nvidia.github.io/apex/amp.html#opt-levels).

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --test-manifest data/ner/dev.csv --cuda --model-path models/without_space.pth --gpu-rank 5 --batch-size 64 -decoder beam --beam-width 800 --alpha .96 --beta 4 --lm-path lm/4_gram.arpa
```

An example script to output a transcription has been provided:

```
python transcribe.py --audio-path path/to/audio.wav --cuda --model-path models/without_space.pth --decoder beam --beam-width 800 --alpha .96 --beta 4 --lm-path lm/4_gram.arpa 
```

To save the predicted transcripts (--path is to folder containg all the audio files).
```
python save.py --cuda --path /home/hemant/dummy/2/ --decoder beam --beam-width 1024 --alpha 1.96 --beta 6 --beam-width 800 --model-path models/without_space.pth --lm-workers 12 --gpu 4 --output_path data/libri_pred_txt/ --lm-path lm/4_gram.arpa
```
## Using an ARPA LM

We support using kenlm based LMs. Below are instructions on how to take the LibriSpeech LMs found [here](http://www.openslr.org/11/) and tune the model to give you the best parameters when decoding, based on LibriSpeech.

### Building your own LM

To build your own LM you need to use the KenLM repo found [here](https://github.com/kpu/kenlm). Have a read of the documentation to get a sense of how to train your own LM. The above steps once trained can be used to find the appropriate parameters.

### Alternate Decoders
By default, `test.py` and `transcribe.py` use a `GreedyDecoder` which picks the highest-likelihood output label at each timestep. Repeated and blank symbols are then filtered to give the final output.

A beam search decoder can optionally be used with the installation of the `ctcdecode` library as described in the Installation section. The `test` and `transcribe` scripts have a `--decoder` argument. To use the beam decoder, add `--decoder beam`. 

## Pre-trained models
Soon


## Experimental results: <br/>
| System   | Category     | Precision | Recall | F1     |
| -------- | ------------ | --------- | ------ | ------ |
| Two-step |Micro average | 0.83      |0.77    |0.80    |
| E2E NER  |Micro average | **0.96**  |**0.85**|**0.90**|

For any queries [contact](raotnameh@gmail.com).

## Acknowledgements

Thanks to [Sean](https://github.com/SeanNaren) to open source his code!
