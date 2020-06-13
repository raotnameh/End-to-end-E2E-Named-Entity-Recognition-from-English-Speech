*<sub>This section describes various python scripts needed for training flair and formatting your data for use with the model for the 2Step NER.</sub>*

## Flair model training, testing and evaluation

### Training your flair model

Input folder structure

```
input_data_folder/
├── train.txt
├── text.txt
└── valid.txt
```
```
python flair_train.py --input input folder --output output folder --gpu cuda/cpu
```
Note: Data should have only two columns, namely, text and ner. Output folder will have multiple files and folders. The model we are interested in best_model.pt .


### Model testing and evaluation

Input folder structure
```
input_data_folder/
├── text.txt
└── valid.txt
```
```
python flair_test_and_dev_evaluation.py --input input folder path --model model path --gpu cuda/cpu
```
Note: Data should have only two columns, namely, text and ner.

Link to pre-trained flair models:

https://drive.google.com/file/d/1-ABVzo2O46q9dFlhZN5nqt1gkUOQYI98/view?usp=sharing

Note: The folder has two models used for our experiments.
 
flair_trained_on_lowercase_conll.pt - Model trained on lowercase CoNLL-2003 data without MISC tag.
flair_trained_on_lowercase_conll.pt - Model trained on lowercase CoNLL-2003 data combined with our manually annotated data, without MISC tag.

## Converting files after predicting named entities through Flair

### Converting data from flair output (dictionary format) to data with special characters

eg.-

```
Input - {"text": "MY NAME IS SREYAN AND I LIVE IN INDIA", "labels": [[11, 17, "PER"], [32, 37, "LOC"]]}

Output format - MY NAME IS |SREYAN] AND I LIVE IN $INDIA]
```

```
python flair_output_to_special_characters.py --input input folder path --output output folder path
```
Note: The structure of your folder should be one single file cotaining one single sentence. The output folder will have the converted text with the similar file name.


### Converting data from special symbols format to CoNLL-2003 format with BIO Tags

eg.-
```
Input - MY NAME IS |SREYAN] AND I LIVE IN $INDIA]

Output -

my O
name O
is O
sreyan B-PER
and O
i O
live O
in O
india B-LOC
```
```
python special_character_to_conll_format.py --input input folder path --output output folder path
```
Note: The structure of your folder should be one single file cotaining one single sentence.The output folder will have the converted text with the similar file name.

### Counting the number of named entities in a folder of files with annotated text (special character format)

```
Input - MY NAME IS |SREYAN] AND I LIVE IN $INDIA]
```

```
python count_tags.py --input input folder path
```
Note: The structure of your folder should be one single file cotaining one single sentence. Output will be showed on your terminal.

## Extra

```
ner_hacks.ipynb
```
This jupyter notebook contains small scripts for NLP data wrangling mainly focusing on CoNLL-2003 data wrangling. The notebook is well commented and self-explainatory.


