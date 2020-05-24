*Tested on Ubuntu 18.04.* <br/>
<sub>**Work in process.<br />**</sub>

## End-to-end Named Entity Recognition (E2E NER) from English Speech

*<sub>The [E2E NER from speech]() implementation used in this project is heavily based on the **[SeanNaren's](https://github.com/SeanNaren/deepspeech.pytorch)** Deepspeech2 (DS2) pytorch implementation and [FlairNLP](https://github.com/flairNLP/flair). I strongly recommend to look at these 2 repos</sub>*

## Overview: 
Named entity recognition (NER) from text has been a widely studied problem and usually extracts semantic information from text. Until now, NER from speech is mostly studied in a two-step pipeline process that includes first applying an automaticspeech recognition (ASR) system on an audio sample and then passing the predicted transcript to a NER tagger. In such cases,the error does not propagate from one step to another as boththe tasks are not optimized in an end-to-end (E2E) fashion. Recent studies confirm that integrated approaches (e.g., E2E ASR)outperform sequential ones (e.g., phoneme based ASR). 

## Problem statement:
In this work, we present an E2E NER approach from Enlgish speech, which jointly optimizes the ASR and NER tagger components. Experimental results show that the proposed E2E approach outperforms  the  classical  two-step  approach. 

## Overview:
* NER is an important task in information extraction systems and very useful in many applications. It has many [progress](https://nlp.cs.nyu.edu/sekine/papers/li07.pdf) and applications such as in optimizing search engine algorithms~, classifying content for news providers, and recommending content. However, NER from speech has many applications such as the privacy concerns in medical recordings (to mute or hide specific words such as patient names), but not a lot of work has been done in this regard.

## Dataset: 

### What we deliver in this repo:
* **Source code files**: For bth the ASR and NER tagger component as standlaone and for the E2E NER.
* **README.md**: Detailed Description of dataset preparation, augmentation(if any), detection network used, training
parameters/hyper-parameters and anchor box tuning.

### Experimental results: <br/>
| System   | Category     | Precision | Recall | F1     |
| -------- | ------------ | --------- | ------ | ------ |
| Two-step |Micro average | 0.83      |0.77    |0.80    |
| E2E NER  |Micro average | **0.96**  |**0.85**|**0.90**|

### Dataset preparation steps:
* The dataset for this task is prepared in two steps:
	* First downloading all the required files.
	* Pre-process the data in the required VOC format.
		* Annotated xml file is created in: *source_code/nidhi/VOC2007/Annotatinos*. <br/>
		* Train and test split is created as given for the task:  *source_code/nidhi/VOC2007/ImageSets/Main*. <br/>
		* All the images are put in one folder: *source_code/nidhi/VOC2007/JPEGImages.* <br/>

* Apart from the default data augmentation (normalizing the image, with mean and std), I did not use any augmented data to train the first draft. <br/>

**Detection network used:** A standard [SSD]("https://arxiv.org/pdf/1512.02325.pdf") architecture is used in this implementation. <br/>
* Default parameters used are as mentioned in the Max deGroot's implentation.
* Total number of classes are 2: one for the product and one for the backgorund class (no product). 
* For the anchor box: As mentioned in the problem statemnt only 1 anchor box per feture map is allowed. we use 6 feture maps (38, 19, 10, 5, 3, 1) as mentioned in the paper with only 1 anchor box each. Additionally the number of default boxes are: 1940 because of the 1 anchor box used for each feature map. We use a fixed ratio (width to height) for the anchor box used i.e.,"0.10:0.12". The reason is explained in **Q&A** Section.


**This is the first draft and a considerable speed increase can be achieved for a particular dataset.**

For any queries [contact](raotnameh@gmail.com).
