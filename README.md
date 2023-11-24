# Pre-training with A Rational Approach for Antibody (PARA)

## Introduction
Antibodies are a specific class of proteins produced by the adaptive immune system to combat invading pathogens. Understanding the wealth of information encoded in antibody amino acid sequences can significantly aid in predicting antibody properties and developing novel therapeutics.

## About PARA
The PARA (Pre-training with A Rational Approach for antibodies) model is a pre-trained model specifically designed for antibody sequences. It leverages a unique training strategy tailored to antibody sequence patterns, combined with an advanced self-encoding model structure inspired by natural language processing (NLP) techniques.

## Abstract
Mining the embedded information in antibody amino acid sequences can benefit antibody property prediction and novel therapeutic development. While protein-specific pre-training models have been useful in extracting latent representations from protein sequences, there is a need for improvement in models specifically for antibody sequences. Existing models often do not fully consider the unique differences between protein sequences and language sequences, nor do they account for the distinct features of antibodies.

The PARA model addresses these gaps by employing a training strategy that conforms to antibody sequence patterns, using an advanced NLP self-encoding model structure. Our results demonstrate that PARA significantly outperforms selected antibody pre-training models, highlighting its capability to capture antibody sequence information effectively.


## Usage

In our GitHub project, we have open-sourced the PARA model and provided an inference script, `inference.py`, which is capable of predicting missing parts in antibody sequences and extracting the latent vector representations post-PARA encoding. To run `inference.py`, you will need to specify the location of the tokenizer and the model weights. While the tokenizer has been made available, we will provide the model weights file at a later date, once an appropriate online hosting solution is found. For a practical usage example, you can run the following command:

```bash
python inference.py -m model_zoo/model.pt -t path_to_abtokenizer

