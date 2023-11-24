# Pre-training with A Rational Approach for Antibody (PARA)

## Introduction
Antibodies are a specific class of proteins produced by the adaptive immune system to combat invading pathogens. Understanding the wealth of information encoded in antibody amino acid sequences can significantly aid in predicting antibody properties and developing novel therapeutics.

## About PARA
The PARA (Pre-training with A Rational Approach for antibodies) model is a pre-trained model specifically designed for antibody sequences. It leverages a unique training strategy tailored to antibody sequence patterns, combined with an advanced self-encoding model structure inspired by natural language processing (NLP) techniques.

## Publication
This repository is associated with the following publication:

**Title:** Pre-training with A Rational Approach for Antibody

**Authors:** Xiangrui Gao, Changling Cao, Lipeng Lai

**Affiliations:** XtalPi Innovation Center, Beijing Institute of Technology

**Contact:** [xiangrui.gao@xtalpi.com](mailto:xiangrui.gao@xtalpi.com), [3220201929@bit.edu.cn](mailto:3220201929@bit.edu.cn), [lipeng.lai@xtalpi.com](mailto:lipeng.lai@xtalpi.com)

## Abstract
Mining the embedded information in antibody amino acid sequences can benefit antibody property prediction and novel therapeutic development. While protein-specific pre-training models have been useful in extracting latent representations from protein sequences, there is a need for improvement in models specifically for antibody sequences. Existing models often do not fully consider the unique differences between protein sequences and language sequences, nor do they account for the distinct features of antibodies.

The PARA model addresses these gaps by employing a training strategy that conforms to antibody sequence patterns, using an advanced NLP self-encoding model structure. Our results demonstrate that PARA significantly outperforms selected antibody pre-training models, highlighting its capability to capture antibody sequence information effectively.

## Results
PARA has been evaluated on several tasks and has shown to outperform other published pre-training models of antibodies, indicating its superior ability to capture the nuances of antibody sequence information.

## Usage

In this repository, we have open-sourced the PARA model, accompanied by a prediction script. This script can be utilized to predict missing segments within antibody sequences. Furthermore, it facilitates the extraction of latent vector representations of antibody sequences post-PARA encoding.
