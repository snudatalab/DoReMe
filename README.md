# Domain-Aware Data Selection for Speech Classification via Meta-Reweighting

This is the code repository for Domain-Aware Data Selection for Speech Classification via Meta-Reweighting.
This includes the implementation of DoReMe (**Do**main-Awa**Re** Data Selection for Speech Classification via **Me**ta-Reweighting)
our novel approach for the speech classification.

## Abstract
Given speeches from diverse domains, how can we train an accurate classifier for a specific target domain utilizing the other source domains?
The problem commonly arises in real-world scenarios, such as identifying the intents of speeches from individuals with a specific speech disorder using speeches of other disorders.
However, previous data selection methods for utilizing the source instances encounter two main challenges: they cannot consider the diversities of source domains, and their hard selecting schemes may ignore helpful source instances if the given information of the target domain is insufficient.
In this work, we propose DoReMe, a domain-aware data selection method for accurate speech classification on a target domain.
The key idea is to softly select source instances by dynamically assigning important scores to each instance based on two similarities: instance-scores and domain-scores.
Extensive experiments show that DoReMe achieves the best classification performance.

## Requirements

We recommend using the following versions of packages:
 - pytorch==1.13.1
 - tqdm==4.66.2
 - pandas==2.0.3
 - torcheval==0.0.7
 - scikit-learn==1.3.2

## Data Overview
We use two datasets.
Download the datasets from the official links.

|        **Dataset**        |                  **Link**                   | 
|:-------------------------:|:-------------------------------------------:| 
|       **Skit-S2I**        |           `https://github.com/skit-ai/speech-to-intent-dataset`           | 
|       **zenodo**        |           `https://github.com/RiTA-nlp/ITALIC/`           | 

## How to Run
You can run the demo script in the directory by the following code.
```
python main.py
```
