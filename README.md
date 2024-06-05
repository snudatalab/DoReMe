# Domain-Aware Data Selection for Speech Classification via Meta-Reweighting

This is the code repository for Domain-Aware Data Selection for Speech Classification via Meta-Reweighting.
This includes the implementation of DOREME (**DO**main-Awa**RE** Data Selection for Speech Classification via **ME**ta-Reweighting)
our novel approach for the speech classification.

## Abstract
Given speeches from diverse domains, how can we train an accurate classifier for a specific target domain utilizing the other source domains?
The problem commonly arises in real-world scenarios, such as identifying the intents of speeches from individuals with a specific speech disorder using speeches of other disorders.
However, previous data selection methods for utilizing the source instances encounter two main challenges: they cannot consider the diversities of source domains, and their hard selecting schemes may ignore helpful source instances if the given information of the target domain is insufficient.
In this work, we propose \method (\methodlong), a domain-aware data selection method for accurate speech classification on a target domain.
The key idea is to softly select source instances by dynamically assigning important scores to each instance based on two similarities: \emph{instance-scores} and \emph{domain-scores}.
Extensive experiments show that \method achieves the best classification performance.
## Requirements

We recommend using the following versions of packages:
###This Code is built for pytorch GPU version. we highly recommend you to install pytorch for the GPU version!
 - pytorch==1.13.1
 - tqdm==4.66.2
 - pandas==2.0.3
 - torcheval==0.0.7
 - scikit-learn==1.3.2

## Data Overview
The datasets are available at (https://drive.google.com/drive/folders/1FlIKRBzJzapUU2PfZgqI8zNlR0e_ALiJ?usp=drive_link).

|        **Dataset**        |                  **Path**                   | 
|:-------------------------:|:-------------------------------------------:| 
|       **Speech-to-intent**        |           `speech-to-intent/`           | 
|       **zenodo**        |           `zenodo/`           | 

## How to Run
You can run the demo script in the directory by the following code.
```
python main.py
```