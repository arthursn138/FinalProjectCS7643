# Final Project for CS 7643 - Deep Learning, Spring 2024, GaTech

Title: Evaluating Multimodal Generalization via Few-Shot Learning \
Authors: Arthur Nascimento, Ghazal Kaviani, Stefan Faulkner

Idea: Finetune different VLMs for datasets with hidden labels and apply few-shot techniques to learn the unseen labels.

Resources:
* Main framework used: [Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models](https://arxiv.org/abs/2301.06267) (CVPR 2023) **--- for internal use: their README is in this repo's root as `README-SOURCE.md`. Remember to name your environment as "final_prohect" for consistency (_i.e._,replace `conda create -n cross_modal python=3.9` by `conda create -n final_project python=3.9` instead.)**
* Datasets: _To be added_
* Pre-trained models: _To be added_

## Action plan from April 10 (<u>Due: April 15</u>)

* Create repo and share (Arthur) - **_done_**
* Clone and download datasets (Stefan, Ghazal) 
* Set up environments (everybody - Arthur) 
* Figure out what VLM they use (everybody - Ghazal) 
* Finetune as is with current backbone on 2 datasets - start with Caltech101 and StanfordCars (everybody - Stefan) 
* Search for other compatible VLMs (backbones) - (Stefan) 