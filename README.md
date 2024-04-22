# Final Project for CS 7643 - Deep Learning, Spring 2024, GaTech

### Title: Evaluating Multimodal Generalization via Few-Shot Learning
Authors: Arthur Nascimento, Ghazal Kaviani, Stefan Faulkner

Idea: Finetune different VLMs for datasets with hidden labels and apply few-shot techniques to learn the unseen labels.


#### Resources:
* Main framework used: [Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models](https://arxiv.org/abs/2301.06267) (CVPR 2023)
    * **For internal use: their README is in this repo's root as `README-SOURCE.md`. Remember to name your environment as "CS7643_project" for consistency (_i.e._,replace `conda create -n cross_modal python=3.9` by `conda create -n CS7643_project python=3.9` instead.)**
* Datasets: We are currently using the 3 datasets listed below. **Follow the instructions in [DATASETS.md](DATASETS.md) to properly install each. It is very important that the file structure matches**. If using other datasets, make sure to poperly edit all files that explicitly call a dataset dictionary (at least all .sh, train.py), which should be at the very top part of the code.
    * Caltech101
    * Oxford_Flowers (Flowers102)
    * UCF101

* Pre-trained models: We are currently using CLIP and ViT as the only encoders. However, we are looking at other VLMs to use as backbones. For a fair comparison, we will use VLMs which were trained jointly for images and text:
    * [CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
    * [FLAVA](https://huggingface.co/docs/transformers/main/en/model_doc/flava)
    * [BLIP](https://huggingface.co/docs/transformers/main/en/model_doc/blip)
    * [BridgeTower](https://huggingface.co/docs/transformers/main/en/model_doc/bridgetower)
    * [LiT](https://huggingface.co/docs/transformers/main/en/model_doc/vision-text-dual-encoder)


#### Creating the environment: 

If you are using Ubuntu, have Conda installed and CUDA 12.0 or greater already properly installed (check with `nvidia-smi`), you can skip the above environment instructions by recreating an enviroment from [environment.yml](environment.yml).

First, run `conda env create -f environment.yml`. You might need to change pytorch-cuda's version to match what your system supports ([check here](https://pytorch.org/get-started/locally/), and make sure to click in Conda under Packages), and 'prefix' to match your system's file structure.
Then, run `pip install -r requirements.txt` and you should be all set!



## Action plan from April 22 (<u>Due: April 24</u>)
* Incorporate all the changes we talked about to the write up. General structure, related work and methods (Stefan)
* Run with more VLMs (Arthur, Ghazal)
    * Only use linear probing (easier to implement with other VLMs) 
    * Embrace the particularities that each model presents for finetuning
    * Need to consider whether we should scale or not the size of the linear probe according to their number of parameters 
* Analysis (Arthur, Ghazal) 
    * How to report qualitative results (which metrics)? 
* Potential other experiments: 
    * Sweep a different range of learning rate and other hyperparams
    * Add different linear heads


## Action plan from April 19 (<u>Due: April 22</u>)
* Everybody in the same page (everyone) - **_done_**
* Fix issues with [eval.py](eval.py) (Arthur and Stefan) - **_done_**
* Seeds 1, 2, 3; Shots 1, 2, 4, 8, 16 (everyone) - **_noted_**
* Start the writeup (Stefan) - **_in progress, incorporating proposed changes_**
* Run with more VLMs (Arthur, Ghazal) - **_in progress_**
* Other experiments and more anaysis? (Arthur, Ghazal) - **_thinking about_**


## Action plan from April 10 (<u>Due: April 15</u>)

* Create repo and share (Arthur) - **_done_**
* Clone and download datasets (Stefan, Ghazal) - **_done_** 
* Set up environments (everybody - Arthur) - **_done_** 
* Figure out what VLM they use (everybody - Ghazal) - **_not done in time_** 
* Finetune as is with current backbone on 2 datasets - start with Caltech101 and StanfordCars (everybody - Stefan)  - **_rerouted_**
* Search for other compatible VLMs (backbones) - (Stefan)  - **_not done in time_**