<p align="center">
<a href="https://www.utoronto.ca/"><img src="https://camh.echoontario.ca/wp-content/uploads/2019/08/UofT-Logo.png" width="200" height="100"></a>
<a href="http://www.lgcorp.com/innovation/sciencepark/introduce"><img src="logo/lgsp.png" width="200" height="100"></a>
</p>



# CVPR20 CLVision Continual Learning [Challenge](https://sites.google.com/view/clvision2020/challenge) 1'st Place Solution for team UT_LG

**Zheda Mai**(University of Toronto), Hyunwoo Kim(LG Sciencepark), Jihwan Jeong (University of Toronto), Scott Sanner (University of Toronto, Vector Institute)

Contact: zheda.mai@mail.utoronto.ca

Final Ranking: https://sites.google.com/view/clvision2020/challenge/challenge-winners

## Introduction

Continual learning is a branch of deep learning that seeks to strike a balance between learning stability and
plasticity. The CVPR 2020 CLVision Continual Learning for Computer Vision challenge is dedicated to evaluating and advancing the current state-of-the-art continual learning methods.

The challenge will be based on the CORe50 dataset and composed of three tracks:

- **New Instances (NI)**: In this setting 8 training batches of the same 50 classes are encountered over time. Each training batch is composed of different images collected in different environmental conditions.
- **Multi-Task New Classes (Multi-Task-NC)\***: In this setting the 50 different classes are split into 9 different tasks: 10 classes in the first batch and 5 classes in the other 8. *In this case the task label will be provided during training and test*.
- **New Instances and Classes (NIC)**: this protocol is composed of 391 training batches containing 300 images of a single class. No task label will be provided and each batch may contain images of a class seen before as well as a completely new class.

#### Metrics

Each solution will be evaluated across a number of metrics:

1. ***Final Accuracy on the Test Set***: should be computed only at the end of the training.
2. ***Average Accuracy Over Time on the Validation Set***: should be computed at every batch/task.
3. **Total Training/Test time**: total running time from start to end of the main function (in Minutes).
4. **RAM Usage**: Total memory occupation of the process and its eventual sub-processes. Should be computed at every epoch (in MB).
5. **Disk Usage**: Only of additional data produced during training (like replay patterns) and also pre-trained weights. Should be computed at every epoch (in MB).

**Final aggregation metric (CL_score)**: weighted average of the 1-5 metrics (0.3, 0.1, 0.15, 0.125, 0.125 respectively



#### Approach

Our approach is based on Experience Replay, a memory-based continual learning method that has been proved effective in various continual learning problems. The details of the approach can be found in our [paper](CVPR2020_CLVision_challenge.pdf). 



## Reproduce the Result

### Data & Environment

Download the dataset and related utilities:
```bash
sh fetch_data_and_setup.sh
```
Setup the conda environment:
```bash
conda env create -f environment.yml
conda activate clvision-challenge
```


### Reproduce the final results for all tracks

```
sh create_submission.sh
```



The parameters for the final submissions:

- `config/final/nc.yml`
- `config/final/ni.yml`
- `config/final/nic.yml`

The detailed explanation of these parameters can be found in `general_main.py`



### Acknowledgement

The starting code of this repository is from the official starting [repository](https://github.com/vlomonaco/cvpr_clvision_challenge).
