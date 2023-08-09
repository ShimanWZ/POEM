# POEM: Partial Observation Experts Modelling
This repository contains the official PyTorch implementation of POEM (Partial Observation Experts Modelling), a contrastive/meta-learning approach for learning representations from partial observations. This approach was introduced in this paper:

[Contrastive Meta-Learning for Partially Observable Few-Shot Learning](https://arxiv.org/abs/2301.13136) (accepted for publication at ICLR 2023)

![image](resources/POEM_Figure1.jpg)
## Installation

Clone the repository (including submodules for use for learning representations of RL agent environments). Create the specified conda environment and install the gym-minigrid package, as below:

```
git clone --recurse-submodules https://github.com/AdamJelley/POEM
conda env create --file environment.yml
conda activate POEM
pip install -e ./gym-minigrid
```

Note: If you have already cloned the project and forgot `--recurse-submodules`, to add the submodules required to run the RL agent environment experiments you can run:
```
git submodule update --init
```

## Usage
### Toy Few-Shot Learning Experiments
Example: Training POEM on partial observations of MiniImageNet:
```
python -m FSL.main --dataset miniimagenet --learner POEM --cropping --use_coordinates --num_crops 5 --n_way 20 --n_support 1 --n_query 5
```
To list full configuration options run:
```
python -m FSL.main --help
```
Toy experiments use on the [torchmeta](https://github.com/tristandeleu/pytorch-meta) package, so the required MiniImageNet and Omniglot datasets can be downloaded automatically. However, torchmeta limits the torchvision version (to torchvision<0.11.0 and >=0.5.0, as in e.g. [this issue](https://github.com/tristandeleu/pytorch-meta/issues/161)). If you don't intend to run the toy experiments you can upgrade the the PyTorch/Torchvision packages from those specified in the environment file.

### Meta-Dataset Benchmarking
The full Meta-Dataset benchmarking reported in the paper was carried out using the [GATE (Generalisation After Transfer Evaluation)](https://github.com/BayesWatch/POEM-Bench) framework, to ensure strict evaluation procedures and using state-of-the-art backbones (as specified) for fair comparisons with baselines. Our benchmarking codebase is available in the repository [POEM-Bench](https://github.com/BayesWatch/POEM-Bench).

Final results on Partially-Observable Meta-Dataset (as introduced in our paper) with ResNet-18 backbones:

| **Test Source** | **Finetune**   | **ProtoNet** | **MAML**   | **POEM**       |
|:---------------:|:--------------:|:------------:|:----------:|:--------------:|
| **Aircraft**    | 46.5+/-0.6     | 48.5+/-1.0   | 37.5+/-0.3 | **55.3+/-0.7** |
| **Birds**       | 62.6+/-0.7     | 67.4+/-1.2   | 52.5+/-0.6 | **71.1+/-0.1** |
| **Flowers**     | 48.5+/-0.4     | 46.4+/-0.7   | 33.5+/-0.3 | **49.2+/-1.5** |
| **Fungi**       | 61.0+/-0.2     | 61.4+/-0.4   | 46.1+/-0.4 | **64.8+/-0.3** |
| **Omniglot**    | 71.3+/-0.1     | 87.8+/-0.1   | 47.4+/-1.0 | **89.2+/-0.7** |
| **Textures**    | **83.2+/-0.4** | 76.7+/-1.6   | 73.1+/-0.4 | 81.4+/-0.6     |
