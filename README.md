# Right Lane segmentation in Duckietown environment
Repository for private project Duckietown at BME VIK (Budapesti Műszaki és Gazdaságtudományi Egyetem Villamosmérnöki és Informatikai Kar).

## Data generation for semantic segmentation in Duckietown environment
For all machine learning task data quality and quantity is of crucial importance.
On one hand more data helps the feature extractor to better generalize while data diversity ensures that the model has seen all kind of possible inputs during training.
Tipically, labelled databases are not publicly available for all kind of tasks and that is especially true for such new challenges sa the Duckietown competition.

Luckily, there is a publicly available open source simulator software that was designed for reinforcement learning tasks and that is free to use, that means with some modifications we were able to extract (or record) some runs that were saved both as an observation picture as well as labelled picture for semantic segmentation.
After some post-processing, observation and label videos were produced that helps in building some supervised or semi-supervised machine learning algorithm.

If we are interested in pictures only, not videos, then these can be taken apart into separate images.

For more information on how the data collection and preprocessing was done, please refer to [this wiki page](TODO site).

### Example training data
Data pair 1:

![Original](doc/res_readme/orig_1.jpg)
![Annotated](doc/res_readme/annot_1.jpg)

Data pair 2:

![Original](doc/res_readme/orig_2.jpg)
![Annotated](doc/res_readme/annot_2.jpg)

It is planned to extend both this description as well as the wiki pages.

## Segmentation network
All the code related to neural networks are implemented in PyTorch and, recently, in PyTorch Lightning.
Versions are always updated, so I plan to use the newest version possible.
If this repo is finalised, or I wish to make a tag of it then I will add specific version informations.

### Database preprocessing, formatting
Simulation database directory structure before preprocessing is expected to be as follows:
```
simData
├── input
└── label
```
After preprocessing:
```
simData
├── test
│   ├── input
│   └── label
├── train
│   ├── input
│   └── label
└── valid
    ├── input
    └── label
```

Real image database directory structure before preprocessing is expected to be as follows:
```
realData
├── input
├── label
└── unlabelled
```
After preprocessing:
```
realData
├── test
│   ├── input
│   └── label
├── train
│   ├── input
│   └── label
└── unlabelled
    └── input
```

When SSDA training is used, the following directory structure is expected (after preprocessing):
```
dataSSDA
├── source
│   ├── input
│   └── label
└── target
    ├── test
    │   ├── input
    │   └── label
    ├── train
    │   ├── input
    │   └── label
    └── unlabelled
        └── input
```

### Base architectures
I have tried a couple of segmentation networks that were:
- A custom network consisting of an encoder path and a decoder path, named EncDecNet among the models.
- FC-DenseNet, that offers less parameters with same accuracy. [Paper](https://arxiv.org/abs/1611.09326), code copied from [here](https://github.com/bfortuner/pytorch_tiramisu).

After some initial trials currently FC-DenseNet57 is in use.
It proved to be easy to train and also offered encouraging accuracy on real images without any domain adaptation technique applied.

### Domain adaptation
However good or encouraging was the result on the real image test it was still not good enough to apply it as a part of another system, to build on it.
So after some advice I tried Semi-Supervised Domain Adaptation with Minimax Entropy (short terms are SSDA MME) according to [this](https://arxiv.org/pdf/1904.06487.pdf) paper.
Fortunately the authors also publicised their own code written in PyTorch, found [here](https://github.com/VisionLearningGroup/SSDA_MME) and it is the current focus to apply the methods they have implemented.

With domain adaptation my hope is that the domain gap can easily be stepped over in this segmentation task.

## Usage
The training process consisted of three (plus one) separate model training:
- Simulator baseline training
- Same training on domain-converted sim data (via CycleGAN - plus one training)
- SSDA MME training

### Simulator baseline
Simulator baseline training can be reproduced using the following command:
```
CUDA_VISIBLE_DEVICES=0 python3.6 RightLaneModule.py --gpus=1 --dataPath ./simData --batchSize 64 --max_epochs 150
```

Naturally CUDA and GPU dependent parameters can be changed to better utilize the hardware.

### Domain Transformation using CycleGAN
After CycleGAN training for sim-to-real conversion the training process is the same.
Simulator baseline training can be reproduced using the following command:
```
CUDA_VISIBLE_DEVICES=0 python3.6 RightLaneModule.py --gpus=1 --dataPath ./simData --batchSize 64 --max_epochs 150
```

### SSDA via Minimax Entropy
The training script uses the baseline weights to shorten the training process.
Otherwise it has the same parameters as the baseline training.
Simulator baseline training can be reproduced using the following command:
```
CUDA_VISIBLE_DEVICES=0 python3.6 RightLaneMMEModule.py --gpus=1 --dataPath ./realData --pretrained_path results/FCDenseNet57weights.pth --batchSize 64 --max_epochs 150
```

### Testing
Trained model evaluation can be done using the provided script test.py.
The script takes command line arguments, see help for details.

Comparison of trained models can be done using comparison.py that generates an image file with sample predictions of each model.

One is able to make predictions for a video (stream of images) via makeDemoVideo.py.
