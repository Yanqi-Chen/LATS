# A Unified Framework for Soft Threshold Pruning
This directory contains the code reproducing this paper. The code is modified based on the open-source code of [STR](https://github.com/RAIVNLab/STR).

## Dependency 

The major dependencies of this code are list as below. The detailed ones are listed in `requirements.txt`

```
# Name                    Version
cudatoolkit               10.2.89
cudnn                     8.2.1.32
numpy                     1.21.4
python                    3.7.11 
pytorch                   1.10.0
tensorboard               2.7.0
torchvision               0.11.1
pyyaml                    6.0
```

## Environment

The running of code requires NVIDIA GPU and has been tested on *CUDA 10.2* and *Ubuntu 16.04*. The hardware platform used in our experiments is shown below.

- GPU: Tesla V100
- CPU: Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz

Each trial requires 8 GPUs.

## Usage

**Note:** You may need to specify different names for each experiment using `--name`, or it would be grueling to find the result of an exact trial. The setting of final threshold is in **Appendix I** of the paper.

#### Dense training on ResNet-50:

```shell
python main.py --multigpu 0,1,2,3,4,5,6,7 --config configs/reparam/resnet50-dense.yaml --print-freq 4096 --data <PATH to ImageNet>
```

#### Dense training on MobileNet-V1:

```shell
python main.py --multigpu 0,1,2,3,4,5,6,7 --config configs/reparam/mobilenetv1-dense.yaml --print-freq 4096 --data <PATH to ImageNet>
```

#### S-LATS on ResNet-50:

```shell
python main.py --multigpu 0,1,2,3,4,5,6,7 --config configs/reparam/resnet50-prune.yaml --gradual sinp --flat-width <final threshold D> --print-freq 4096 --data <PATH to ImageNet> --name <Name of this experiment>
```

#### S-LATS on ResNet-50 (1024 batch size):

```shell
python main.py --multigpu 0,1,2,3,4,5,6,7 --config configs/reparam/resnet50-prune.yaml --gradual sinp --flat-width <final threshold D> --batch-size 1024 --lr 0.512 --print-freq 4096 --data <PATH to ImageNet> --name <Name of this experiment>
```

#### PGH on ResNet-50:

```shell
python main.py --multigpu 0,1,2,3,4,5,6,7 --config configs/reparam/resnet50-prune.yaml --gradual sinppgh --flat-width <final threshold D> --print-freq 4096 --data <PATH to ImageNet> --name <Name of this experiment>
```

#### LATS on ResNet-50:

```shell
python main.py --multigpu 0,1,2,3,4,5,6,7 --config configs/reparam/resnet50-prune.yaml --gradual sinp --flat-width <final threshold D> --print-freq 4096 --data <PATH to ImageNet> --low-freq --name <Name of this experiment>
```

#### S-LATS on MobileNet-V1:

```shell
python main.py --multigpu 0,1,2,3,4,5,6,7 --config configs/reparam/mobilenetv1-prune.yaml --gradual sinp --flat-width <final threshold D> --print-freq 4096 --data <PATH to ImageNet> --name <Name of this experiment>
```
