# Understanding 3D Object Interaction from a Single Image

Code release for our paper

**Understanding 3D Object Interaction from a Single Image**

[Shengyi Qian][sq], [David F. Fouhey][dff]

[[`Project Page`](https://jasonqsy.github.io/3DOI/)]  [[`arXiv`](https://arxiv.org/abs/2305.09664)] [[`demo`](https://huggingface.co/spaces/shengyi-qian/3DOI)]

![teaser](docs/teaser.png)

Please check the [project page](https://jasonqsy.github.io/3DOI/) for more details and consider citing our paper if it is helpful:

```
@article{qian2023understanding,
    title={Understanding 3D Object Interaction from a Single Image},
    author={Qian, Shengyi and Fouhey, David F},
    journal={arXiv preprint arXiv:2305.09664},
    year={2023}
}
```

If you are interested in the inference-only code, you can also try our demo code on [hugging face](https://huggingface.co/spaces/shengyi-qian/3DOI).


## Setup

We are using anaconda to set up the python environment. It is tested on pytorch 2.0.1,

```bash
# python
conda create -n monoarti python=3.9
conda activate monoarti

# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# other packages
pip install accelerate
pip install submitit
pip install hydra-core --upgrade --pre
pip install hydra-submitit-launcher --upgrade
pip install visdom
pip install packaging plotly imageio imageio-ffmpeg matplotlib h5py opencv-contrib-python
pip install tqdm wandb pycocotools
```

Create `checkpoints` for all experiments.

```bash
mkdir exps
```

If necessary, download our [pretrained model](https://drive.google.com/file/d/1ZBhUqoflC57JLd0DjOZAzErhN4fDV3c6/view?usp=sharing) and put it at `exps/model_final.pth`


## Dataset

I haven't had enough time to release the 3D Object Interaction dataset officially. Please feel free to email syqian@umich.edu if you want to use it in your project.

## Inference

To test the model on any 3DOI or other dataset, run

```bash
python test.py --config-name sam checkpoint_path=multirun/sam/2023-04-29-23-35-06/0/checkpoints/checkpoint_195.pth output_dir=vis train.batch_size=1 test.mode='export_wall' test.split='test'
```

## Training 

To train our model with segment-anything backbone,

```bash
python train.py --config-name sam
```

To train our model with DETR backbone,

```bash
python train.py --config-name detr
```

## Acknowledgment

We reuse the code of [ViewSeg](https://github.com/facebookresearch/viewseg), [DETR](https://github.com/facebookresearch/detr) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).


[sq]: https://github.com/JasonQSY
[dff]: https://github.com/dfouhey