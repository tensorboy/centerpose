# The repo is based on [CenterNet](https://arxiv.org/abs/1904.07850), which aimed for push the boundary of human pose estimation
multi person pose estimation using center point detection:
![](readme/fig2.png)

## Main results

### Keypoint detection on COCO validation

| Backbone     |  AP       |  FPS         | Download | 
|--------------|-----------|--------------|----------|
|DLA-34        | 60.5      |    23        |   [model](https://drive.google.com/open?id=151aD93nHG_oGju1xxOmwoDNjfeif6uGi)  |
|Resnet-50     | 53.0      |    40        |   [model](https://drive.google.com/open?id=1k_kpn7tCpX4CHEEiCqdNxLRXZc-ky-uY)  |
|MobilenetV3   | 45.1      |    20        |   [model](https://drive.google.com/open?id=1T8_YsPiW7EmLHQfh_Zk37hTsiJpdaAN1)  |

## Installation

git submodule init&git submodule update
Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

We support demo for image/ image folder, video, and webcam. 

First, download the model [multi_pose_dla_1x](https://drive.google.com/open?id=151aD93nHG_oGju1xxOmwoDNjfeif6uGi) for human pose estimation) 
from the [Model zoo](https://drive.google.com/open?id=1UG2l8XtjOfBtG_GLpSdxlWS2wxFR8hQF) and put them in anywhere.

Run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../models/ctdet_coco_dla_2x.pth
~~~
We provide example images in `CenterNet_ROOT/images/` (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/demo)). If set up correctly, the output should look like

<p align="center"> <img src='readme/det1.png' align="center" height="230px"> <img src='readme/det2.png' align="center" height="230px"> </p>

For webcam demo, run     
~~~
python demo.py multi_pose --demo /path/to/image/or/folder/or/video/or/webcam --load_model ../models/multi_pose_dla_3x.pth
~~~
The result for the example images should look like:

<p align="center">  <img src='readme/pose1.png' align="center" height="200px"> <img src='readme/pose2.png' align="center" height="200px"> <img src='readme/pose3.png' align="center" height="200px">  </p>

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets.

We provide config files for all the experiments in the [experiments](experiments) folder.

```
cd ./tools python -m torch.distributed.launch --nproc_per_node 4 train.py --cfg ../experiments/*yalm
```

## Develop

If you are interested in training CenterNet in a new dataset, use CenterNet in a new task, or use a new network architecture for CenterNet, please refer to [DEVELOP.md](readme/DEVELOP.md). Also feel free to send us emails for discussions or suggestions.

## License

MIT License (refer to the LICENSE file for details).

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }
