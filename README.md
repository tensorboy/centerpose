# The repo is based on [CenterNet](https://arxiv.org/abs/1904.07850), which aimed for push the boundary of human pose estimation
multi person pose estimation using center point detection:
![](readme/fig2.png)

## Main results

### Keypoint detection on COCO validation 2017
<p align="center"> <img src='readme/performance.png' align="center" height="512px"></p>
<center>
| Backbone     |  AP       |  FPS         | Download | 
|--------------|-----------|--------------|----------|
|DLA-34        | 62.3      |    23      |   [model](https://drive.google.com/open?id=151aD93nHG_oGju1xxOmwoDNjfeif6uGi)  |
|Resnet-50     | 53.0      |    28      |   [model](https://drive.google.com/open?id=1k_kpn7tCpX4CHEEiCqdNxLRXZc-ky-uY)  |
|MobilenetV3   | 45.1      |    30      |   [model](https://drive.google.com/open?id=1T8_YsPiW7EmLHQfh_Zk37hTsiJpdaAN1)  |
|ShuffleNetV2  | 34.6      |    25      |   [model]()  |
|High Resolution| 49.5     |    16      |   [model]()  |
|HardNet| 34.6     |    30        |   [model]()  |
</center>
## Installation

git submodule init&git submodule update
Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

We support demo for image/ image folder, video, and webcam. 

First, download the model [multi_pose_dla_1x](https://drive.google.com/open?id=151aD93nHG_oGju1xxOmwoDNjfeif6uGi) for human pose estimation) 
from the [Model zoo](https://drive.google.com/open?id=1UG2l8XtjOfBtG_GLpSdxlWS2wxFR8hQF) and put them in anywhere.

Run:
    
~~~
python demo.py --cfg ../experiments/res_50_512x512.yaml --TESTMODEL /your/model/path/res_50_1x.pth --DEMOFILE ../images/33823288584_1d21cf0a26_k.jpg --DEBUG 1
~~~
The result for the example images should look like:


## Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets.

We provide config files for all the experiments in the [experiments](experiments) folder.

```
cd ./tools python -m torch.distributed.launch --nproc_per_node 4 train.py --cfg ../experiments/*yalm
```

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
