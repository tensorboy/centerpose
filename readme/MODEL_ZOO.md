# MODEL ZOO

### Common settings and notes

- The experiments are run with pytorch 1.1, CUDA 10.1, and CUDNN 7.5.
- Training times are measured on our servers with 8 gtx1080ti GPUs.
- Testing times are measured on our local machine with gtx1080ti GPU. 
- The models can be downloaded directly from [Google drive](https://drive.google.com/open?id=1UG2l8XtjOfBtG_GLpSdxlWS2wxFR8hQF).

## Object Detection

### COCO

| Model                    | GPUs |Train time(h)| Test time (ms) |   AP        |  Download | 
|--------------------------|------|-------------|----------------|-------------|-----------|
|[multi\_pose\_hg_1x](../experiments/multi_pose_hg_1x.sh)    |   5  |62           | 151            | 58.7        | [model](https://drive.google.com/open?id=1HBB5KRaSj-m-vtpGESm7_3evNP5Y84RS) |
|[multi\_pose\_hg_3x](../experiments/multi_pose_hg_3x.sh)    |   5  |188          | 151            | 64.0        | [model](https://drive.google.com/open?id=1n6EvwhTbz7LglVXXlL9irJia7YuakHdB) |
|[multi\_pose\_dla_1x](../experiments/multi_pose_dla_1x.sh)   |   8  |30           | 44             | 54.7        | [model](https://drive.google.com/open?id=1VeiRtuXfCbmhQNGV-XWL6elUzpuWN-4K) |
|[multi\_pose\_dla_3x](../experiments/multi_pose_dla_3x.sh)   |   8  |70           | 44             | 58.9        | [model](https://drive.google.com/open?id=1PO1Ax_GDtjiemEmDVD7oPWwqQkUu28PI) |
|[resnet_50](../experiments/multi_pose_dla_3x.sh)   |   8  |70           | 44             | 58.9        | [model](https://drive.google.com/open?id=1P2Hub3g9I-w43hDXffxJ9PBwfl0k14Te) |

#### Notes
- All models are trained on keypoint train 2017 images which contains at least one human with keypoint annotations (64115 images).
- The evaluation is done on COCO keypoint val 2017 (5000 images).
- Flip test is used by default.
- The models are fine-tuned from the corresponding center point detection models.
- Dla training schedule: `1x`: train for 140 epochs with learning rate dropped 10 times at the 90 and 120 epoch.`3x`: train for 320 epochs with learning rate dropped 10 times at the 270 and 300 epoch.
- Hourglass training schedule: `1x`: train for 50 epochs with learning rate dropped 10 times at the 40 epoch.`3x`: train for 150 epochs with learning rate dropped 10 times at the 130 epoch.
