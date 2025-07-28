# [CVPR 25 Highlight] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos
This repository will very soon contain the PyTorch code for our **CVPR 2025 paper** and the associated datasets: 

<!--[Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](https://vision.cs.utexas.edu/projects/lang-view)<br />-->
*Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos*<br/>
[Sagnik Majumder](https://sagnikmjr.github.io/), [Tushar Nagarajan](https://tushar-n.github.io/), [Ziad Al-Halah](https://users.cs.utah.edu/~ziad/), [Reina Pradhan](https://www.linkedin.com/in/reina-pradhan), [Kristen Grauman](https://www.cs.utexas.edu/~grauman)<br />


Project website: [https://vision.cs.utexas.edu/projects/lang-view](https://vision.cs.utexas.edu/projects/lang-view/)

<p align="center">
  <img src="concept.png" width="1124" height="350">
</p>

## Abstract
Given a multi-view video, which viewpoint is most informative for a human observer? Existing methods rely on heuristics or expensive "best-view" supervision to answer this question, limiting their applicability. We propose a weakly supervised approach that leverages language accompanying an instructional multi-view video as a means to recover its most informative viewpoint(s). Our key hypothesis is that the more accurately an individual view can predict a view-agnostic text summary, the more informative it is. To put this into action, we propose LangView, a framework that uses the relative accuracy of view-dependent caption predictions as a proxy for best view pseudo-labels. Then, those pseudo-labels are used to train a view selector, together with an auxiliary camera pose predictor that enhances view-sensitivity. During inference, our model takes as input only a multi-view video--no language or camera poses--and returns the best viewpoint to watch at each timestep. On two challenging datasets comprised of diverse multi-camera setups and how-to activities, our model consistently outperforms state-of-the-art baselines, both with quantitative metrics and human evaluation.


## Code and Datasets
### TODOs: hope to complete all by 8/3

- ~~Code~~ ✅ 
- ~~Env setup instructions~~  ✅ 
- ~~Train and test bash commands for Ego-Exo4D~~ ✅ 
- ~~Checkpoint release  for Ego-Exo4D~~ ✅ 
- ~~Data release for Ego-Exo4D~~ ✅
- ~~Auto-metric eval scripts~~ ✅ 
- ~~```train.py```, ```test.py``` for LEMMA~~✅
- Auto-metric eval scripts for LEMMA
- ~~Train and test bash commands for LEMMA~~✅
- ~~Checkpoint release for LEMMA~~✅
- ~~Data release for LEMMA~~ ✅

## Dependencies
This code has been tested with ```python 3.9.18``` with ```torch 2.2.2+cu121``` and ```torchvision 0.17.2+cu121```. Additional python package requirements are available in ```requirements.txt```.   
  
Install the remaining dependencies either by 
```
pip3 install -r requirements.txt
``` 
or by parsing ```requirements.txt``` to get the names and versions of individual dependencies and install them individually.

## Data
<!-- Data link and extraction instructions COMING SOON! -->

Download the data segments from [this link](https://utexas.box.com/s/mrp2w8qq8kxqpzlwlghnxj1d2i5bvnt3), copy them to the repo root and run the following commands: 
```
cat data_part_* > data.tar
tar -xvf data.tar
```

For LEMMA frames, download the data from the [dataset website](https://sites.google.com/view/lemma-activity/home/dataset?authuser=0) and link ```data/lemma/datapoint_images``` to the ```data-002``` directory in the downloaded data directory.


## Run commands, tested with 8 V100s
Download the EgoVLPv2 pretrained checkpoint from [this link](https://utexas.box.com/shared/static/0ma3omfj7eb94kqvg0kg8qe5mxdnasxr.zip) and put it at this path: ```pretrained_checkpoints/egovlpV2_model_best_egoExo30nov2024.pth```.

Download the Lang-View checkpoint directory from [this link](https://utexas.box.com/shared/static/u29w5455dupm5rexbuworwxzxduitof6.zip) and put it at this path: ```runs```

###### Ego-Exo4D training
```
python3 train.py --run-dir runs/egoExo4d_release --log-tb --data-parallel --use-datapointVideoClips --randomize-trainViewOrder --unfreeze-videoEncoder --use-minMultiHotLoss --trainDatapoints-filePath data/labels/train/videoLlama_cider_all3Agree.pkl,data/labels/train/videoLlamaWvicuna_cider_all3Agree.pkl,data/labels/train/videoChat2_cider_all3Agree.pkl --valDatapoints-filePath data/labels/val/videoLlama_cider_all3Agree.pkl,data/labels/val/videoLlamaWvicuna_cider_all3Agree.pkl,data/labels/val/videoChat2_cider_all3Agree.pkl --multiBestViewAggregator-multiPseudoLabler --use-relativeCameraPoseLoss --maskOut-invalidRelativeCameraPoseLoss-inTraining --relativeCameraPoseLoss-rotationInAngles --relativeCameraPoseLoss-rotationAsClasses --relativeCameraPoseLoss-coordsInAngles --relativeCameraPoseLoss-coordsAsClasses 
```

###### Ego-Exo4D testing
<!-- Download the Ego-Exo4D checkpoint from [this link](https://utexas.box.com/shared/static/x56paq0un6f2y8xkcorhbl5jkndajhiv.zip) and put it at this path: ```runs/egoExo4d_release/data/valBestCkpt_maxCaptioningScore.pth``` -->

```
python3 test.py --run-dir runs/egoExo4d_release --data-parallel --use-datapointVideoClips --unfreeze-videoEncoder --use-relativeCameraPoseLoss --relativeCameraPoseLoss-rotationInAngles --relativeCameraPoseLoss-rotationAsClasses --relativeCameraPoseLoss-coordsInAngles --relativeCameraPoseLoss-coordsAsClasses
```

To compute auto-metrics, run the following scripts: ```scripts/ego_exo4d/format_predictedVIewScores.ipynb```, ```scripts/ego_exo4d/run_captioningMetrics.py``` and ```scripts/ego_exo4d/compute_captioningScores.ipynb``` one after the other. 

###### LEMMA training
```
python3 train_lemma.py --run-dir runs/lemma_release --data-parallel --isLemma-dataset --use-datapointVideoClips --randomize-trainViewOrder --unfreeze-videoEncoder --use-minMultiHotLoss --trainDatapoints-filePath data/lemma/labels/train/videoLlama_cider_all3Agree.pkl,data/lemma/labels/train/videoLlamaWvicuna_cider_all3Agree.pkl,data/lemma/labels/train/videoChat2_cider_all3Agree.pkl --valDatapoints-filePath data/lemma/labels/val/videoLlama_cider_all3Agree.pkl,data/lemma/labels/val/videoLlamaWvicuna_cider_all3Agree.pkl,data/lemma/labels/val/videoChat2_cider_all3Agree.pkl --multiBestViewAggregator-multiPseudoLabler --use-egovlpV2-patchLevelVisualFeats
```
###### LEMMA testing
```
python3 test_lemma.py --isLemma-dataset --data-parallel --run-dir runs/lemma_release --use-datapointVideoClips --unfreeze-videoEncoder --use-egovlpV2-patchLevelVisualFeats
```
Auto-metric computation scripts COMING SOON!


## Citation
```
@InProceedings{Majumder_2025_CVPR,
    author    = {Majumder, Sagnik and Nagarajan, Tushar and Al-Halah, Ziad and Pradhan, Reina and Grauman, Kristen},
    title     = {Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {29016-29028}
}
```

# License
This project is released under the MIT license, as found in the LICENSE file.
