# [CVPR 25 Highlight] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos
This repository will very soon contain the PyTorch code for our **CVPR 2025 paper** and the associated datasets: 

[Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos]([https://vision.cs.utexas.edu/projects/ego_av_corr/](https://vision.cs.utexas.edu/projects/lang-view/))<br />
Sagnik Majumder, Tushar Nagarajan, Ziad Al-Halah, Reina Pradhan, Kristen Grauman<br />


Project website: [https://vision.cs.utexas.edu/projects/lang-view](https://vision.cs.utexas.edu/projects/lang-view/)

<p align="center">
  <img src="concept.png" width="500" height="350">
</p>

## Abstract
Given a multi-view video, which viewpoint is most informative for a human observer? Existing methods rely on heuristics or expensive "best-view" supervision to answer this question, limiting their applicability. We propose a weakly supervised approach that leverages language accompanying an instructional multi-view video as a means to recover its most informative viewpoint(s). Our key hypothesis is that the more accurately an individual view can predict a view-agnostic text summary, the more informative it is. To put this into action, we propose LangView, a framework that uses the relative accuracy of view-dependent caption predictions as a proxy for best view pseudo-labels. Then, those pseudo-labels are used to train a view selector, together with an auxiliary camera pose predictor that enhances view-sensitivity. During inference, our model takes as input only a multi-view video--no language or camera poses--and returns the best viewpoint to watch at each timestep. On two challenging datasets comprised of diverse multi-camera setups and how-to activities, our model consistently outperforms state-of-the-art baselines, both with quantitative metrics and human evaluation.


## Code and Datasets
Coming soon!!



## Citation
```
@article{majumder2024viewpoint,
  title={Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Videos},
  author={Majumder, Sagnik and Nagarajan, Tushar and Al-Halah, Ziad and Pradhan, Reina and Grauman, Kristen},
  journal={arXiv preprint arXiv:2411.08753},
  year={2024}
}
```

# License
This project is released under the MIT license, as found in the LICENSE file.
