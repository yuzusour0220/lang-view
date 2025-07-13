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
### TODOs: will complete by 7/19 

- ~~Add code~~ ✅ 
- Env setup instructions
- Checkpoint release
- Data release
- Auto-metric eval scripts


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
