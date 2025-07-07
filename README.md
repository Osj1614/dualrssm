# Self-supervised risk factor model using dual Recurrent State Space Models

Dual RSSM is a framework for learning risk factor model that separately models the overall market and individual assets.

The algorithm is based on [Self-supervised risk factor model using dual Recurrent State Space Models](https://www.sciencedirect.com/science/article/abs/pii/S0950705125010810).

This implementation uses a pytorch.


# Requirements

## Dependencies

```
torch>=2.0
numpy>=1.23
pandas>=2.0
scikit-learn>=1.3
tqdm>=4.60
```


## Dataset

This repository requires your own dataset of assets for running.

Please refer to jupyter notebook files on raw_data/ for preprocessing your dataset.


# Training a model

Before running, make sure your dataset is properly directed in rssm/trainer.py.

Other hyperparameter settings could be also set on rssm/trainer.py.

To train the dual rssm model, run the following command.
```
python train_rssm.py
```


# Reference 
```
@article{lee2025dualrssm,
title = {Self-supervised risk factor model using dual Recurrent State Space Models},
journal = {Knowledge-Based Systems},
pages = {114036},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.114036},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125010810},
author = {Ji-hun Lee and Seungjun Oh and Jongchan Park and Da-Hea Kim and Yusung Kim},
keywords = {Risk factor model, Recurrent State Space Model, Market Pricing},
}
```
