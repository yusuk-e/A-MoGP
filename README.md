# Aggregated Multi-output Gaussian Processes
This repository contains the code for the paper:
- [Aggregated Multi-output Gaussian Processes with Knowledge Transfer Across Domains](https://arxiv.org/abs/2206.12141)

In this work, we present a multi-output Gaussian process (MoGP) model that infers functions for attributes using multiple aggregate datasets of respective granularities. The experiments demonstrate that the proposed model outperforms in the task of refining coarse-grained aggregate data on real-world datasets: Time series of air pollutants in Beijing and various kinds of spatial datasets from New York City and Chicago.

A preliminary version of this work appeared in the Proceedings of NeurIPS’19:

- [Spatially Aggregated Gaussian Processes with Multivariate Areal Outputs](https://papers.nips.cc/paper/2019/hash/a941493eeea57ede8214fd77d41806bc-Abstract.html)

## Requirements
- Python 3.8.13
- PyTorch 1.12.0
- Cython 0.29.21
- GeoPandas 0.9.0

## Quick example (1D problem)
- [1D_problem/Exp/input.csv](1D_problem/Exp/input.csv) is the INPUT file for [1D_problem/A-MoGP_1d.py](1D_problem/A-MoGP_1d.py). The first line is the target data for the prediction. 
- To run the code:
```
python A-MoGP_1d.py --latent_process 1 # Change as needed
```
- To use knowledge transfer across domains (INPUT file: [1D_problem/Exp/input_trans.csv](1D_problem/Exp/input_trans.csv)):
```
python A-MoGP_trans_1d.py --latent_process 1 # Change as needed
```
## Quick example (2D problem)
- To run the preprocessing code in [2D_problem/boundary](2D_problem/boundary):
```
python setup.py build_ext --inplace
```
```
python make.py --g_scale 0.5
```
- How to use the code ([A-MoGP_2d.py](2D_problem/A-MoGP_2d.py), [A-MoGP_trans_2d.py](2d_problem/A-MoGP_trans_2d.py)) is similar to that of 1D problem.

## Citations
```
@inproceedings{tanaka2019,
 author = {Tanaka, Yusuke and Tanaka, Toshiyuki and Iwata, Tomoharu and Kurashima, Takeshi and Okawa, Maya and Akagi, Yasunori and Toda, Hiroyuki},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Spatially Aggregated Gaussian Processes with Multivariate Areal Outputs},
 volume = {32},
 year = {2019}
}

@misc{tanaka2022,
  author = {Tanaka, Yusuke and Tanaka, Toshiyuki and Iwata, Tomoharu and Kurashima, Takeshi and Okawa, Maya and Akagi, Yasunori and Toda, Hiroyuki},
  title = {Aggregated Multi-output Gaussian Processes with Knowledge Transfer Across Domains},
  publisher = {arXiv},
  doi = {10.48550/ARXIV.2206.12141},
  year = {2022},
}
```
