# Variance Networks

This repo contains the code for our pre-print paper [Variance Networks: When Expectation Does Not Meet Your Expectations](https://arxiv.org/abs/1803.03764).

## Talk by Dmitry Molchanov

[![ICML 2017 Oral Presentation by Dmitry Molchanov](https://bayesgroup.github.io/images/vn.png)](https://youtu.be/KwfED-brvj8)

## Experiments

We actually have two version of the code:
- **TensorFlow implementation** is done with python 2.7, and will help to reproduce CIFAR results i.e. training variance layers via variational dropout.
- **PyTorch implementation** is a way more accurate and reproduce results on MNIST and the toy problem. It requires python 3.6 and pytorch 0.3.

# Citation

If you found this code useful please cite our paper

```
@article{neklyudov2018variance,
  title={Variance Networks: When Expectation Does Not Meet Your Expectations},
  author={Neklyudov, Kirill and Molchanov, Dmitry and Ashukha, Arsenii and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:1803.03764},
  year={2018}
}
```