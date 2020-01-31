# Variance Networks
The code for our ICLR 2019 paper on [Variance Networks: When Expectation Does Not Meet Your Expectations](https://arxiv.org/abs/1803.03764).


## Talk video

[![](http://i3.ytimg.com/vi/KwfED-brvj8/maxresdefault.jpg)](https://youtu.be/KwfED-brvj8)

# Code
We actually have two version of the code:
- **TensorFlow implementation** is done with python 2.7, and will help to reproduce CIFAR results i.e. training variance layers via variational dropout.
- **PyTorch implementation** is a way more accurate and reproduces results on MNIST and the toy problem. It requires python 3.6 and pytorch 0.3.

# Citation
If you found this code useful please cite our paper
```
@article{neklyudov2018variance,
  title={Variance Networks: When Expectation Does Not Meet Your Expectations},
  author={Neklyudov, Kirill and Molchanov, Dmitry and Ashukha, Arsenii and Vetrov, Dmitry},
  journal={7th International Conference on Learning Representations},
  year={2019}
}
```
