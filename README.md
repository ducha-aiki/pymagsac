#### This repo is deprecated! The Python wrapper is now merged to the main repo that you can find at [Link](https://github.com/danini/magsac).

# MAGSAC++'s Python Wrapper

This repository contains an Python wrapper of [MAGSAC](https://arxiv.org/abs/1803.07469.pdf) and [MAGSAC++](https://openaccess.thecvf.com/content_CVPR_2020/papers/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.pdf). 

The main repository is at https://github.com/danini/magsac

If you use the algorithm, please cite

```
@inproceedings{barath2019magsac,
	author = {Barath, Daniel and Matas, Jiri and Noskova, Jana},
	title = {{MAGSAC}: marginalizing sample consensus},
	booktitle = {Conference on Computer Vision and Pattern Recognition},
	year = {2019},
}

@inproceedings{barath2019magsacplusplus,
	author = {Barath, Daniel and Noskova, Jana and Ivashechkin, Maksym and Matas, Jiri},
	title = {{MAGSAC}++, a fast, reliable and accurate robust estimator},
	booktitle = {arXiv preprint:1912.05909},
	year = {2019},
}
```

If you use it for fundamental matrix estimation with DEGENSAC turned on, please cite

```
@inproceedings{Chum2005,
  author = {Chum, Ondrej and Werner, Tomas and Matas, Jiri},
  title = {Two-View Geometry Estimation Unaffected by a Dominant Plane},
  booktitle = {CVPR},
  year = {2005},
}
```


# Performance of MAGSAC++

MAGSAC++ is the state of the art according to "RANSAC in 2020" CVPR tutorial's [experiments](http://cmp.felk.cvut.cz/cvpr2020-ransac-tutorial/presentations/RANSAC-CVPR20-Mishkin.pdf).

# Performance of MAGSAC

MAGSAC is the state of the art according to the recent study Yin et.al."[Image Matching across Wide Baselines: From Paper to Practice](https://arxiv.org/abs/2003.01587.pdf)", 2020.

![IMW-benchmark](img/ransacs.png)


![IMW-Challenge](img/ransacs2.png)


# Installation

To build and install `python_cpp_example`, clone or download this repository and then, from within the repository, run:

```bash
python3 ./setup.py install
```

or

```bash
pip3 install .
```

# Example of usage

```python
import pymagsac
H, mask = pymagsac.findHomography(src_pts, dst_pts, 3.0)
F, mask = pymagsac.findFundamentalMatrix(src_pts, dst_pts, 3.0)

```

See also this [notebook](examples/example.ipynb)


# Requirements

- Python 3
- CMake 2.8.12 or higher
- OpenCV 3.4
- A modern compiler with C++14 support


# Acknowledgements

This wrapper part is based on great [Benjamin Jack `python_cpp_example`](https://github.com/benjaminjack/python_cpp_example).
