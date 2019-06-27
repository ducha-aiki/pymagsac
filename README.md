# pymagsac

This repository contains an Python wrapper of [MAGSAC](https://arxiv.org/abs/1803.07469.pdf). 
Here Daniel will write, how cool MAGSAC is

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
- A modern compiler with C++11 support


# Acknowledgements

This wrapper part is based on great [Benjamin Jack `python_cpp_example`](https://github.com/benjaminjack/python_cpp_example).