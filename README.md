# TensorFlow Implementation of the Stacked Tensorial Neural Network (STNN) Architecture

## Introduction

This repository implements the neural network architecture from [(Wagner, 2023)](#References), called a "stacked tensorial neural network" (STNN).
It also provides tools for applying the network to the PDE problem considered from the paper. 

Roughly speaking, the STNN combines multiple tensorial neural networks [(Wang, et al, 2023)](#References) into a single network that seeks to represent the PDE solution over a range of parameters (e.g., domain geometry). Capturing the parametric dependence is important because it means the network does not have to be retrained when the problem's parameters change.

## Requirements

The core features require TensorFlow 2.15.0 or greater. Also, handling the PDE system requires the sparse
linear algebra library from either [SciPy](https://scipy.org/) or [CuPy](https://cupy.dev/). See `requirements.txt` for more information and other dependencies.

The current release (1.1.0) has been tested on Windows 10 using Python 3.11.5, TensorFlow 2.15.0, and CuPy 13.0.0. 

## Installation

To install the latest version, download or clone the repository and run `pip install .` from the root
directory. For example,
> git clone https://github.com/caleb399/stacked_tensorial_nn.git \
> cd stnn \
> pip install .

To manage package dependencies, it is recommended to install in a fresh Python virtual environment.

## Usage
- **Loading a model:**
  - To load an STNN model into TensorFlow, you can use the `build_stnn` function in `stnn/nn/stnn.py`. The function's
  argument should be a Python dictionary containing the STNN configuration. The dictionary can be populated manually or loaded
  from an existing configuration file. 
  See `stnn/nn/stnn.py` for details and `examples/stnn_config.json` for an example.
- **Making Predictions:**
  - To make predictions with the loaded model, use the command `rho = model.predict([params, bf])`.  Examples of optimized inference using
  OpenVINO can be found in `examples/inference`.
  - The `params` array should have the shape `(N, 3)`, where `N` is the batch size. It should contain the PDE parameters `(ell, a2, ecc)`.
  - The `bf` array contains the boundary data as described in the paper, with shape `(N, 2*nx2, nx3)`. The class method `PDEsystem..generate_random_bc` in `stnn/pde/pde_system.py` can be used to generate random boundary conditions as described in the paper.
  - The output `rho` is an array with shape `(N, nx1, nx2)`, representing the density `rho(x1,y1) = int(f(x1,x2,x3), x3 = -pi..pi)`.
- **Directly solving the PDE:**
  - The function `direct_solution` in `examples/inference.py` demonstrates solving the PDE system directly using the sparse linear algebra backend, which is driven by either `scipy` or `cupy` (see `stnn/linalg_backend.py`). Note that `cupy` is much faster than `scipy` for the problems considered here (up to 20x speedup), but it requires an NVIDIA GPU.
- **Visualization:**
  - For visualization of predictions, you can use the `plot_comparison` function in `stnn/utils/plotting.py`.

## Organization

The main components are:

#### Submodule "nn"
- Houses the TensorFlow implementation of the STNN architecture.
- Tensor networks are built using the [t3f](https://github.com/Bihaqo/t3f).

#### Submodule "pde"
- Contains the `PDESystem` class encapsulating the finite-difference discretization of the PDE.
- Attributes include domain grids and sparse matrices representing the discretized PDE.
- Grids and matrices construction occurs in `ellipse.py` and `circle.py` for elliptical and circular domains, respectively.

#### Submodule "data"
- Tools for generating and preprocessing labeled data.
- Here, the boundary conditions and PDE parameters are the inputs, and the PDE solution is the output (the "label"). Generating data involves solving the PDE for a large sets of inputs, which boils down to solving a large number of high-dimesional (~500,000) sparse linear systems.
- For solving the linear systems, `stnn/linalg_backend.py` provides interfaces to the SciPy and CuPy sparse linear algebra libraries.

#### Submodule "utils"
- tools for I/O operations and post-processing (visualization, statistics)

#### Additional Sections:

- `/paper`: Contains scripts for validating the results in (Wagner, 2023), including model weights from Table 1 of the paper. 
Due to size limitations, the datasets used for training and testing are not included in the current release (1.1.0),
but will be made available in the future once a suitable hosting platform has been found.
- `/examples`: Demonstrates model optimization for inference using [TensorFlow Lite](https://www.tensorflow.org/lite) and [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html).

### References

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Wagner, Caleb G (2023). "Stacked tensorial neural networks for reduced-order modeling of a parametric partial differential equation." https://arxiv.org/abs/2312.14979

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Wang, Maolin, Yu Pan, Xiangli Yang, Guangxi Li, and Zenglin Xu (2023). "Tensor networks meet neural networks: A survey." https://arxiv.org/abs/2302.09019
