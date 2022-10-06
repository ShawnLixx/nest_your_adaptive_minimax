# Nest Your Adaptive Algorithm for Parameter-Agnostic Nonconvex Minimax Optimization

Code for NeurIPS 2022 paper ["Nest Your Adaptive Algorithm for Parameter-Agnostic
Nonconvex Minimax Optimization"](https://arxiv.org/abs/2206.00743).
Junchi Yang<sup>\*</sup>, Xiang Li<sup>\*</sup>, Niao He.

We adapted code from https://github.com/Louis-udm/Reproducing-certifiable-distributional-robustness

Please install the following packages of Python before running the code:
- pytorch
- matplotlib
- numpy
- tensorflow
- tensorboard

To get the results for test functions, please run
````
cd test_func
bash run.sh
````
The figures will be saved in current path.

To get the results for distributional robustness optimization, please run
````
cd distributional_robust
bash run.sh  # for synthetic dataset
bash run_mnist.sh  # for MNIST dataset
````
We use tensorboard to record the experimental data. To see the results, please use
````
tensorboard --logdir logs
````
