import random
from collections import defaultdict
from typing import OrderedDict
from functools import partial
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=8, help='random seed')
parser.add_argument('--n_iter', type=int, default=3000, help='number of gradient calls')
parser.add_argument('--lr_y', type=float, default=0.01, help='learning rate of y')
parser.add_argument('--init_x', type=float, default=None, help='init value of x')
parser.add_argument('--init_y', type=float, default=None, help='init value of y')
parser.add_argument('--r', type=float, default=1, help='ratio of stepsize y and stepsize x')
parser.add_argument('--L', type=float, default=1, help='parameter for the test function')
parser.add_argument('--grad_noise_y', type=float, default=0, help='gradient noise variance')
parser.add_argument('--grad_noise_x', type=float, default=0, help='gradient noise variance')
parser.add_argument('--func', type=str, default='quadratic', help='function name')

# plot related
parser.add_argument('--ylim_top', type=float, default=None, help='y_lim top')
parser.add_argument('--ylim_bottom', type=float, default=None, help='y_lim bottom')
parser.add_argument('--plot_sample', type=int, default=None, help='number of points to plot')
parser.add_argument('--plot_smooth', type=float, default=0, help='smooth curve')
parser.add_argument('--plot_size', type=int, default=3, help='curve size')
parser.add_argument('--alpha', type=float, default=0.9, help='transparency')
parser.add_argument('--save', action='store_true', help='save figure instead of showing')
parser.set_defaults(save=False)
args = parser.parse_args()

# Set precision to 64
torch.set_default_dtype(torch.float64)

# function to smooth the plot
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Computer Modern Roman"
plt.rcParams["font.size"] = 13
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["lines.linewidth"] = args.plot_size

# Different functions
functions = OrderedDict()

L = args.L
functions["quadratic"] = {
        "func":
            lambda x, y: -1/2 * (y ** 2) + L * x * y - (L ** 2 / 2) * (x ** 2),
        }
functions["McCormick"] = {
        "func":
            lambda x, y: torch.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1 + y[0] * x[0] + y[1] * x[1] \
                - 0.5 * (y[0] ** 2 + y[1] ** 2),
        }

optimizers = OrderedDict()
if args.func == 'McCormick':
    # Adam is extremely unstable on McCormick functions, so we need a large eps
    eps = 0.8
    optimizers["Adam"] = partial(torch.optim.Adam, eps=eps)
    optimizers["AMSGrad"] = partial(torch.optim.Adam, amsgrad=True, eps=eps)
else:
    eps = 1e-8
    optimizers["Adam"] = partial(torch.optim.Adam, eps=eps)
    optimizers["AMSGrad"] = partial(torch.optim.Adam, amsgrad=True, eps=eps)

optimizers["AdaGrad"] = torch.optim.Adagrad
optimizers["GDA"] = torch.optim.SGD

# Color list for plot
color_list = ["#ff1f5b", "#00cd6c", "#009ade", "#af58ba"]
colors = {}
for i, (optim_name, c) in enumerate(zip(optimizers.copy().keys(), color_list)):
    optimizers[f"NeAda-{optim_name}"] = optimizers[optim_name]
    colors[optim_name] = c

# Reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


print(f"Function: {args.func}")
fun = functions[args.func]["func"]

if args.func == "McCormick":
    dim = 2
else:
    dim = 1

# learning rate
ratio = args.r
lr_y = args.lr_y
lr_x = lr_y / ratio

if args.init_x is None:
    init_x = torch.randn(dim)
else:
    init_x = torch.Tensor([args.init_x])
if args.init_y is None:
    init_y = torch.randn(dim)
else:
    init_y = torch.Tensor([args.init_y])
print(f"init x: {init_x}, init y: {init_y}")

# training loop
n_iter = args.n_iter
for index, optim_key in enumerate(optimizers.keys()):
    print(f"optimizer: {optim_key}")
    # clone to make sure every optimizer has the same initialization
    x = torch.nn.parameter.Parameter(init_x.clone())
    y = torch.nn.parameter.Parameter(init_y.clone())

    if "NeAda" in optim_key:
        optim_name = optim_key[6:]
    else:
        optim_name = optim_key

    optim = optimizers[optim_name]
    optim_x = optim([x], lr=lr_x)
    optim_y = optim([y], lr=lr_y)

    results = defaultdict(list)

    grad_calls = []  # save the steps where we record the gradient norm
    i = 0
    outer_loop_count = 0
    while i < n_iter:
        if "NeAda" in optim_key:
            # inner loop
            required_err = 1 / (outer_loop_count + 1)
            inner_step = 0
            inner_err = required_err + 1  # ensure execute at least one step 
            # stop when number of steps >= stop_constant * outer_loop_count
            # any stop_constant > 0 satisfies achieves the best rate in theory
            stop_constant = 1 
            if args.func == 'quadratic':
                # it is easier in the quadratic case, we stop it earlier
                stop_constant = 0.1
            while inner_err > required_err and i < n_iter and inner_step < stop_constant * outer_loop_count:
                inner_step += 1
                # update y
                optim_x.zero_grad()
                optim_y.zero_grad()
                l = -fun(x, y)
                l.backward()
                # stocastic gradient
                y.grad += torch.randn(dim) * args.grad_noise_y
                optim_y.step()

                inner_err = torch.norm(y.grad) ** 2
                i += 1

            if i == n_iter:
                break

            # outer loop
            # update x
            optim_x.zero_grad()
            optim_y.zero_grad()
            l = fun(x, y)
            l.backward()

            # record the deterministic gradient norm
            i += 1
            grad_calls.append(i)
            results['x_grad'].append(torch.norm(x.grad).item())
            outer_loop_count += 1
            # stocastic gradient
            x.grad += torch.randn(dim) * args.grad_noise_x
            optim_x.step()

        else:  # other optimizers
            optim_x.zero_grad()
            optim_y.zero_grad()
            l = fun(x, y)
            l.backward()
            # record gradient first, since we need deterministic gradients norm
            i += 2
            grad_calls.append(i)
            results['x_grad'].append(torch.norm(x.grad).item())
            # stocastic gradient
            y.grad = -y.grad + args.grad_noise_y * torch.randn(dim)
            x.grad += args.grad_noise_x * torch.randn(dim)
            optim_x.step()
            optim_y.step()

    results['x_grad'] = np.array(results['x_grad'])
    grad_calls = np.array(grad_calls)
    # Sample less points to make the plot more clear
    if args.plot_sample is not None and len(results['x_grad']) > args.plot_sample:
        # uniformlyl sample
        max_iter = grad_calls[-1]
        gap = max_iter // args.plot_sample
        current = 0
        new_result = []
        new_steps = []
        for i, step in enumerate(grad_calls):
            if step >= current * gap:
                new_result.append(results['x_grad'][i])
                new_steps.append(step)
                current += 1
        results['x_grad'] = np.array(new_result)
        grad_calls = np.array(new_steps)

    # Smooth the curve
    if args.plot_smooth > 0:
        results['x_grad'] = smooth(results['x_grad'], args.plot_smooth)

    label_name = optim_key

    if 'NeAda' in optim_key:
        ls = '-'
    else:
        if args.func == 'McCormick':
            ls = (index, (1, 1))
        else:
            ls = (index*1.5, (3, 5))
        if optim_name == 'GDA':
            markevery = (1, 15)

    plt.plot(grad_calls, results['x_grad'], label=label_name, linestyle=ls,
            color=colors[optim_name], alpha=args.alpha)


plt.legend()
plt.yscale('log')
plt.xlabel("\\#gradient calls")
plt.ylabel("$\\|\\nabla_x f(x, y)\\|$")

if args.ylim_top is not None:
    plt.ylim(top=args.ylim_top)
if args.ylim_bottom is not None:
    plt.ylim(bottom=args.ylim_bottom)

# save figure
if args.save:
    filename = ""
    if args.func == 'quadratic':
        filename += f"L{args.L}"
    else:
        filename += f"{args.func}"
    filename += f"_r_{args.r}_lry_{args.lr_y}"
    if args.grad_noise_x != 0:
        filename += f"_noisex_{args.grad_noise_x}"
    if args.grad_noise_y != 0:
        filename += f"_noisey_{args.grad_noise_y}"
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
else:
    plt.show()
