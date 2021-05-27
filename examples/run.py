#!/usr/bin/env python
"""
Train a H-DNN on a 2D example.
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run.py                --dataset       [DATASET]               \
                             --nf            [NUMBER OF FEATURES]    \
                             --net_type      [MODEL NAME]            \
                             --n_layers      [NUMBER OF LAYERS]      \
                             --t_end         [FINAL TIME]            \
                             --gradient_info [GRADIENT INFORMATION]  \
                             --default       [DEFAULT]               \
                             --gpu           [GPU ID]
Flags:
  --dataset: 2D dataset to use. Available options are: swiss_roll, double_moons.
  --nf: Number of features, i.e. length of the input vector.
  --net_type: Network model to use. Available options are: MS1, MS2, MS3, H1_J1, H1_J2, H2.
  --n_layers: Number of layers for the chosen the model.
  --t_end: Time corresponding to the end on the forward propagation.
  --gradient_info: whether to calculate the gradient norms for each layer at each iteration.
  --default: whether to run with preselected parameters.
"""

from viewers.viewers import viewContour2D, viewTestData, viewPropagatedPoints, plot_grad_x_iter
from examples.train_2d_example import train_2d_example

import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='swiss_roll')
    parser.add_argument('--nf', type=int, default=4)
    parser.add_argument('--net_type', type=str, default='H1_J1')
    parser.add_argument('--n_layers', type=int, default=16)
    parser.add_argument('--t_end', type=float, default=1.2)
    parser.add_argument('--gradient_info', type=bool, default=False)
    args = parser.parse_args()

    # # Train the network
    model, model_c, train_acc, test_acc, data2d, label, partition, domain, gradients_matrix = \
        train_2d_example(args.dataset, args.net_type, args.nf, args.n_layers, args.t_end, args.gradient_info)

    # Print classification results
    print('Train accuracy: %.2f%% - Test accuracy: %.2f%%' % (train_acc*100, test_acc*100))

    # # Plot propagated points (in two dimensions)
    plt.figure(2)
    plt.title('t_end: %.2f - %d layers' % (args.t_end, args.n_layers) + ' - 2D projection of propagated points')
    viewPropagatedPoints(model, partition, data2d, label)

    # # Plot classification results
    plt.figure(1)
    plt.title('t_end: %.2f - %d layers' % (args.t_end, args.n_layers) + ' - Test acc %.2f%%' % (test_acc * 100))
    viewContour2D(domain, model, model_c)
    viewTestData(partition, data2d, label)

    # # Plot gradients
    if args.gradient_info:
        plot_grad_x_iter(gradients_matrix, colorscale=True, log=True)
    plt.show()

