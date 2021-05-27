#!/usr/bin/env python
"""
Train an 8-node H2-DNN on a 2D example.
Each node has dimension 2. The interconnection is given in Fig. 2 [1].
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run_distributed.py    --dataset       [DATASET]               \
                             --n_layers      [NUMBER OF LAYERS]      \
                             --sparsity      [SPARSITY]              \
                             --gradient_info [GRADIENT INFORMATION]  \
                             --gpu           [GPU ID]
Flags:
  --dataset: 2D dataset to use. Available options are: swiss_roll, double_circles.
  --n_layers: Number of layers for the H2-DNN.
  --sparsity: Select 'sparse' for nodes connected as Fig. 2 [1]. Select 'full' to connect all nodes at each layer.
  --gradient_info: whether to calculate the gradient norms for each layer at each iteration.
"""

from viewers.viewers import viewContour2D, viewTestData, plot_grad_x_iter
from examples.train_2d_example import train_2d_example

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='swiss_roll')
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--sparsity', type=str, default='sparse')
    parser.add_argument('--gradient_info', type=bool, default=False)
    args = parser.parse_args()

    # Setting network parameters
    nf = 16  # number of features
    net_type = "H2"  # network type
    input_ch = [0, 12]  # input features entering in p^[1] and q^[5]
    t_end = 0.5 * args.n_layers

    # # Train the network
    model, model_c, train_acc, test_acc, data2d, label, partition, domain, gradients_matrix = \
        train_2d_example(args.dataset, net_type, nf, args.n_layers, t_end, args.gradient_info, sparse=args.sparsity)

    # Print classification results
    print('Train accuracy %.4f - Test accuracy %.4f' % (train_acc, test_acc))

    # # Plot classification results
    plt.figure(1)
    plt.title('t_end: %.2f - %d layers' % (t_end, args.n_layers) + ' - Test acc %.2f%%' % (test_acc*100))
    viewContour2D(domain, model, model_c, input_ch)
    viewTestData(partition, data2d, label, input_ch)

    # # Plot gradients
    if args.gradient_info:
        plot_grad_x_iter(gradients_matrix, colorscale=True, log=True)
    plt.show()
