import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from data.datasets import _data_extension


def viewContour2D(domain, model, model_c, input_ch=None):
    '''
    Coloured regions in domain represent the prediction of the DNN given the Hamiltonian net (model) and the output
    layer (modelc).
    input_ch indicates the indexes where the input data is plugged
    For 2d datasets.
    '''
    N = 200
    xa = np.linspace(domain[0], domain[1], N)
    ya = np.linspace(domain[2], domain[3], N)
    xv, yv = np.meshgrid(xa, ya)
    y = np.stack([xv.flatten(), yv.flatten()])
    y = np.expand_dims(y.T, axis=2)
    data2d = torch.from_numpy(y).float()
    nf = model.nf
    if nf != 2:
        data2d = _data_extension(data2d, nf, input_ch)
        # dataSize = data2d.shape[0]
        # if input_ch is not None:
        #     # input_ch is a list of two elements. The elements indicate where the data enters.
        #     idx_x = input_ch[0]
        #     idx_y = input_ch[1]
        # else:
        #     idx_x = 0
        #     idx_y = nf-1
        # data2d = torch.cat((torch.zeros(dataSize, idx_x-0, 1),
        #                   data2d[:, idx_x:idx_x+1, :],
        #                   torch.zeros(dataSize, idx_y-idx_x-1, 1),
        #                   data2d[:, idx_y:idx_y+1, :],
        #                   torch.zeros(dataSize, nf-1-idx_y, 1)), 1)
    
    with torch.no_grad():
        labels = torch.ge(model_c(model(data2d)), 0).int()
    plt.contourf(xa, ya, labels.view([N, N]), levels=[-0.5, 0.5, 1.5], colors=['#EAB5A0', '#99C4E2'])


def viewTestData(partition, data2d, labels, input_ch=None):
    if input_ch is not None:
        # input_ch is a list of two elements. The elements indicate where the data enters.
        idx_x = input_ch[0]
        idx_y = input_ch[1]
    else:
        nf = data2d.shape[1]
        idx_x = 0
        idx_y = nf-1
    # Plot test data for 2d datasets.
    testDataSize = len(partition['test'])
    mask0 = (labels[partition['test'], 0] == 0).view(testDataSize)
    plt.plot(data2d[partition['test'], idx_x, :].view(testDataSize).masked_select(mask0),
             data2d[partition['test'], idx_y, :].view(testDataSize).masked_select(mask0), 'r+',
             markersize=2)

    mask1 = (labels[partition['test'], 0] == 1).view(testDataSize)
    plt.plot(data2d[partition['test'], idx_x, :].view(testDataSize).masked_select(mask1),
             data2d[partition['test'], idx_y, :].view(testDataSize).masked_select(mask1), 'b+',
             markersize=2)


def viewPropagatedPoints(model, partition, data2d, labels, input_ch=None):
    if input_ch is not None:
        # input_ch is a list of two elements. The elements indicate where the data enters.
        idx_x = input_ch[0]
        idx_y = input_ch[1]
    else:
        nf = data2d.shape[1]
        idx_x = 0
        idx_y = nf-1
    test_data_size = labels[partition['test'], 0].size(0)
    mask0 = (labels[partition['test'], 0] == 0).view(test_data_size)
    YN = model(data2d[partition['test'], :, :]).detach()
    plt.plot(YN[:, idx_x, :].view(test_data_size).masked_select(mask0),
                YN[:, idx_y, :].view(test_data_size).masked_select(mask0), 'r+')

    mask1 = (labels[partition['test'], 0] == 1).view(test_data_size)
    plt.plot(YN[ :, idx_x, :].view(test_data_size).masked_select(mask1),
                YN[ :, idx_y, :].view(test_data_size).masked_select(mask1), 'b+')


def plot_grad_x_layer(gradients_matrix, colorscale=False, log=True):
    # Plot the gradient norms at each layer (different colors = different iterations)
    [tot_iters, nf, _, n_layers1] = gradients_matrix.shape
    n_layers = n_layers1 - 1

    if not colorscale:
        plt.figure()
        z = np.linspace(1, n_layers, n_layers)
        legend = []
        for ii in range(1, tot_iters, 100):
            plt.plot(z, np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:])
            legend.append("Iteration %s" % str(ii))
        for ii in range(1, tot_iters, 1):
            if np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:].sum() == 0:
                print("zero found at %s" % str(ii))
        plt.xlabel("Layers")
        plt.ylabel(r'$\left\|\frac{\partial y_N}{\partial y_\ell}\right\|$', fontsize=12)
        if log:
            plt.yscale('log')
        plt.legend(legend)
    else:
        z = np.linspace(1, n_layers, n_layers)
        fig, ax = plt.subplots()
        n = tot_iters
        # setup the normalization and the colormap
        normalize = mcolors.Normalize(vmin=1, vmax=n)
        colormap = cm.get_cmap('jet', n - 1)
        legend = ['Lower bound']
        ax.plot([1, n_layers], [1, 1], 'k--')
        plt.legend(legend)
        for ii in range(1, n, 1):
            ax.plot(z, np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:],
                    color=colormap(normalize(ii)),
                    linewidth=0.5)
        plt.xlabel("Layer $\ell$")
        plt.ylabel(r'$\left\|\frac{\partial y_N}{\partial y_\ell}\right\|$', fontsize=12)
        if log:
            plt.yscale('log')
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        cb = plt.colorbar(scalarmappaple)
        cb.set_label('# iteration')
        plt.tight_layout()


def plot_grad_x_iter(gradients_matrix, colorscale=False, log=True, one_line=True):
    # Plot the gradient norms at each iteration (different colors = different layers)
    [tot_iters, nf, _, n_layers1] = gradients_matrix.shape
    n_layers = n_layers1 - 1

    if not colorscale:
        plt.figure()
        z = np.linspace(1, tot_iters-1, tot_iters-1)
        legend = []
        for ii in range(1, n_layers):
            plt.plot(z, np.linalg.norm(gradients_matrix[:, :, :, ii], axis=(1, 2), ord=2)[1:])
            legend.append("Layer %s" % str(ii))
        plt.xlabel("Iteration")
        plt.ylabel(r'$\|\|\frac{\partial y_N}{\partial y_\ell}\|\|$', fontsize=12)
        if log:
            plt.yscale('log')
        plt.legend(legend)
        return legend
    else:
        x = np.linspace(0, tot_iters - 1, tot_iters)
        fig, ax = plt.subplots()
        n = n_layers
        # setup the normalization and the colormap
        normalize = mcolors.Normalize(vmin=1, vmax=n)
        colormap = cm.get_cmap('jet', n - 1)
        if one_line:
            legend = ['Lower bound']
            ax.plot([0, gradients_matrix.shape[0]], [1, 1], 'k--')
            plt.legend(legend)
        for ii in range(1, n_layers, 1):
            j = n_layers-ii
            ax.plot(x, np.linalg.norm(gradients_matrix[:, :, :, j], axis=(1, 2), ord=2), color=colormap(normalize(ii)),
                    linewidth=0.5)
        plt.xlabel("Iterations")
        plt.ylabel(r'$\|\|\frac{\partial y_N}{\partial y_{N-\ell}}\|\|$', fontsize=12)
        if log:
            plt.yscale('log')
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        cb = plt.colorbar(scalarmappaple)
        cb.set_label('Depth $\ell$')
        plt.tight_layout()
