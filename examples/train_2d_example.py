import torch
from torch.utils import data
from torch import nn
import numpy as np

from data.datasets import swiss_roll, double_circles, double_moons, Dataset
from regularization.regularization import regularization
from integrators.integrators import MS1, MS2, MS3, H1, H2, H2_sparse, Classification, get_intermediate_states


def train_2d_example(dataset='swiss_roll', net_type='H1', nf=4, n_layers=8, t_end=1, gradient_info=False, sparse=None,
                     seed=None):

    if dataset == 'swiss_roll':
        data_gen = swiss_roll
    elif dataset == 'double_circles':
        data_gen = double_circles
    elif dataset == 'double_moons':
        data_gen = double_moons
    else:
        raise ValueError("%s data set is not yet implemented" % dataset)

    if sparse is not None:
        if net_type != 'H2':
            raise ValueError("Sparse networks only implemented for H2-DNNs")

    out = 1

    # Set seed
    if seed is None:
        seed = np.random.randint(10000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # define data
    data_size = 8000
    train_data_size = 4000
    test_data_size = data_size - train_data_size
    if sparse is None:
        data2d, labels, domain = data_gen(data_size, nf=nf)
    else:
        data2d, labels, domain = data_gen(data_size, nf=nf, input_ch=[0, 12])
        if sparse != 'sparse' and sparse != 'full':
            raise ValueError("sparse variable can be either 'sparse' or 'full'")
        else:
            mask_k, mask_j = set_masks_sparse(sparse, nf)

    partition = {'train': range(0, data_size, 2),
                 'test': range(1, data_size, 2)}

    # # Select training parameters
    alpha = 5e-4
    alphac = 1e-4
    learning_rate = 0.5e-1
    max_iteration = 50
    max_in_iteration = 10

    # define network structure and optimizer
    batch_size = 125
    training_set = Dataset(partition['train'], data2d, labels)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    h = t_end / n_layers

    if sparse is not None:
        model = H2_sparse(n_layers, t_end, nf=16, mask_k=mask_k, mask_j=mask_j)
    elif net_type == 'MS1':
        model = MS1(n_layers, t_end, nf=nf)
    elif net_type == 'MS2':
        model = MS2(n_layers, t_end, nf=nf)
    elif net_type == 'MS3':
        model = MS3(n_layers, t_end, nf=nf)
    elif net_type == 'H1_J1':
        model = H1(n_layers, t_end, nf=nf, select_j='J1')
    elif net_type == 'H1_J2':
        model = H1(n_layers, t_end, nf=nf, select_j='J2')
    elif net_type == 'H2':
        model = H2(n_layers, t_end, nf=nf)
    else:
        raise ValueError("%s model is not yet implemented" % net_type)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer_k = torch.optim.Adam(model.parameters(), lr=learning_rate)  # , weight_decay=alpha/100)

    if gradient_info:
        loss_func2 = nn.Identity()
        gradients_matrix = np.zeros([int(train_data_size/batch_size) * max_iteration, model.nf, model.nf, n_layers + 1])
    else:
        gradients_matrix = None

    # check before correct rate
    print('%s example using a %d-layer %s-DNN with %d features. Alpha=%.1e. Final_time=%.2f'
            % (dataset, n_layers, net_type, nf, alpha, t_end))

    # Training network
    for epoch in range(max_iteration):

        training_iterator = iter(training_generator)

        for i_k in range(int(data2d[partition['train']].size(0) / training_generator.batch_size)):

            local_samples, local_labels = next(training_iterator)

            model_c = Classification(nf=nf)
            optimizer_w = torch.optim.Adam(model_c.parameters(), lr=learning_rate)
            with torch.no_grad():
                YN = model(local_samples)

            for i_w in range(max_in_iteration):  # Inner iteration

                optimizer_w.zero_grad()
                loss = loss_func(model_c(YN), local_labels)
                loss = loss + alphac * 0.5 * (torch.norm(model_c.W) ** 2 + torch.norm(model_c.mu) ** 2)
                loss.backward()
                optimizer_w.step()

            if gradient_info:
                local_samples.requires_grad = True
                matrix_aux = np.zeros([model.nf, model.nf, n_layers + 1])
                for k in range(model.nf):
                    optimizer_k.zero_grad()
                    Y_out = get_intermediate_states(model, local_samples)
                    YN = Y_out[-1]
                    loss = loss_func2(YN[:, k, 0].sum())
                    loss.backward()
                    for j in range(n_layers + 1):
                        matrix_aux[:, k, j] = Y_out[j].grad[:, :, 0].numpy().sum(axis=0) / training_generator.batch_size
                gradients_matrix[epoch * int(train_data_size / batch_size) + i_k, :, :, :] = matrix_aux
                local_samples.requires_grad = False

            optimizer_k.zero_grad()
            K = model.getK()
            b = model.getb()
            loss = loss_func(model_c(model(local_samples)), local_labels)
            loss += regularization(alpha, h, K, b)
            loss.backward()
            li = list(optimizer_k.state)
            if not (len(li) == 0):
                for ii in range(2):
                    optimizer_k.state[li[ii]]['step'] = epoch
            optimizer_k.step()

        if epoch % 10 == 0 and out > 0:
            model_c = Classification(nf=nf)
            optimizer_w = torch.optim.Adam(model_c.parameters(), lr=learning_rate)
            with torch.no_grad():
                YN = model(local_samples)
            for i_w in range(max_in_iteration):  # Inner iteration
                optimizer_w.zero_grad()
                loss = loss_func(model_c(YN), local_labels)
                loss = loss + alphac * 0.5 * (torch.norm(model_c.W) ** 2 + torch.norm(model_c.mu) ** 2)
                loss.backward()
                optimizer_w.step()
                acc = (torch.ge(model_c(model(local_samples)), 0) == local_labels).sum().numpy() / batch_size
            print('\tTrain Epoch: {:2d} - Loss: {:.6f} - Accuracy: {:.0f}%'.format(epoch, loss, acc*100))

    # Train classification layer with all the data

    model_c = Classification(nf=nf)
    optimizer_w = torch.optim.Adam(model_c.parameters(), lr=learning_rate)

    for epoch in range(max_iteration):

        training_iterator = iter(training_generator)

        for i_w in range(int(data2d[partition['train']].size(0) / training_generator.batch_size)):

            local_samples, local_labels = next(training_iterator)
            with torch.no_grad():
                YN = model(local_samples)

            optimizer_w.zero_grad()
            loss = loss_func(model_c(YN), local_labels)
            loss = loss + alphac * 0.5 * (torch.norm(model_c.W) ** 2 + torch.norm(model_c.mu) ** 2)
            loss.backward()
            optimizer_w.step()

    # Accuracy results

    with torch.no_grad():
        train_acc = (torch.ge(model_c(model(data2d[partition['train'], :, :])), 0) == labels[partition['train'], :]
                     ).sum().numpy() / train_data_size
        test_acc = (torch.ge(model_c(model(data2d[partition['test'], :, :])), 0) == labels[partition['test'], :]
                    ).sum().numpy() / test_data_size

    return model, model_c, train_acc, test_acc, data2d, labels, partition, domain, gradients_matrix


def set_masks_sparse(sparse, nf):
    if nf != 16:
        print("Proceeding with nf=16...")
    if sparse == 'sparse':
        mask_aux = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 1],
                                 [1, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 1, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 1, 1],
                                 [1, 0, 0, 0, 0, 0, 1, 1]], dtype=torch.float)
        mask_k = torch.cat((torch.cat((mask_aux, torch.zeros(8, 8)), dim=1),
                            torch.cat((torch.zeros(8, 8), mask_aux), dim=1)), dim=0)
        mask_aux = torch.eye(8)
        mask_j = torch.cat((torch.cat((torch.zeros(8, 8), mask_aux), dim=1),
                            torch.cat((mask_aux, torch.zeros(8, 8)), dim=1)), dim=0)
        mask_k = mask_k.type(torch.bool)
        mask_j = mask_j.type(torch.bool)
    elif sparse == 'full':
        mask_aux = torch.ones(8, 8)
        mask_k = torch.cat((torch.cat((mask_aux, torch.zeros(8, 8)), dim=1),
                            torch.cat((torch.zeros(8, 8), mask_aux), dim=1)), dim=0)
        mask_aux = torch.eye(8)
        mask_j = torch.cat((torch.cat((torch.zeros(8, 8), mask_aux), dim=1),
                            torch.cat((mask_aux, torch.zeros(8, 8)), dim=1)), dim=0)
        mask_k = mask_k.type(torch.bool)
        mask_j = mask_j.type(torch.bool)
    else:
        raise ValueError("%s is not a valid parameter" % sparse)
    return mask_k, mask_j
