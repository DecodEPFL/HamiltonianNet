import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from data.datasets import double_moons, Dataset
from integrators.integrators import Classification
from viewers.viewers import viewContour2D, viewTestData, plot_grad_x_iter


class FullyConnected(nn.Module):
    # a classic MLP net
    def __init__(self, n_layers, nf=4, random=True):
        super().__init__()

        self.n_layers = n_layers
        self.act = nn.Tanh()
        self.nf = nf

        if random:
            K = torch.randn(self.nf, self.nf, self.n_layers)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf, self.nf, self.n_layers)
            b = torch.ones(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K, True)
        self.b = nn.Parameter(b, True)

    def forward(self, Y0, ini=0, end=None):

        Y = Y0.transpose(1, 2)

        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            Y = self.act(F.linear(Y, self.K[:, :, j].transpose(0, 1), self.b[:, 0, j]))

        NNoutput = Y.transpose(1, 2)

        return NNoutput

    def getK(self):
        return self.K

    def getB(self):
        return self.b


def get_intermediate_states(model, Y0):
    Y0.requires_grad = True
    # Y_out N-element list containing the intermediates states. Size of each entry: n_samples * dim2 * dim1
    # Y_out[n] = torch.zeros([125, 4, 1]), with n=0,..,9
    Y_out = [Y0]
    i = 0
    for j in range(model.n_layers):
        Y = model.forward(Y_out[j], ini=j, end=j + 1)
        Y_out.append(Y)
        Y_out[j + 1].retain_grad()
    return Y_out

# # Select network parameters
n_layers = 8
# n_layers = 32
# nf = 4
nf = 6

# define data
data_size = 8000
train_data_size = 4000
test_data_size = data_size - train_data_size
data2d, labels, domain = double_moons(data_size, nf=nf)
partition = {'train': range(0, data_size, 2),
             'test': range(1, data_size, 2)}

# define network structure and optimizer
training_set = Dataset(partition['train'], data2d, labels)
training_generator = data.DataLoader(
    training_set, batch_size=125, shuffle=True)

# Define hyperparameters
learning_rate = 5e-2
alpha = 1e-4  # Weight decay for output layer
alphac = 1e-4  # Weight decay for FC layers

model = FullyConnected(n_layers, nf=nf)
modelc = Classification(nf=nf)

lossFunc = nn.BCEWithLogitsLoss()
lossFunc2 = nn.Identity()
optimizer_k = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_w = torch.optim.Adam(modelc.parameters(), lr=learning_rate)

max_iteration = 30
max_in_iteration = 10

batch_size = training_generator.batch_size
gradparam = np.zeros((int(train_data_size / 125) * max_iteration, model.nf, model.nf, n_layers))
grad = np.zeros((int(train_data_size / 125) * max_iteration, model.nf, model.nf, n_layers))
gammagrad = np.zeros((int(train_data_size / 125) * max_iteration, model.nf, model.nf, n_layers))
gradients_training = np.zeros([32*max_iteration, n_layers])
lossFunc2 = nn.Identity()
gradients_matrix = np.zeros([int(train_data_size / batch_size) * max_iteration, model.nf, model.nf, n_layers + 1])


# training network

for epoch in range(max_iteration):

    training_iterator = iter(training_generator)

    for i_k in range(int(data2d[partition['train']].size(0) / training_generator.batch_size)):

        local_samples, local_labels = next(training_iterator)
        local_samples.requires_grad = True

        modelc = Classification(nf=nf)
        optimizer_w = torch.optim.Adam(modelc.parameters(), lr=learning_rate)
        with torch.no_grad():
            YN = model(local_samples)

        for i_w in range(max_in_iteration):  # epoch of W

            optimizer_w.zero_grad()
            loss = lossFunc(modelc(YN), local_labels)
            loss = loss + alphac * 0.5 * (torch.norm(modelc.W) ** 2 + torch.norm(modelc.mu) ** 2)
            loss.backward()
            optimizer_w.step()

        local_samples.requires_grad = True
        matrix_aux = np.zeros([model.nf, model.nf, n_layers + 1])
        for k in range(model.nf):
            optimizer_k.zero_grad()
            Y_out = get_intermediate_states(model, local_samples)
            YN = Y_out[-1]
            loss = lossFunc2(YN[:, k, 0].sum())  # loss = lossFunc2(YN[:, k, 0].sum())
            loss.backward()
            for j in range(n_layers + 1):
                matrix_aux[:, k, j] = Y_out[j].grad[:, :, 0].numpy().sum(axis=0) / training_generator.batch_size
        gradients_matrix[epoch * int(train_data_size / batch_size) + i_k, :, :, :] = matrix_aux
        local_samples.requires_grad = False

        optimizer_k.zero_grad()
        YN = model(local_samples)
        loss = lossFunc(modelc(YN), local_labels)  # , h*alpha, model.K.data, model.b.data)
        loss_diff = 0
        for j in range(int(n_layers) - 1):
            loss_diff += alpha * (1 / 2 * torch.norm(model.K[:, :, j]) ** 2 +
                                  1 / 2 * torch.norm(model.b[:, :, j]) ** 2)
        loss = loss + loss_diff
        loss.backward()

        grad[epoch * int(train_data_size / 125) + i_k, :, :, :] = model.K.grad
        for param in model.parameters():
            if param.grad.shape[0] != 1:
                gradparam[epoch * int(train_data_size / 125) + i_k, :, :, :] = param.grad
                break

        li = list(optimizer_k.state)
        if not (len(li) == 0):
            for ii in range(2):
                optimizer_k.state[li[ii]]['step'] = epoch

        gammagrad[epoch * int(train_data_size / 125) + i_k, :, :, :] = (-1) * model.K.detach()
        optimizer_k.step()
        gammagrad[epoch * int(train_data_size / 125) + i_k, :, :, :] += model.K.detach().numpy()

modelc = Classification(nf=nf)
optimizer_w = torch.optim.Adam(modelc.parameters(), lr=learning_rate)

for epoch in range(max_iteration):

    training_iterator = iter(training_generator)

    for i_w in range(int(data2d[partition['train']].size(0) / training_generator.batch_size)):

        local_samples, local_labels = next(training_iterator)
        YN = model(local_samples)
        YN.retain_grad()
        optimizer_w.zero_grad()
        loss = lossFunc(modelc(YN), local_labels)
        loss = loss + alphac * 0.5 * (torch.norm(modelc.W) ** 2 + torch.norm(modelc.mu) ** 2)
        loss.backward()
        li = list(optimizer_w.state)
        if not (len(li) == 0):
            for ii in range(2):
                optimizer_w.state[li[ii]]['step'] = epoch
        optimizer_w.step()

# check after correct rate
with torch.no_grad():

    train_acc = (torch.ge(modelc(model(
        data2d[partition['train'], :, :])), 0) == labels[partition['train'], :]).sum().numpy() / train_data_size

    test_acc = (torch.ge(modelc(model(
        data2d[partition['test'], :, :])), 0) == labels[partition['test'], :]).sum().numpy() / test_data_size

# Print classification results
print('Fully connected net - #layers: %d - Train accuracy %.4f - Test accuracy %.4f' % (n_layers, train_acc, test_acc))

plt.figure(1)
viewContour2D(domain, model, modelc)
viewTestData(partition, data2d, labels)
plt.title('%d-layer MLP' % n_layers + ' - Test acc %.2f%%' % (test_acc*100))
plt.show()

# plot_grad_x_layer(gradients_matrix, colorscale=True)
plot_grad_x_iter(gradients_matrix, colorscale=True, log=True, one_line=False)
