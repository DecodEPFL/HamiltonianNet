import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy.sparse import coo_matrix


class MS1(nn.Module):
    # Hamiltonian neural network, as presented in [3].
    # MS_1-DNN
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   K(t) = [ 0 \tilde{K}(t) ; -\tilde{K}^T(t) 0 ],
    #   J(y,t) K(t) = I.
    # Discretization method: Verlet

    def __init__(self, n_layers, t_end, nf, random=True):
        super().__init__()

        self.n_layers = n_layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = (nf//2)*2

        if random:
            K = torch.randn(self.nf // 2, self.nf // 2, self.n_layers)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf // 2, self.nf // 2, self.n_layers)
            b = torch.zeros(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K, True)
        self.b = nn.Parameter(b, True)

    def getK(self):
        K = torch.zeros(self.nf, self.nf, self.n_layers)
        K[:self.nf//2, self.nf//2:, :] = self.K
        K[self.nf//2:, :self.nf//2, :] = (-1) * self.K.transpose(0, 1)
        return K

    def getb(self):
        # return self.b * self.B
        return self.b

    def forward(self, Y0, ini=0, end=None):

        # The size of Y0 is (sampleNumber, nf, 1)
        dim = len(Y0.shape)
        Y_aux = Y0.transpose(1, dim-1)
        Y, Z = torch.split(Y_aux, self.nf//2, dim=dim-1)
        # Y = Y0[:, :self.nf//2, :].transpose(1, 2)
        # Z = Y0[:, self.nf//2:, :].transpose(1, 2)  # torch.zeros(sampleNumbers, 1, self.nf)

        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            Z = Z + self.h * self.act(
                F.linear(Y, self.K[:, :, j], self.b[self.nf//2:self.nf, 0, j]))
            Y = Y + self.h * self.act(
                F.linear(Z, -self.K[:, :, j].transpose(0, 1), self.b[0:self.nf//2, 0, j]))

        NNoutput = torch.cat((Y, Z), dim-1).transpose(1, dim-1)

        return NNoutput


class MS2(nn.Module):
    # Anti-symmetric Hamiltonian neural network, as presented in [3,4].
    # MS_2-DNN
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   K(t) = -K^T(t),
    #   J(y,t) K(t) = I.
    # Discretization method: Forward Euler

    def __init__(self, n_layers, t_end, nf, random=True, nt=None):
        super().__init__()

        self.n_layers = n_layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = nf

        I = np.arange(0, self.nf ** 2)
        A = np.transpose(I.reshape(self.nf, self.nf), (1, 0)).reshape(1, self.nf**2).squeeze(0)
        i = np.concatenate([[np.concatenate([I, I], axis=0)], [np.concatenate([I, A], axis=0)]], axis=0)
        v = np.concatenate([np.ones(self.nf**2), -1*np.ones(self.nf**2)])
        Q = coo_matrix((v, i), shape=(self.nf**2, self.nf**2)).tocsc()
        Q.eliminate_zeros()
        Q = Q[:, Q.getnnz(0)>0].tocoo()
        values = torch.Tensor(Q.data)
        indices = torch.stack([torch.tensor(Q.row, dtype=torch.long), torch.tensor(Q.col, dtype=torch.long)])
        shape = Q.shape
        self.Q = torch.sparse.FloatTensor(indices, values, shape)

        if random:
            K = torch.randn(self.nf ** 2 - self.nf, 1, self.n_layers)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf ** 2 - self.nf, 1, self.n_layers)
            b = torch.zeros(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K, True)
        self.b = nn.Parameter(b, True)

    def getK(self):
        theta = F.linear(self.K.transpose(2, 0), self.Q.to_dense()).transpose(2, 0)
        theta = theta.view([self.nf, self.nf, self.n_layers])
        return theta

    def getb(self):
        return self.b

    def forward(self, Y0, ini=0, end=None):

        # the size of Y0 is (sampleNumber, nf, 1)
        dim = len(Y0.shape)
        Y = Y0.transpose(1, dim-1)

        K = self.getK()
        b = self.getb()

        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            Y = Y + self.h * self.act(
                F.linear(Y, K[:, :, j].transpose(0, 1), b[:, 0, j]))

        NNoutput = Y.transpose(1, dim-1)

        return NNoutput


class MS3(nn.Module):
    # Two-layer Hamiltonian neural network, as presented in [4].
    # MS_3-DNN
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   K(t) = [ 0 K_1(t) ; K_2(t) 0 ],
    #   J(y,t) = [ 0 I ; -I 0 ].
    # Discretization method: Verlet

    def __init__(self, n_layers, t_end, nf, random=True, nt=None):
        super().__init__()

        self.n_layers = n_layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = nf
        if self.nf%2 == 1:
            print("Possible errors due to extended dimension being odd (it should be even)")

        if random:
            K1 = torch.randn(self.nf // 2, self.nf // 2, self.n_layers)
            K2 = torch.randn(self.nf // 2, self.nf // 2, self.n_layers)
            b1 = torch.randn(self.nf // 2, 1, self.n_layers)
            b2 = torch.randn(self.nf // 2, 1, self.n_layers)
        else:
            K1 = torch.ones(self.nf // 2, self.nf // 2, self.n_layers)
            K2 = torch.ones(self.nf // 2, self.nf // 2, self.n_layers)
            b1 = torch.zeros(self.nf // 2, 1, self.n_layers)
            b2 = torch.zeros(self.nf // 2, 1, self.n_layers)

        self.K1 = nn.Parameter(K1, True)
        self.K2 = nn.Parameter(K2, True)
        self.b1 = nn.Parameter(b1, True)
        self.b2 = nn.Parameter(b2, True)

    def getK(self):
        K = torch.cat((torch.cat((torch.zeros(self.K1.size()), self.K1), 1),
                       torch.cat((self.K2, torch.zeros(self.K1.size())), 1)), 0)
        return K

    def getb(self):
        b = torch.cat((self.b1, self.b2), 0)
        return b

    def forward(self, Y0, ini=0, end=None):

        dim = len(Y0.shape)
        Y_aux = Y0.transpose(1, dim-1)
        Y, Z = torch.split(Y_aux, self.nf//2, dim=dim-1)

        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            Z = Z - self.h * F.linear(self.act(
                F.linear(Y, self.K1[:, :, j].transpose(0, 1), self.b2[:, 0, j])), self.K1[:, :, j])
            Y = Y + self.h * F.linear(self.act(
                F.linear(Z, self.K2[:, :, j].transpose(0, 1), self.b1[:, 0, j])), self.K2[:, :, j])

        NNoutput = torch.cat((Y, Z), dim-1).transpose(1, dim-1)

        return NNoutput


class H1(nn.Module):
    # Hamiltonian neural network, as presented in [1,2].
    # H_1-DNN and H_2-DNN
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   J(y,t) = J_1 = [ 0 I ; -I 0 ]  or  J(y,t) = J_2 = [ 0 1 .. 1 ; -1 0 .. 1 ; .. ; -1 -1 .. 0 ].
    # Discretization method: Forward Euler
    def __init__(self, n_layers, t_end, nf, random=True, select_j='J1'):
        super().__init__()

        self.n_layers = n_layers  # nt: number of layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = nf

        if random:
            K = torch.randn(self.nf, self.nf, self.n_layers)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf, self.nf, self.n_layers)
            b = torch.zeros(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K, True)
        self.b = nn.Parameter(b, True)

        if select_j == 'J1':
            j_identity = torch.eye(self.nf//2)
            j_zeros = torch.zeros(self.nf//2, self.nf//2)
            self.J = torch.cat((torch.cat((j_zeros, j_identity), 0), torch.cat((- j_identity, j_zeros), 0)), 1)
        else:
            j_aux = np.hstack((np.zeros(1), np.ones(self.nf-1)))
            J = j_aux
            for j in range(self.nf-1):
                j_aux = np.hstack((-1 * np.ones(1), j_aux[:-1]))
                J = np.vstack((J, j_aux))
            self.J = torch.tensor(J, dtype=torch.float32)

    def getK(self):
        return self.K

    def getb(self):
        return self.b

    def getJ(self):
        return self.J

    def forward(self, Y0, ini=0, end=None):

        dim = len(Y0.shape)
        Y = Y0.transpose(1, dim-1)

        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            Y = Y + self.h * F.linear(self.act(F.linear(
                Y, self.K[:, :, j].transpose(0, 1), self.b[:, 0, j])), torch.matmul(self.J, self.K[:, :, j]))

        NNoutput = Y.transpose(1, dim-1)

        return NNoutput


class H2(nn.Module):
    # Hamiltonian neural network, as presented in [1].
    # H_2-DNN
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   J(y,t) = J_1 = [ 0 I ; -I 0 ]
    # Discretization method: Symplectic Euler
    def __init__(self, n_layers, t_end, nf=4, random=True):
        super().__init__()

        self.n_layers = n_layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        if nf%2 == 0:
            self.nf = nf
        else:
            self.nf = nf+1
            print("Number of features need to be and even number -- setting nf = %i" % self.nf)
        self.mask = torch.cat((torch.cat((torch.ones((nf//2, nf//2), dtype=torch.bool),
                                          torch.zeros((nf//2, nf//2), dtype=torch.bool)), dim=1),
                               torch.cat((torch.zeros((nf//2, nf//2), dtype=torch.bool),
                                          torch.ones((nf//2, nf//2), dtype=torch.bool)), dim=1)), dim=0)
        if random:
            K = torch.randn(self.nf, self.nf, self.n_layers) * self.mask.unsqueeze(2).repeat(1, 1, self.n_layers)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf, self.nf, self.n_layers) * self.mask.unsqueeze(2).repeat(1, 1, self.n_layers)
            b = torch.ones(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K * self.mask.unsqueeze(2).repeat(1, 1, self.n_layers), True)
        self.b = nn.Parameter(b, True)

    def getK(self):
        return self.K

    def getb(self):
        return self.b

    def getJ(self):
        return self.J

    def forward(self, Y0, ini=0, end=None):

        # the size of Y0 is (sampleNumber, nf, 1)

        Y = Y0.transpose(1, 2).clone()
        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            my_mask = self.mask
            Y[:, :, self.nf//2:] = Y[:, :, self.nf//2:] + self.h * F.linear(
                self.act(F.linear(Y[:, :, :self.nf//2], (self.K[:, :, j].transpose(0, 1) * my_mask)[:self.nf//2, :self.nf//2],
                                  self.b[:self.nf//2, 0, j])),
                (self.K[:, :, j] * my_mask)[:self.nf//2, :self.nf//2])
            Y[:, :, :self.nf//2] = Y[:, :, :self.nf//2] - self.h * F.linear(
                self.act(F.linear(Y[:, :, self.nf//2:], (self.K[:, :, j].transpose(0, 1) * my_mask)[self.nf//2:, self.nf//2:],
                                  self.b[self.nf//2:, 0, j])),
                (self.K[:, :, j] * my_mask)[self.nf//2:, self.nf//2:])
        NNoutput = Y.transpose(1, 2)

        return NNoutput


class H2_sparse(nn.Module):
    # Hamiltonian neural network [1], in a distributed fashion way, considering subsystems.
    # H_2-DNN
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    # Discretization method: Symplectic Euler
    def __init__(self, n_layers, t_end, nf=4, random=True, mask_k=None, mask_j=None):
        super().__init__()

        self.n_layers = n_layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        self.nf = nf
        if nf % 2 == 1:
            print("nf need to be even")
            return
        if mask_k is not None and mask_j is not None:
            mask = True
            mask = mask and mask_k.shape[0] == nf
            mask = mask and mask_k.shape[1] == nf
            mask = mask and mask_j.shape[0] == nf
            mask = mask and mask_j.shape[1] == nf
            # Check also that they have at least sparsity I_M
        if not mask or mask_k is None or mask_j is None:
            print("masks are not well defined")
            return
        self.mask_k = mask_k
        self.mask_j = mask_j

        if random:
            K = torch.randn(self.nf, self.nf, self.n_layers)
            b = torch.randn(self.nf, 1, self.n_layers)
        else:
            K = torch.ones(self.nf, self.nf, self.n_layers)
            b = torch.ones(self.nf, 1, self.n_layers)

        self.K = nn.Parameter(K * self.mask_k.unsqueeze(2).repeat(1, 1, self.n_layers), True)
        self.b = nn.Parameter(b, True)

        J = torch.zeros(self.nf, self.nf)
        J[nf//2:, :nf//2] = -torch.ones(nf//2, nf//2)
        J[:nf//2, nf//2:] = torch.ones(nf//2, nf//2)
        self.J = J * mask_j

    def getK(self):
        return self.K

    def getb(self):
        return self.b

    def getJ(self):
        return self.J

    def forward(self, Y0, ini=0, end=None):

        # the size of Y0 is (sampleNumber, nf, 1)

        Y = Y0.transpose(1, 2).clone()
        if end is None:
            end = self.n_layers

        for j in range(ini, end):
            Y[:, :, self.nf//2:] = Y[:, :, self.nf//2:] + self.h * F.linear(
                self.act(F.linear(Y[:, :, :self.nf//2],
                                  (self.K[:, :, j].transpose(0, 1) * self.mask_k)[:self.nf//2, :self.nf//2],
                                  self.b[:self.nf//2, 0, j])),
                torch.matmul(self.J, self.K[:, :, j] * self.mask_k)[self.nf//2:, :self.nf//2])
            Y[:, :, :self.nf//2] = Y[:, :, :self.nf//2] + self.h * F.linear(
                self.act(F.linear(Y[:, :, self.nf//2:],
                                  (self.K[:, :, j].transpose(0, 1) * self.mask_k)[self.nf//2:, self.nf//2:],
                                  self.b[self.nf//2:, 0, j])),
                torch.matmul(self.J, self.K[:, :, j] * self.mask_k)[:self.nf//2, self.nf//2:])
        NNoutput = Y.transpose(1, 2)

        return NNoutput


def get_intermediate_states(model, Y0):
    Y0.requires_grad = True
    # Y_out N-element list containing the intermediates states. Size of each entry: n_samples * dim2 * dim1
    # Y_out[n] = torch.zeros([batch_size, nf, 1]), with n=0,1,..,
    Y_out = [Y0]
    i = 0
    for j in range(model.n_layers):
        Y = model.forward(Y_out[j], ini=j, end=j + 1)
        Y_out.append(Y)
        Y_out[j + 1].retain_grad()
    return Y_out


class Classification(nn.Module):
    def __init__(self, nf=2, nout=1):
        super().__init__()
        self.nout = nout
        self.W = nn.Parameter(torch.zeros(self.nout, nf), True)
        self.mu = nn.Parameter(torch.zeros(1, self.nout), True)

    def forward(self, Y0):
        Y = Y0.transpose(1, 2)
        NNoutput = F.linear(Y, self.W, self.mu).squeeze(1)
        return NNoutput
