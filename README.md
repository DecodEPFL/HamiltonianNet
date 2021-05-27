# Hamiltonian Deep Neural Networks

PyTorch implementation of Hamiltonian deep neural networks as presented in [1].

## Installation

```bash
git clone https://github.com/ClaraGalimberti/HamiltonianNet.git

cd hamiltonianNet

python setup.py install
```

## Basic usage

2D classification examples:
```bash
./examples/run.py --dataset [DATASET] --model [MODEL]
```
where available values for `DATASET` are `swiss_roll` and `double_moons`.

Distributed training on 2D classification examples:
```bash
./examples/run_distributed.py --dataset [DATASET]
```
where available values for `DATASET` are `swiss_roll` and `double_circles`.

Classification over MNIST dataset:
```bash
./examples/run_MNIST.py --model [MODEL]
```
where available values for `MODEL` are `MS1` and `H1`.


To reproduce the counterexample of Appendix III:
```bash
./examples/gradient_analysis/perturbation_analysis.py
```


<!--
## H-DNNs

We extend the network structures proposed by [3] and [4] and we include them in a unified and more general model that we call Hamiltonian deep neural networks (H-DNNs). H-DNNs are obrained after the discretization of an ordinary differential equation (ODE) that represents a time-varying Hamiltonian system.

From a system theory perspective, these systems are relevant since they offer a modelling framework for systems based on energy functions. From this approach, a Hamiltonian function represents the total stored energy of the system.

The time varying dynamics of a Hamiltonian system is given by

```math
\dot{y}(t) = J(y,t) \frac{\partial H(y,t)}{\partial y}
```

where $`y(t) \in \mathbb{R}^N`$ represents the state, $`H(y,t): \mathbb{R}^N \times \mathbb{R} \rightarrow \mathbb{R}`$ is the Hamiltonian function and the $`N \times N`$ matrix, called interconnection matrix, satisfies $`J(y,t) = - J^T(y,t)`$ $`\forall t`$.


In this work, we consider Hamiltonian systems based on the energy function

```math
H(y,t) = [\log \cosh (K(t) y(t) + b(t))]^T \boldsymbol{1}
```

where $`\log(\cdot)`$ and $`\cosh(\cdot)`$ are applied element wise, and $`K(t)`$ and $`b(t)`$ are the trainable parameters.

Then, the ODE of Eq. 1 is given by

```math
\dot{y}(t) = J(y,t) K^T(t) \tanh (K(t) y(t) + b(t))
```

After selecting a proper discretization method, we can define different DNN structures based on the ODE of Eq.3.

We implement some DNNs using both forward Euler and semi-implicit Euler discretization, and imposing restictions over $`J`$ and $`K`$. Details can be found in Table 1.



|  Name                         | Restictions | 
| :---:                         | :---:          | 
| MS<sub>1</sub>-DNN | $`K(t) = \begin{bmatrix} 0 & K_0(t) \\ -K_0^T(t) & 0 \\ \end{bmatrix}`$   and   $`J(y,t)K(t) = I`$ | 
| MS<sub>2</sub>-DNN | $`K(t) = -K^T(t)`$   and   $`J(y,t)K(t) = I`$ | 
| MS<sub>3</sub>-DNN | $`K(t) = \begin{bmatrix} 0 & K_1(t) \\ K_2(t) & 0 \\ \end{bmatrix}`$   and   $`J(y,t) = \begin{bmatrix} 0 & I \\ -I & 0 \\ \end{bmatrix}`$ | 
| H<sub>1-J1</sub>-DNN | $`J(y,t) = \begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix} \\`$ | 
| H<sub>1-J2</sub>-DNN | $`J(y,t) = \begin{bmatrix} 0 & 1 & \dots & 1 \\ -1 & 0 & \dots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ -1 & -1 & \dots & 0 \\ \end{bmatrix}`$ | 
| H<sub>2</sub>-DNN | $`J(y,t) = \begin{bmatrix} 0 & -X^T \\ X & 0 \end{bmatrix} \\`$  and $`K(t) = \begin{bmatrix} K_p(t) & 0 \\ 0 & K_q(t) \\ \end{bmatrix}`$| 


_Remark: MS<sub>i</sub> networks were introduced in [3,4]. We have adapted to match our framework._
-->

## References
[1]
Clara L. Galimberti, Luca Furieri, Liang Xu and Giancarlo Ferrrari Trecate.
"Hamiltonian Deep Neural Networks Guaranteeing Non-vanishing Gradients by Design,"
ArXiv ???:????, 2021.

[2]
Clara L. Galimberti, Liang Xu and Giancarlo Ferrrari Trecate.
"A unified framework for Hamiltonian deep neural networks,"
arXiv:2104.13166, 2021.

[3] 
Eldad Haber and Lars Ruthotto.
"Stable architectures for deep neural networks,"
Inverse Problems, vol. 34, p. 014004, Dec 2017.

[4] 
Bo Chang, Lili Meng, Eldad Haber, Lars Ruthotto, David Begert and Elliot Holtham.
"Reversible architectures for arbitrarily deep residual neural networks,"
AAAI Conference on Artificial Intelligence, 2018.
