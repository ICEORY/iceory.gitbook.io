# BNN (Binarized Neural Networks)

In this paper, the authors proposed a method to train Binarized Neural Networks (BNNs), a network with binary weights and activations. The proposed BNNs drastically reduce the memory consumption (size and number of accesses) and have higher power-efficiency as it replaces most arithmetic operations with bit-wise operations. The code implemented in [Theano](https://github.com/MatthieuCourbariaux/BinaryNet) and [Torch](https://github.com/itayhubara/BinaryNet) is available on GitHub.

## Proposed Method

### Binarization Strategies

Constrain both weights and activation to either +1 or -1 has higher efficiency in hardware. The authors discussed two binarization functions including deterministic and stochastic. Formulation of deterministic binarization function is:
$$
x^b = sign(x)=
\begin{cases}
+1 & if ~x\ge 0 \\
-1 & otherwise,
\end{cases} 
\tag{1}
$$


The stochastic binarization function is:

$$
x^b = 
\begin{cases}
+1, & \mathrm{with~probability}~p=\sigma(x) \\
-1, & \mathrm{with~probability}~1-p,
\end{cases} \tag{2}
$$

where $$\sigma$$ is the "*hard sigmoid*" function:

$$
\sigma(x) = clip(\frac{x+1}{2},0,1) = \max(0,\min(1,\frac{x+1}{2})) \tag{3}
$$

The authors suggested that the stochastic binarization is harder to implement as it requires the hardware to generate random bits, though it is more appealing than the deterministic binarization, so they preferred to use the deterministic binarization function in their experiments.

### Gradient

Real-valued gradients are computed and accumulated in real-valued variables in this paper, as high precision is required for SGD. Previous work shows that using "straight-through estimator" can help the network training faster, the authors used straight-through estimator of $$\frac{\partial C}{\partial r}$$ simplified as:
$$
g_r = g_q1_{|r|\le1} \tag{4}
$$
which cancels the gradient when $$r$$ is too large. The derivation $$1_{|r|\le1}$$ can also be seen as propagating the gradient through *hard tanh*:
$$
\mathrm{Htanh}(x)=clip(x,-1,1)=\max(-1,\min(1,x)) \tag{5}
$$
The real-valued weights $$w^r$$ first projected to $$[-1,+1]$$ and then quantized to binarized weights $$w^b$$ using $$w^b=sign(w^r)$$. 

### Shift-based Batch Normalization

The authors proposed a shift-based batch normalization (SBN) to achieve the results of BN so as to speed up computation of batch normalization. The algorithm is shown as follows:
$$
\mu_B \gets \frac{1}{m}\sum_{i=1}^m x_i \\
C(x_i) \gets (x_i-\mu_B) \\
\sigma^2_B \gets \frac{1}{m} \sum_{i=1}^m (C(x_i)\ll\gg AP2(C(x_i))) \\
\hat{x_i} \gets C(x_i) \ll \gg AP2((\sqrt{\sigma^2_B+\epsilon})^{-1}) \\
y_i \gets AP2(\gamma) \ll \gg \hat{x_i} \tag{6}
$$
Where AP2 is the approximate power-of-2, $$\ll\gg$$ indicates both left and right binary shift operations.

### Shift-based AdaMax

Since ADAM requires many multiplications, the authors suggested to use shift-based AdaMax which is shown as follows:
$$
m_t \gets \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
v_t \gets \max(\beta_2 \cdot v_{t-1}, g|t|) \\
\theta_t \gets \theta_{t-1} - (\alpha \ll \gg (1-\beta_1)) \cdot \hat{m} \ll \gg v_t^{-1} \tag{7}
$$
Where $$g_t^2$$ indicates the element-wise square $$g_t\circ g_t$$. Good default setting are $$\alpha=2^{-10},1-\beta_1=2^{-3},1-\beta_2=2^{-10}$$. All operations on vectors are element-wise and $$\beta_1^t$$, $$\beta_2^t$$ denote $$\beta_1$$ and $$\beta_2$$ to the power $$t$$.

### Binarized Input

Since the input representation has much fewer channels than the internal representations in computer vision and it is easy to convert continuous-valued inputs to fixed point numbers, the authors suggested to compute output of first layer by:
$$
s=x \cdot w^b \\
s=\sum_{n=1}^8 2^{n-1}(x^n \cdot w^b) \tag{8}
$$
where $$x$$ is a vector of 1024 8-bit inputs, $$x_1^8$$ is the most significant bit of the first input, $$w^b$$ is a vector of 1024 1-bit weights and $$s$$ is the resulting weighted sum.

## Training Method

**Step 1, forward:** binarized weights and apply SBN

**Step 2, backward:** compute real-valued gradient $$g_a$$ with constraint descripted in Equation (4), and compute gradient of weights 

**Step 3, update:** update weights with constraint descripted in Equation (4)

**Repeating:** repeating step 1 to step 3, until finish the training.

## Experimental Results

The authors evaluated their method on three data sets including MNIST, SVHN and CIFAR-10, results are shown as follows:

| Method     | MNIST | SVHN  | CIFAR-10 |
| ---------- | ----- | ----- | -------- |
| BNN Torch7 | 1.40% | 2.53% | 10.15%   |
| BNN Theano | 0.96% | 2.80% | 11.40%   |

# Extension of BNN

Following the work of BNN, the authors proposed a [training method](https://www.ganghua.org/publication/AAAI17.pdf) to improve performance of BNN in four folds: (1) using low learning rate (the authors suggested to use the learning rate of 1e-4); (2) using PReLU instead of ReLU to absorb the scaling factor for weights to the activation function; (3) introducing a regularization term to the loss function to encourage the weights to be bipolar; (4) using scale layer in fully connected layer to bring the outputs to normal.

The regularization term introduced in this paper is formulated by:
$$
J(W,b) = L(W,b)+\lambda \sum_{l=1}^L \sum_{i=1}^{N_l} \sum_{j=1}^{M_l} (1-(W_{l,ij})^2) \tag{9}
$$
To improve the accuracy, the authors used multiple binarizations for the activation:
$$
A_l \approx \sum_{i=1}^m (\alpha_{l,i} H{l,i}) \tag{10}
$$
For $$i=1$$, $$H_{l,1}$$ is the sign of $$A_l$$ and $$\alpha_{l,i}$$ is the average absolute value of $$A_l$$, for $$i\gt 1$$, $$H_{l,i}$$ and $$\alpha_{l,i}$$ is calculated in the way based on residual approximation error from step $$i-1$$: $$E_{L,I} = a_l-\sum_{j=1}^{i-1}\alpha_{l,j}\ast H_{l,j}$$. So the output $$O_l$$ is calculated by:
$$
O_l = W_l \cdot A_{l-1} \approx \sum_{i=1}^m (\alpha_{l-1,i}xnor-popcnt(B_l, H_{l-1,i})) \tag{11}
$$

## Experimental Results

The authors conducted experiments on ImageNet with AlexNet and NIN, the results are shown as follows:

| Method               | Bits of Activation | Precision of Last Layer | Compression Rate | Accuracy        |
| -------------------- | ------------------ | ----------------------- | ---------------- | --------------- |
| AlexNet BNN          | 1                  | Full                    | 10.3$$\times$$   | 50.4/ 27.9      |
| AlexNet XNOR-net     | 1                  | Full                    | 10.3$$\times$$   | 69.2 / 44.2     |
| AlexNet DoReFa       | 2                  | Full                    | 10.3$$\times$$   | - / 49.8        |
| AlexNet Extended-BNN | 2                  | Binary                  | 31.2$$\times$$   | **71.1 / 46.6** |
| NIN Extended-BNN     | 2                  | Binary                  | 23.6 $$\times$$  | **75.6 / 51.4** |