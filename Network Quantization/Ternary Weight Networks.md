# TWN (Ternary Weight Networks)

The authors introduced [ternary weight networks](https://www.researchgate.net/publication/303270485_Ternary_Weight_Networks?ev=auth_pub) (TWNs) to address the limited storage and computational resources issues in hardware. The quantization problem can be formulated as follows:
$$
\begin{cases}
\alpha^*, W^{t*} = &\arg\min_{\alpha, W^t} J(\alpha, W^t) = \|W-\alpha W^t\|_2^2 \\ 
s.t. & a\ge0, W_i^t\in\{-1,0,1\}, i=1,2,\dots, n. 
\end{cases} \tag{1}
$$
Here $$n$$ is the size of filter, $$W$$ represents weights of the network. With $$W\approx \alpha W^t$$ and assuming the convolutional layer do not have bias term, forward propagation of ternary weight networks is as follows:
$$
\begin{cases}
Z & = &X*W \approx X*(\alpha W^t) = (\alpha X)\oplus W^t \\
X^{next} & = & g(Z)
\end{cases} \tag{2}
$$
where $$X$$ indicates inputs, $$*$$ indicates convolutional operation, $$g$$ is the non-linear activation function, $$\oplus$$ indicates the inner product or convolutional operation without any multiplication, $$X^{next}$$ indicates the outputs.

The approximated solution of $$W$$ with threshold-based ternary function is as follows:
$$
W_i^t = f_t(W_i|\triangle) = 
\begin{cases}
+1, if~W_i \gt \triangle \\
0, if~|W_i| \le \triangle \\
-1, if~W_i \lt -\triangle 
\end{cases} \tag{3}
$$
The optimized objective function can be written as:
$$
\alpha^*, \triangle^* = \arg\min_{\alpha\ge0, \triangle\gt 0}(|I_\triangle|\alpha^2-2(\sum_{i\in I_\triangle}|W_i|)\alpha+c_\triangle) \tag{4}
$$
where $$I_\triangle = \{i|~|W|\gt\triangle\}$$ and $$|I_\triangle|$$ denotes the number of elements in $$I_\triangle$$; $$c_\triangle = \sum_{i\in I_\triangle^c}W_i^2$$ is a $$\alpha$$-independent constant. Thus the optimal solutions of the objective function can be computed as follows: 
$$
\alpha_\triangle^* = \frac{1}{|I_\triangle|}\sum_{i\in I_\triangle}|W_i| \\
\triangle^* = \arg\max_{\triangle\gt 0}(\sum_{i\in I_\triangle}|W_i|^2) \tag{5}
$$
Here solution of $$\triangle$$ is approximated by $$\triangle^*\approx0.7\cdot E(|W|) \approx \frac{0.7}{n}\sum_{i=1}^n|W_i|$$.

## Training Methods

The training of ternary weight networks can be summarized to three steps: quantization, training and updating. Quantization phase is to quantize the weights of convolutional layers using Equation (5), then apply standard forward and backward propagation to the network, and update parameters using standard SGD. [Source code](https://github.com/fengfu-chris/caffe-twns) is available on GitHub.

### Important Tips

- $$\alpha$$ is used as the scaling factor for input $$X$$ not for weights $$W$$
- gradient is computed using $$W^t$$ 
- quantized weights are used during forward and backward but not during parameters update, so $$W_l^r \gets W_l^r - \eta \frac{\partial C}{\partial W_l^t} $$
- first compute $$\triangle$$ then compute mask, finally compute $$\alpha$$ 
- $$\triangle$$ is computed with all $$|W^r|$$ while $$\alpha$$ is computed only with those $$|W^r|>\triangle$$
- apply weight_decay would lead results worse 
- [this blog](http://blog.csdn.net/xjtu_noc_wei/article/details/52862282) is useful for implementation

## Experimental Results

Three data sets are used in this paper, including MNIST, CIFAR-10, ImageNet. To different data sets, the authors conducted experiments using LeNet-5 (32-C5 + MP2 + 64-C5 + MP2 + 512FC + SVM), VGG-inspired network (2$$\times$$(128-C3) + MP2 + 2$$\times$$(256-C3) + MP2 + 2$$\times$$(512-C3) + MP2 + 1024-FC + Softmax), ResNet-18, respectively. Network architecture and parameters setting for different data sets are shown as follows:

|                                          | MNIST   | CIFAR-10 | ImageNet             |
| ---------------------------------------- | ------- | -------- | -------------------- |
| network architecture                     | LeNet-5 | VGG-7    | ResNet-18 (B)        |
| weight decay                             | 1e-4    | 1e-4     | 1e-4                 |
| mini-batch size of BN                    | 50      | 100      | 64($$\times$$4 GPUs) |
| initial learning rate                    | 0.01    | 0.1      | 0.1                  |
| learning rate decay (divided by 10) epochs | 15, 25  | 80, 120  | 30, 40, 50           |
| momentum                                 | 0.9     | 0.9      | 0.9                  |

Comparison of the proposed method and the previous methods are shown as follows:

|          Method           | MINIST | CIFAR-10 | ImageNet Top1 (ResNet-18 / ResNet-18B) | ImageNet Top5 (ResNet-18 / ResNet-18B) |
| :-----------------------: | :----: | :------: | :------------------------------------: | :------------------------------------: |
|            TWN            | 99.35  |  92.56   |              61.8 / 65.3               |              84.2 / 86.2               |
|           BPWN            | 99.05  |  90.18   |              57.5 / 61.6               |              81.2 / 83.9               |
|   FPWN (full precision)   | 99.41  |  92.88   |              65.4 / 67.6               |              86.76 / 88.0              |
|      Binary Connect       | 98.82  |  91.73   |                   -                    |                   -                    |
| Binarized Neural Networks |  88.6  |  89.85   |                   -                    |                   -                    |
|  Binary Weight Networks   |   -    |    -     |                  60.8                  |                  83.0                  |
|         XNOR-Net          |   -    |    -     |                  51.2                  |                  73.2                  |