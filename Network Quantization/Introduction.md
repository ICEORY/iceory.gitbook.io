<!-- toc -->

# Introduction

---

Deep convolutional neural networks (DCNN) have achieved great performance on computer vision tasks, including  image classification, object detection and image segmentation. However, it is difficult to deploy deep learning (DL) methods in mobile devices, as deep neural networks always need high computational resource and large storage. Recent studies on adopting DL to hardware mainly focus on two folds: the first one is to design lighter architecture with comparable performance (i.e.  [SqueezeNet of Han et.al.](https://arxiv.org/abs/1602.07360) ) ; the second fold is network quantization, which aims to compress models and speed up computation of DCNN by converting high precision data to low precision version. Other coding methods including Huffman Coding, Compressed Sparse Column (CSC) and Hashing Coding also help reducing size of models. 

In this paper, we try to compare several quantization and compression methods as shown in follows:

- [Incremental network quantization](https://arxiv.org/abs/1702.03044) (INQ), a lossless quantization method with incremental idea. Combined with dynamic network surgery (DNS), the authors compressed AlexNet for 53$$\times$$ without accuracy loss
- [Binary weights network](https://arxiv.org/abs/1603.05279) (BWN) and XNOR-Net
- [Ternary weights network](https://arxiv.org/abs/1605.04711) (TWN)
- [Binarized neural networks](https://arxiv.org/abs/1602.02830) (BNN)
- [Trained ternary quantization](https://arxiv.org/abs/1612.01064) (TTQ)
- [Deep compression](https://arxiv.org/abs/1510.00149) with network quantization, pruning and Huffman code