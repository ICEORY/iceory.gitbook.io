# ThiNet: a filter level pruning method for deep neural network

experimental results on ImageNet:
Baseline: 
top1error: 23.99, top5error: 7.07

Method: use features from original model as input and prune channels layer by layer without fine-tuning.
pruning 30% of channels:
top1error: 24.58 (0.59), top5error: 7.47 (0.40)
pruning 50% of channels:
top1error: 26.24 (2.25), top5error: 8.33 (1.26)

Method: use features from pruned model as input and prune channels layer by layer without fine-tuning.
pruning 30% of channels:
top1error: 24.94 (0.95), top5error: 7.58 (0.51)
pruning 50% of channels:
top1error: 26.58 (2.59), top5error: 8.37 (1.30)

