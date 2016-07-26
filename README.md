# tfdepth

## What is this?
An implementation of [stochastic depth](http://arxiv.org/pdf/1603.09382v2.pdf) in the [RESnet](https://github.com/xuyuwei/resnet-tf) architecture using TensorFlow

## What does this do?
Put simply, this is a modification to the RESnet neural network which stochastically skips layers during training. This allows for more efficient and (more importantly) quicker training, without sacrificing model accuracy.

## Motivations
I have seen implementations of the paper in a few other Machine Learning frameworks like Theano, but none using TensorFlow, so I thought I'd take a crack at making one myself. Such an implementation would be especially useful for commercial/industry purposes, since the TensorFlow framework is advantageous with such use cases for a number of reasons (supported/maintained by Google, easy graph visualization, deployability, etc.)



