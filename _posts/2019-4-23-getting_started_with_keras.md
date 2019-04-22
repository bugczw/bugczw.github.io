---
layout: post
title: keras入门指南
date: 2019-4-22
tags: [Keras, 神经网络, 机器学习, 学习日志]
---

本文是基于[Keras中文文档](http://keras-cn.readthedocs.io/en/latest/) 的整理、总结与思考。


### 概述 ###

Keras作为基于python的深度学习库，由纯Python编写而成并基于Tensorflow、Theano以及CNTK后端。其主要优点是简易和快速的原型设计，CNN与RNN的结合应用，CPU和GPU的无缝切换。其主要有函数是模型构成。

### 模型分类 ###

Keras模型类型分为函数式(Functional)以及序贯(Sequential)模型。其中序贯模型本质是一系列网络层按顺序构成的栈，属于函数式模型的一种特数情况。其一个简单的示例如下：

    > ```python
    > from keras.models import Sequential
	> from keras.layers import Dense, Activation
	> model = Sequential()
	>
	> # 网络层通过.add()堆叠搭建
	> model.add(Dense(units=64, input_dim=100))
    > model.add(Activation("relu"))
    > model.add(Dense(units=10))
    > model.add(Activation("softmax"))
	>
	> # 通过.compile()编译模块
	> model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	>
	> # 通过SGD实现模型的SGD优化
	> from keras.optimizers import SGD
    > model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
	>
	> # 通过.fit()按batch进行一定次数的迭代来训练网络，默认前向传播训练网络
	> model.fit(x_train, y_train, epochs=5, batch_size=32)
	>
	> # 通过.train_on_batch()手动将数据送入网络训练
	> model.train_on_batch(x_batch, y_batch)
	>
	> # 通过.evaluate()实现模型的评估
	> loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    >
	> # 通过.predict()实现模型对新数据的预测
	> classes = model.predict(x_test, batch_size=128)
	> ```

### 网络结构 ###

![](/assets/getting_started_with_keras/01.jpg)

Keras常用的网络层为[高级激活层(Advanced Activation Layer)](https://keras-cn.readthedocs.io/en/latest/layers/advanced_activation_layer/) 、[卷积层(Convolutional Layer)](https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/) 、[常用层(Core Layer)](https://keras-cn.readthedocs.io/en/latest/layers/core_layer/)  、[嵌入层 (Embedding Layer)](https://keras-cn.readthedocs.io/en/latest/layers/embedding_layer/) 、[局部连接层(Locally Connceted Layer)](https://keras-cn.readthedocs.io/en/latest/layers/locally_connected_layer/) 、[合并层(Merge Layer)](https://keras-cn.readthedocs.io/en/latest/layers/merge/) 、[噪声层(Noise Layer)](https://keras-cn.readthedocs.io/en/latest/layers/noise_layer/) 、[批规范化层(Batch Normalization Layer)](https://keras-cn.readthedocs.io/en/latest/layers/normalization_layer/) 、[池化层(Pooling Layer)](https://keras-cn.readthedocs.io/en/latest/layers/pooling_layer/) 、 [递归层(Recurrent Layer)](https://keras-cn.readthedocs.io/en/latest/layers/recurrent_layer/) 、[包装器(Wrapper)](https://keras-cn.readthedocs.io/en/latest/layers/wrapper/) 、[自定义层(Writing Layer)](https://keras-cn.readthedocs.io/en/latest/layers/writting_layer/) 。

### 网络配置 ###

![](/assets/getting_started_with_keras/02.jpg)

### 预处理功能 ###

![](/assets/getting_started_with_keras/03.jpg)

### 个人总结 ###

Keras集合了Tensorflow、Theano以及CNTK，提供使用者简单便利的API接口。其模型的构建基于一层层的层叠加，便于模型的拓展。同时Keras提供可视化接口便于模型的理解。

### 参考资料 ###

[Keras官方文档](https://keras.io/)
[Keras中文文档](http://keras-cn.readthedocs.io/en/latest/)
[CSDN博客](https://blog.csdn.net/sinat_26917383/article/details/72857454)