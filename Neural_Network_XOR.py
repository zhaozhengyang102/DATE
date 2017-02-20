from collections import Counter

from matplotlib import pyplot as plt

import numpy
import math

def sigmoid(t):
    return 1 / (1 + math.exp(-t))#积分连续化

def neuron_output(weights, inputs):
    return sigmoid(numpy.dot(weights, inputs))#点乘运算x= a1*b1+a2*b2+a3*b3

def feed_forward(neural_network, input_vector):
    outputs = []
        # 先带入[[20, 20, -30],[20, 20, -10]]层，再带入[-60, 60, -30]层
    for layer in neural_network:
        input_with_bias = input_vector + [1]#加入偏移成为三维向量，从而实现点乘运算
        output = [neuron_output(neuron, input_with_bias) #运算得出sigmoid(-30)sigmoid(-10)
                  for neuron in layer]#按顺序取出[20, 20, -30],[20, 20, -10]
        outputs.append(output)#运算结果加入输出列
        input_vector = output#把第一层的结果传入第二层的输入值
    return outputs #返回全部层输出结果

xor_network = [[[20, 20, -30],[20, 20, -10]] #"和"，"或"层
    ,[[-60, 60, -30]]]#单隐层

for x in[0,1]:
    for y in[0,1]:
        print  ( x, y, feed_forward(xor_network,[x, y])[-1] )#调用feed_forward,返回列表最后一个值











