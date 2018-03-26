import os,sys,inspect
sys.path.insert(0, '../')
import layer_adder
import numpy as np
import caffe2.python.predictor.predictor_exporter as pe
from math import ceil
from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    visualize,
    workspace,
    )

class Inceptionv4():

    '''
    Args:
        model: is the intialized model_helper
        prev_blob: prev_blob is the data that was taken from the dataset
        sp_batch_norm: is the momentum for spatial batch normalization
    '''
    def __init__(self, model, is_test):
        self.model = model
        self.la = layer_adder.Layer_Adder(model, is_test)
        self.prev_blob = None

    def Inception_Stem(self, data):
        self.la.add_conv_layer(3, 32, [3, 3], 'valid', stride= 2, prev_blob= data)
        self.la.add_conv_layer(32, 32, [3, 3], 'valid')
        branch0 = self.la.add_conv_layer(32, 64, [3, 3], 'same')

        branch1 = self.la.add_conv_layer(64, 96, [3, 3], 'valid', stride= 2, prev_blob= branch0)

        maxpool1 = self.la.add_max_pool(branch0, 3, 2)

        concat1 = self.la.concat_layers(branch1, maxpool1)

        self.la.add_conv_layer(160, 64, [1, 1], 'same', prev_blob= concat1)
        branch2 = self.la.add_conv_layer(64, 96, [3, 3], 'valid')

        self.la.add_conv_layer(160, 64, [1, 1], 'same', prev_blob= concat1)
        self.la.add_conv_layer(64, 64, [1, 7], 'same')
        self.la.add_conv_layer(64, 64, [7, 1], 'same')
        branch3 = self.la.add_conv_layer(64, 96, [3, 3], 'valid')

        concat2 = self.la.concat_layers(branch2, branch3)

        maxpool2 = self.la.add_max_pool(concat2, 3, 2)

        branch4 = self.la.add_conv_layer(192, 192, [3, 3], 'valid', stride= 2, prev_blob= concat2)

        return self.la.concat_layers(maxpool2, branch4)


    def Inception_A(self, input):

        self.la.add_avg_pool(input)
        layer_1 = self.la.add_conv_layer(384, 96, [1, 1], 'same')

        layer_2 = self.la.add_conv_layer(384, 96, [1, 1], 'same', prev_blob= input)

        self.la.add_conv_layer(384, 64, [1, 1], 'same', prev_blob = input)
        layer_3 = self.la.add_conv_layer(64, 96, [3, 3], 'same')

        self.la.add_conv_layer(384, 64, [1, 1], 'same', prev_blob= input)
        self.la.add_conv_layer(64, 96, [3, 3], 'same')
        layer_4 = self.la.add_conv_layer(96, 96, [3, 3], 'same')

        return self.la.concat_layers(layer_1, layer_2, layer_3, layer_4)

    def Inception_B(self, input):

        self.la.add_avg_pool(input)
        layer_1 = self.la.add_conv_layer(1024, 128, [1, 1], 'same')

        layer_2 = self.la.add_conv_layer(1024, 384, [1, 1], 'same', prev_blob= input)

        self.la.add_conv_layer(1024, 192, [1, 1], 'same', prev_blob= input)
        self.la.add_conv_layer(192, 224, [7, 1], 'same')
        layer_3 = self.la.add_conv_layer(224, 256, [7, 1], 'same')

        self.la.add_conv_layer(1024, 192, [1, 1], 'same', prev_blob= input)
        self.la.add_conv_layer(192, 192, [7, 1], 'same')
        self.la.add_conv_layer(192, 224, [1, 7], 'same')
        self.la.add_conv_layer(224, 224, [7, 1], 'same')
        layer_4 = self.la.add_conv_layer(224, 256, [1, 7], 'same')

        return self.la.concat_layers(layer_1, layer_2, layer_3, layer_4)

    def Inception_C(self, input):

        self.la.add_avg_pool(input)
        layer_1 = self.la.add_conv_layer(1536, 256, [1, 1], 'same')

        layer_2 = self.la.add_conv_layer(1536, 256, [1, 1], 'same', prev_blob= input)

        sub_layer_1 = self.la.add_conv_layer(1536, 384, [1, 1], 'same', prev_blob= input)
        layer_3 = self.la.add_conv_layer(384, 256, [1, 3], 'same', prev_blob= sub_layer_1)
        layer_4 = self.la.add_conv_layer(384, 256, [3, 1], 'same', prev_blob= sub_layer_1)

        self.la.add_conv_layer(1536, 384, [1, 1], 'same', prev_blob= input)
        self.la.add_conv_layer(384, 448, [3, 1], 'same')
        sub_layer_2 = self.la.add_conv_layer(448, 512, [1, 3], 'same')
        layer_5 = self.la.add_conv_layer(512, 256, [3, 1], 'same', prev_blob= sub_layer_2)
        layer_6 = self.la.add_conv_layer(512, 256, [1, 3], 'same', prev_blob= sub_layer_2)

        return self.la.concat_layers(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6)

    def Reduction_A(self, input):

        layer_1 = self.la.add_max_pool(input, 3, 2)

        layer_2 = self.la.add_conv_layer(384, 384, [3, 3], 'valid', stride= 2, prev_blob= input)

        self.la.add_conv_layer(384, 192, [1, 1], 'same', prev_blob= input)
        self.la.add_conv_layer(192, 224, [3, 3], 'same',)
        layer_3 = self.la.add_conv_layer(224, 256, [3, 3], 'valid', stride= 2)

        return self.la.concat_layers(layer_1, layer_2, layer_3)

    def Reduction_B(self, input):

        layer_1 = self.la.add_max_pool(input, 3, 2)

        self.la.add_conv_layer(1024, 192, [1, 1], 'same', prev_blob= input)
        layer_2 = self.la.add_conv_layer(192, 192, [3, 3], 'valid', stride= 2)

        self.la.add_conv_layer(1024, 256, [1, 1], 'same', prev_blob= input)
        self.la.add_conv_layer(256, 256, [1, 7], 'same')
        self.la.add_conv_layer(256, 320, [7, 1], 'same')
        layer_3 = self.la.add_conv_layer(320, 320, [3, 3], 'valid', stride= 2)

        return self.la.concat_layers(layer_1, layer_2, layer_3)

def create_Inceptionv4(model, data, num_labels, label= None, is_test= False, no_loss= False, no_bias= True):

    inception = Inceptionv4(model, is_test)

    inception.la.block_name = 'stem'
    prev_blob = inception.Inception_Stem(data)
    
    inception.la.block_name = 'block_A'
    inception.la.layer_num = 1

    for i in range(4):
        prev_blob = inception.Inception_A(prev_blob)

    inception.la.block_name = 'reduction_A'
    inception.la.layer_num = 1

    prev_blob = inception.Reduction_A(prev_blob)

    inception.la.block_name = 'block_B'
    inception.la.layer_num = 1

    for i in range(7):
        prev_blob = inception.Inception_B(prev_blob)

    inception.la.block_name = 'reduction_B'
    inception.la.layer_num = 1

    prev_blob = inception.Reduction_B(prev_blob)

    inception.la.block_name = 'block_C'
    inception.la.layer_num = 1

    for i in range(3):
        prev_blob = inception.Inception_C(prev_blob)

    inception.la.block_name = 'end_layers'
    inception.la.layer_num = 1

    prev_blob = inception.la.add_avg_pool(prev_blob, kernel= 8, pad= 0)
    prev_blob = inception.la.add_dropout(prev_blob, .8)

    prev_blob = inception.model.Flatten(prev_blob)

    prev_blob = inception.la.add_fc_layer(prev_blob, 1536, num_labels)

    return inception.la.add_softmax(prev_blob, label)
