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

class VGG19():

    def __init__(self, model, is_test):
        self.model = model
        self.la = layer_adder.Layer_Adder(model, is_test)

    def VGG_block_0(self, input):
        self.la.add_conv_layer(3, 64, [3, 3], 'same', prev_blob= input)
        conv1 = self.la.add_conv_layer(64, 64, [3, 3], 'same')
        return self.la.add_max_pool(conv1, 2, 2)

    def VGG_block_1(self, input):
       self.la.add_conv_layer(64, 128, [3, 3], 'same', prev_blob= input)
       conv1 = self.la.add_conv_layer(128, 128, [3, 3], 'same')
       return self.la.add_max_pool(conv1, 2, 2)

    def VGG_block_2(self, input):
        self.la.add_conv_layer(128, 256, [3, 3], 'same', prev_blob= input)

        for i in range(3):
            conv = self.la.add_conv_layer(256, 256, [3, 3], 'same')

        return self.la.add_max_pool(conv, 2, 2)

    '''
        This is techincally both block 3 and 4.
        Input_filter controls the type of block it is.
    '''
    def VGG_block_3_4(self, input, input_filter):
        self.la.add_conv_layer(input_filter, 512, [3, 3], 'same', prev_blob= input)

        for i in range(3):
            conv = self.la.add_conv_layer(512, 512, [3, 3], 'same')

        return self.la.add_max_pool(conv, 2, 2)

    def VGG_block_final(self, input, num_labels):
        fc1 = self.la.add_fc_layer(input, 512 * 7 * 7, 4096)
        dp1 = self.la.add_dropout(fc1, .5)
        flat1 = self.la.model.Flatten(dp1)

        fc2 = self.la.add_fc_layer(flat1, 4096, 4096)
        dp2 = self.la.add_dropout(fc2, .5)
        flat2 = self.la.model.Flatten(dp2)

        return self.la.add_fc_layer(flat2, 4096, num_labels)

def create_VGG19(model, data, num_labels, label= None, is_test= False, no_loss= False, no_bias= True):
    vgg19 = VGG19(model, is_test)

    vgg19.la.block_name = 'block_0'
    prev_blob = vgg19.VGG_block_0(data)

    vgg19.la.block_name = 'block_1'
    vgg19.la.layer_num = 1
    prev_blob = vgg19.VGG_block_1(prev_blob)

    vgg19.la.block_name = 'block_2'
    vgg19.la.layer_num = 1
    prev_blob = vgg19.VGG_block_2(prev_blob)

    vgg19.la.block_name = 'block_3'
    vgg19.la.layer_num = 1
    prev_blob = vgg19.VGG_block_3_4(prev_blob, 256)

    vgg19.la.block_name = 'block_4'
    vgg19.la.layer_num = 1
    prev_blob = vgg19.VGG_block_3_4(prev_blob, 512)

    vgg19.la.block_name = 'block_final'
    vgg19.la.layer_num = 1
    prev_blob = vgg19.VGG_block_final(prev_blob, num_labels)

    return vgg19.la.add_softmax(prev_blob, label= label)


