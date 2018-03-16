import skimage
import skimage.io as io
import skimage.transform 
import sys
import numpy as np
import os
import math
import caffe2.python.predictor.predictor_exporter as pe
from matplotlib import pyplot
import matplotlib.image as mpimg
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
    def __init__(self, model, is_test, sp_batch_norm=.99):
        self.model = model
        # self.prev_blob = prev_blob
        self.is_test = is_test
        self.sp_batch_norm = sp_batch_norm
        self.layer_num = 1
        self.block_name = ''

    def add_relu_activ(self):
        self.prev_blob = brew.relu(
            self.model,
            self.prev_blob,
            self.prev_blob,
            )
        return self.prev_blob

    def add_sp_batch_norm(self, filters):
        self.prev_blob = brew.spatial_bn(
            self.model,
            self.prev_blob,
            '%s_sp_batch_norm_%d' % (self.block_name, self.layer_num),
            filters,
            epsilon=1e-3,
            momentum=self.sp_batch_norm,
            is_test= self.is_test,
            )
        return self.prev_blob

    def calculate_padding(self, prev_blob, kerenl, stride):
        if prev_blob is not None:
            row = prev_blob.Shape[2]
            col = prev_blob.Shape[1]
            R_out = ceil(float(row)/float(stride))
            C_out = ceil(float(col)/float(stride))

            Pr = ((R_out - 1) * stride + kernel - row)
            Pc = ((C_out - 1) * stride + kernel - col)
            return [Pr, Pc]
        else:
            row = self.prev_blob.Shape[2]
            col = self.prev_blob.Shape[1]
            R_out = ceil(float(row)/float(stride))
            C_out = ceil(float(col)/float(stride))

            Pr = ((R_out - 1) * stride + kernel - row)
            Pc = ((C_out - 1) * stride + kernel - col)
            return [Pr, Pc]

    def add_conv_layer(self, input_filters, output_filters, kernel, padding, stride= 1, prev_blob= None):
        pad = calculate_padding(prev_blob, kernel, stride)

        if prev_blob == None:
            if padding == 'same':
                self.prev_blob = brew.conv(
                self.model,
                self.prev_blob,
                '%s_conv_%d' % (self.block_name, self.layer_num),
                input_filters,
                output_filters,
                kernel= kernel,
                stride= stride,
                pad= pad,
                no_bias= True,
                )
            else:
                self.prev_blob = brew.conv(
                    self.model,
                    self.prev_blob,
                    '%s_conv_%d' % (self.block_name, self.layer_num),
                    input_filters,
                    output_filters,
                    kernel= kernel,
                    stride= stride,
                    pad= 0,
                    no_bias= True,
                    )
        else:
            if padding == 'same':
                self.prev_blob = brew.conv(
                    self.model,
                    prev_blob,
                    '%s_conv_%d' % (self.block_name, self.layer_num),
                    input_filters,
                    output_filters,
                    kernel= kernel,
                    stride= stride,
                    pad= pad,
                    no_bias= True,
                    )
            else:
                self.prev_blob = brew.conv(
                    self.model,
                    prev_blob,
                    '%s_conv_%d' % (self.block_name, self.layer_num),
                    input_filters,
                    output_filters,
                    kernel= kernel,
                    stride= stride,
                    pad= 0,
                    no_bias= True,
                    )

        # self.add_sp_batch_norm(output_filters)
        self.add_relu_activ()
        self.layer_num += 1

        return self.prev_blob

    def add_max_pool(self, prev_blob, kernel, stride, pad= 0):
        self.prev_blob = brew.max_pool(
            self.model,
            prev_blob,
            '%s_max_pool_%d' % (self.block_name, self.layer_num),
            kernel= kernel,
            stride= stride,
            pad= pad,
            )
        self.layer_num += 1
        return self.prev_blob

    def add_avg_pool(self, prev_blob, kernel= 3, stride= 1, global_pool= False):
        self.prev_blob = brew.average_pool(
            self.model,
            prev_blob,
            '%s_avg_pool_%d' % (self.block_name, self.layer_num),
            kernel= kernel,
            stride= stride,
            global_pooling= global_pool,
            )
        self.layer_num += 1
        return self.prev_blob

    def add_dropout(self, prev_blob, ratio, is_test= False):
        self.prev_blob = brew.dropout(
            self.model,
            prev_blob,
            '%s_dropout_%d' % (self.block_name, self.layer_num),
            ratio= ratio,
            is_test= is_test
            )
        self.layer_num += 1
        return self.prev_blob

    def add_softmax(self, prev_blob, label= None):
        if label is not None:
            (softmax, loss) = self.model.SoftmaxWithLoss(
            [prev_blob, label],
            ['softmax', 'loss'],
            ) 
            return (softmax, loss)
        else:
            return brew.softmax(self.model, prev_blob, 'softmax')

    def concat_layers(self, *args, axis=1):
        self.prev_blob = brew.concat(
            self.model,
            args,
            '%s_concat_%d' % (self.block_name, self.layer_num),
            # add_axis= 0,
            # axis= axis,
            )

        self.layer_num += 1
        return self.prev_blob

    def Inception_Stem(self, data):
        self.add_conv_layer(3, 32, [3,3], 'valid', stride= 2, prev_blob= data)
        self.add_conv_layer(32, 32, [3,3], 'valid')
        conv0 = self.add_conv_layer(32, 64, [3,3], 'same')

        conv1 = self.add_conv_layer(64, 96, [3,3], 'valid', stride= 2, prev_blob= conv0)

        maxpool1 = self.add_max_pool(conv0, 3, 2)

        concat1 = self.concat_layers(conv1, maxpool1)

        # utilize prev_blob here
        self.add_conv_layer(160, 64, [1, 1], 'same', prev_blob= concat1)
        conv2 = self.add_conv_layer(64, 96, [3,3], 'valid')

        self.add_conv_layer(160, 64, [1, 1], 'same', prev_blob= concat1)
        self.add_conv_layer(64, 64, [1, 7], 'same')
        self.add_conv_layer(64, 64, [7, 1], 'same')
        conv3 = self.add_conv_layer(64, 96, [3,3], 'valid')

        concat2 = self.concat_layers(conv2, conv3)

        local_prev = self.add_max_pool(self.prev_blob, [1, 1], 2)
        self.add_conv_layer(192, 192, [3,3], 'valid', prev_blob= concat2)

        self.concat_layers(local_prev, self.prev_blob)

        return self.prev_blob

    def Inception_A(self, input):

        self.add_avg_pool(input)
        layer_1 = self.add_conv_layer(384, 96, [1,1], 'same')

        layer_2 = self.add_conv_layer(384, 96, [1,1], 'same', prev_blob= input)

        self.add_conv_layer(384, 64, [1,1], 'same', prev_blob = input)
        layer_3 = self.add_conv_layer(64, 96, [3,3], 'same')

        self.add_conv_layer(384, 64, [1,1], 'same', prev_blob= input)
        self.add_conv_layer(64, 96, [3,3], 'same')
        layer_4 = self.add_conv_layer(64, 96, [3,3], 'same')

        return self.concat_layers(layer_1, layer_2, layer_3, layer_4)

    def Inception_B(self, input):

        self.add_avg_pool(input)
        layer_1 = self.add_conv_layer(1024, 128, [1,1], 'same')

        layer_2 = self.add_conv_layer(1024, 384, [1,1], 'same', prev_blob= input)

        self.add_conv_layer(1024, 192, [1,1], 'same', prev_blob= input)
        self.add_conv_layer(192, 224, [1, 7], 'same')
        layer_3 = self.add_conv_layer(224, 256, [1, 7], 'same')

        self.add_conv_layer(1024, 192, [1,1], 'same', prev_blob= input)
        self.add_conv_layer(192, 192, [1, 7], 'same')
        self.add_conv_layer(192, 224, [7, 1], 'same')
        self.add_conv_layer(224, 224, [1, 7], 'same')
        layer_4 = self.add_conv_layer(224, 256, [7, 1], 'same')

        return self.concat_layers(layer_1, layer_2, layer_3, layer_4)

    def Inception_C(self, input):

        self.add_avg_pool(input)
        layer_1 = self.add_conv_layer(1536, 256, [1,1], 'same')

        layer_2 = self.add_conv_layer(1536, 256, [1,1], 'same', prev_blob= input)

        sub_layer_1 = self.add_conv_layer(1536, 384, [1,1], 'same', prev_blob= input)
        layer_3 = self.add_conv_layer(384, 256, [1, 3], 'same', prev_blob= sub_layer_1)
        layer_4 = self.add_conv_layer(384, 256, [3, 1], 'same', prev_blob= sub_layer_1)

        self.add_conv_layer(1536, 384, [1,1], 'same', prev_blob= input)
        self.add_conv_layer(384, 448, [1, 3], 'same')
        sub_layer_2 = self.add_conv_layer(448, 512, [3, 1], 'same')
        layer_5 = self.add_conv_layer(512, 256, [3, 1], 'same', prev_blob= sub_layer_2)
        layer_6 = self.add_conv_layer(512, 256, [1, 3], 'same', prev_blob= sub_layer_2)

        return self.concat_layers(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6)

    def Reduction_A(self, input):

        layer_1 = self.add_max_pool(input, 3, 2)

        layer_2 = self.add_conv_layer(384, 384, [3,3], 'valid', stride= 2, prev_blob= input)

        self.add_conv_layer(384, 192, [1,1], 'same', prev_blob= input)
        self.add_conv_layer(192, 224, [3,3], 'same',)
        layer_3 = self.add_conv_layer(224, 256, [3,3], 'valid', stride= 2)

        return self.concat_layers(layer_1, layer_2, layer_3)

    def Reduction_B(self, input):

        layer_1 = self.add_max_pool(input, 3, 2)

        self.add_conv_layer(1024, 192, 1, 'same', prev_blob= input)
        layer_2 = self.add_conv_layer(192, 192, [3,3], 'valid', stride= 2)

        self.add_conv_layer(1024, 256, [1,1], 'same', prev_blob= input)
        self.add_conv_layer(256, 256, [1, 7], 'same')
        self.add_conv_layer(256, 320, [7, 1], 'same')
        layer_3 = self.add_conv_layer(320, 320, [3,3], 'valid', stride= 2)

        return self.concat_layers(layer_1, layer_2, layer_3)

def create_Inceptionv4(model, data, num_labels, label= None, is_test= False, no_loss= False, no_bias= True):

    inception = Inceptionv4(model, is_test)

    inception.block_name = 'stem'
    prev_blob = inception.Inception_Stem(data)
    
    inception.block_name = 'block_A'
    inception.layer_num = 1

    for i in range(4):
        prev_blob = inception.Inception_A(prev_blob)

    inception.block_name = 'reduction_A'
    inception.layer_num = 1

    prev_blob = inception.Reduction_A(prev_blob)

    inception.block_name = 'block_B'
    inception.layer_num = 1

    for i in range(7):
        prev_blob = inception.Inception_B(prev_blob)

    inception.block_name = 'reduction_B'
    inception.layer_num = 1

    prev_blob = inception.Reduction_B(prev_blob)

    inception.block_name = 'block_C'
    inception.layer_num = 1

    for i in range(3):
        prev_blob = inception.Inception_C(prev_blob)

    inception.block_name = 'end_layers'
    inception.layer_num = 1

    prev_blob = inception.add_avg_pool(prev_blob)
    prev_blob = inception.add_dropout(prev_blob, .8)

    return inception.add_softmax(prev_blob, label)
