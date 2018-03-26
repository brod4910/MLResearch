import os,sys,inspect
import numpy as np
from math import ceil
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    visualize,
    workspace,
    )

class Layer_Adder():
    def __init__(self, model, is_test, sp_batch_norm= .99):
        self.model = model
        self.layer_num = 1
        self.block_name = ''
        self.is_test = is_test
        self.sp_batch_norm = sp_batch_norm
        self.prev_blob = None

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

    def calculate_padding(self, padding_type, kernel):
        assert padding_type in ['same', 'valid']
        if padding_type == 'same':
            return tuple((k - 1) // 2 for k in kernel)
        return tuple(0 for __ in kernel)

    def add_conv_layer(self, input_filters, output_filters, kernel, padding_type, stride= 1, prev_blob= None):
        padding = self.calculate_padding(padding_type, kernel)

        if prev_blob is None:
            self.prev_blob = brew.conv(
            self.model,
            self.prev_blob,
            '%s_conv_%d' % (self.block_name, self.layer_num),
            input_filters,
            output_filters,
            kernel= kernel,
            stride= stride,
            pad_t= padding[0],
            pad_b= padding[0],
            pad_l= padding[1],
            pad_r= padding[1],
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
                pad_t= padding[0],
                pad_b= padding[0],
                pad_l= padding[1],
                pad_r= padding[1],
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

    def add_avg_pool(self, prev_blob, kernel= 3, stride= 1, pad= 1,global_pool= False):
        self.prev_blob = brew.average_pool(
            self.model,
            prev_blob,
            '%s_avg_pool_%d' % (self.block_name, self.layer_num),
            kernel= kernel,
            stride= stride,
            pad= pad,
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

    def add_fc_layer(self, prev_blob, num_inputs, num_labels):
        self.prev_blob = brew.fc(
                self.model,
                prev_blob,
                'pred',
                num_inputs,
                num_labels,
                )

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
            axis= axis,
            )

        self.layer_num += 1
        return self.prev_blob