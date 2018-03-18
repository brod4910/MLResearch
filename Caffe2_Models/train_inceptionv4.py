from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import time
import os
import shutil
from matplotlib import pyplot

from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    visualize,
    workspace,
)
from caffe2.python import data_parallel_model_utils, dyndep, optimizer
from caffe2.python import timeout_guard
from caffe2.proto import caffe2_pb2
from IPython import display

import inceptionv4
# from caffe2.python.modeling.initializers import Initializer, PseudoFP16Initializer
import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.python.predictor_constants import predictor_constants as predictor_constants

def AddImageInput(model, reader, batch_size, img_size, dtype, is_test):
    '''
    The image input operator loads image and label data from the reader and
    applies transformations to the images (random cropping, mirroring, ...).
    '''
    data, label = brew.image_input(
        model,
        reader, ["data", "label"],
        batch_size=batch_size,
        # output_type=dtype,
        use_gpu_transform=False,
        use_caffe_datum=False,
        # mean=128.,
        # std=128.,
        scale=299,
        crop=img_size,
        db_type= 'lmdb',
        # mirror=1,
        is_test=is_test,
    )

    data = model.StopGradient(data, data)
    return data, label

def Train_Model():
    current_folder = os.path.join(os.path.expanduser('~'), 'Development', 'MLResearch', 'datasets',
     'blood-cells')
    print(current_folder)
    data_folder = os.path.join(current_folder, 'dataset2-master')
    root_folder = current_folder
    db_missing = False

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)   
        print("Your data folder was not found!! This was generated: {}".format(data_folder))

    # Look for existing database: lmdb
    if os.path.exists(os.path.join(data_folder,'blood_cells_train_lmdb')):
        print("lmdb train db found!")
    else:
        db_missing = True
        
    if os.path.exists(os.path.join(data_folder,'blood_cells_test_lmdb')):
        print("lmdb test db found!")
    else:
        db_missing = True

    arg_scope = {
    "order" : "NCHW",
    "use_cudnn" : False,
    }
    train_model = model_helper.ModelHelper(name="inceptionv4_train", arg_scope=arg_scope)

    reader = train_model.CreateDB(
        "reader",
        db= os.path.join(data_folder,'blood_cells_train_lmdb'),
        db_type= 'lmdb',
        )

    data, label = AddImageInput(train_model, reader, 64, 299, 'lmdb', False)

    # data, label = AddInput(train_model, batch_size=64, db=os.path.join(data_folder,'blood_cells_train_lmdb'), db_type='lmdb')

    softmax, loss = inceptionv4.create_Inceptionv4(train_model, data, 4, label)

    accuracy = train_model.Accuracy([softmax, label], "accuracy")

    train_model.AddGradientOperators([loss])
    optimizer.build_adam(
        train_model,
        base_learning_rate=.001,
        )

    train_model.Print('accuracy', [], to_file=1)
    train_model.Print('loss', [], to_file=1)
    train_model.Print('stem_conv_8', [])
    train_model.Print('stem_conv_12', [])

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)

    # pyplot.figure()
    # d = workspace.FetchBlob('data')
    # # _ = visualize.NCHW.ShowMultiple(data)
    # pyplot.show()

    total_iters = 200
    accuracy = np.zeros(total_iters)
    loss = np.zeros(total_iters)

    for i in range(total_iters):
        workspace.RunNet(train_model.net)
        accuracy[i] = workspace.blobs['accuracy']
        loss[i] = workspace.blobs['loss']

    pyplot.plot(loss, 'b')
    pyplot.plot(accuracy, 'r')
    pyplot.legend(('Loss', 'Accuracy'), loc='upper right')

    pyplot.show()
    # for b in workspace.Blobs(): print(b)

    # graph = net_drawer.GetPydotGraphMinimal(train_model.net.Proto().op, "Inception", rankdir="LR", minimal_dependency=True)
    # graph.write_png("Inception.png")


if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    Train_Model()