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
caffe2.python import data_parallel_model as dpm
from caffe2.python import timeout_guard
from caffe2.proto import caffe2_pb2
from IPython import display

import vgg19
# from caffe2.python.modeling.initializers import Initializer, PseudoFP16Initializer
import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.python.predictor_constants import predictor_constants as predictor_constants


def Train_Model(args):
    current_folder = os.path.join(sys.path[0], '../../', 'datasets',
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


    train_data_db = os.path.join(data_folder,'blood_cells_train_lmdb')
    test_data_db = os.path.join(data_folder, 'blood_cells_test_lmdb')

    arg_scope = {
    "order" : "NCHW",
    "use_cudnn" : args.gpu,
    }
    train_model = model_helper.ModelHelper(name="vgg19_train", arg_scope=arg_scope)

    reader = train_model.CreateDB(
        "train_reader",
        db= train_data_db,
        db_type= 'lmdb',
        )

    devices = []
    for i in range(args.shards):
        devices.append(i)

    train_data_count = 9893

    batch_per_device = 16
    total_batch_size = batch_per_device * len(devices)

    num_labels = 4

    base_learning_rate = .001 * total_batch_size

    weight_decay = (5 * 10**(−4))

    stepsize = int(2 * train_data_count / total_batch_size)

    def add_image_input_ops(model):
        '''
        The image input operator loads image and label data from the reader and
        applies transformations to the images (random cropping, mirroring, ...).
        '''
        data, label = model.ImageInput(
            reader,
            ["data", "label"],
            batch_size=batch_per_device,
            # mean: to remove color values that are common
            # mean=128.,
            # std is going to be modified randomly to influence the mean subtraction
            # std=128.,
            # scale to rescale each image to a common size
            scale=224,
            # crop to the square each image to exact dimensions
            crop=224,
            # not running in test mode
            is_test=False,
            # mirroring of the images will occur randomly
            mirror=1
        )
        # prevent back-propagation: optional performance improvement; may not be observable at small scale
        data = model.StopGradient(data, data)

    def create_vgg_model_ops(model, loss_scale= 1.0):
        [softmax, loss] = vgg19.create_VGG19(model, "data", num_labels, "label")

        prefix = model.net.Proto().name
        loss = model.net.Scale(loss, prefix + "_loss", scale= loss_scale)
        brew.accuracy(model, [softmax, "label"], prefix + "_accuracy")
        return [loss]

    def add_parameter_update_ops(model):
        brew.add_weight_decay(model, weight_decay)
        iter = brew.iter(model, "iter")
        lr = model.net.LearningRate(
            [iter],
            "lr",
            base_lr=base_learning_rate,
            policy="step",
            stepsize=stepsize,
            gamma=0.1,
        )
        for param in model.GetParams():
            param_grad = model.param_to_grad[param]
            param_momentum = model.param_init_net.ConstantFill(
                [param], param + '_momentum', value=0.0
            )

            # Update param_grad and param_momentum in place
            model.net.MomentumSGDUpdate(
                [param_grad, param_momentum, lr, param],
                [param_grad, param_momentum, param],
                # almost 100% but with room to grow
                momentum=0.9,
                # netsterov is a defenseman for the Montreal Canadiens, but
                # Nesterov Momentum works slightly better than standard momentum
                nesterov=1,
            )

    if args.gpu:
        dpm.Parallelize_GPU(
            train_model,
            input_builder_fun=add_image_input_ops,
            forward_pass_builder_fun=create_vgg_model_ops,
            param_update_builder_fun=add_parameter_update_ops,
            devices=devices,
            optimize_gradient_memory=True,
        )
    else:
        dpm.Parallelize_CPU(
            train_model,
            input_builder_fun=add_image_input_ops,
            forward_pass_builder_fun=create_vgg_model_ops,
            param_update_builder_fun=add_parameter_update_ops,
            devices=devices,
            optimize_gradient_memory=True,
            )

    train_model.Print('accuracy', [], to_file=1)
    train_model.Print('loss', [], to_file=1)
    train_model.Print('label', [], to_file=1)
    train_model.Print('softmax', [], to_file=1)

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)

    def accuracy(model):
        accuracy = []
        prefix = model.net.Proto().name
        for device in model._devices:
            accuracy.append(
                np.asscalar(workspace.FetchBlob("{}_{}/{}_accuracy".format('gpu' if args.gpu else 'cpu', device, prefix))))
        return np.average(accuracy)

    # graph = net_drawer.GetPydotGraphMinimal(train_model.net.Proto().op, "Inception", rankdir="LR", minimal_dependency=True)
    # graph.write_png("VGG19.png")


def GetArgParser():
    parser = argparse.ArgumentParser(description='VGG19 Train')

    parser.add_argument(
        '--shards', 
        type=int, 
        default=1)
    parser.add_argument(
        '--gpu',
        action='store_true',
        )

    return parser

if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

    args = GetArgParser().parse_known_args()

    Train_Model(args)