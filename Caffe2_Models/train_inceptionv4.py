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

import inceptionv4
# from caffe2.python.modeling.initializers import Initializer, PseudoFP16Initializer
import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.python.predictor_constants import predictor_constants as predictor_constants


def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = brew.db_input(
        model,
        blobs_out=["data_uint8", "label"],
        batch_size=batch_size,
        db=db,
        db_type=db_type,
    )
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
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

	# print("training data folder:" + data_folder)
	# print("workspace root folder:" + root_folder)

	arg_scope = {"order" : "NCHW"}
	train_model = model_helper.ModelHelper(name="inceptionv4_train", arg_scope=arg_scope)


	data, label = AddInput(train_model, batch_size=64, db=os.path.join(data_folder,'blood_cells_train_lmdb'), db_type='lmdb')
	# print('Print data: \n', data)
	# print('Print labels: \n', label)

	inception = inceptionv4.create_Inceptionv4(train_model, data, 4, label)

	workspace.RunNetOnce(train_model.param_init_net)
	# workspace.RunNet(train_model.net)
	# print(str(train_model.net.Proto())[:] + '\n...')
	# print(str(train_model.param_init_net.Proto())[:400] + '\n...')

	for b in workspace.Blobs(): print(b)

	# pyplot.figure()
	# blob = workspace.FetchBlob('label')
	# _ = visualize.NCHW.ShowMultiple(blob)
	# pyplot.show()

	# inception = Inceptionv4(train_model, X, 0,)

	# inception.Inception_Stem(train_model)


if __name__ == '__main__':
	core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
	Train_Model()