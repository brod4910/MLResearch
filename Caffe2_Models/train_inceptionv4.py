from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import time
import os

from caffe2.python import core, workspace, experiment_util, data_parallel_model
from caffe2.python import data_parallel_model_utils, dyndep, optimizer
from caffe2.python import timeout_guard, model_helper, brew
from caffe2.proto import caffe2_pb2

import caffe2.python.models.resnet as resnet
from caffe2.python.modeling.initializers import Initializer, PseudoFP16Initializer
import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.python.predictor_constants import predictor_constants as predictor_constants

def Train_Model():

	train_model = model_helper.ModelHelper(name="inceptionv4_train", arg_scope=arg_scope)

	inception = Inceptionv4(train_model, X, 0,)

	inception.Inception_Stem(train_model)


if __name__ == '__main__':
	core.GlobalInit(['caffe2', '--caffe2_log_level=2'])
	arg_scope = {"order" : "NCHW"}
	
	Train_Model()