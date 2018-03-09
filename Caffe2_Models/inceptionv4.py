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
	def __init__(self, model, prev_blob, test_flag, sp_batch_norm=.99):
		self.model = model
		self.prev_blob = prev_blob
		self.test_flag = test_flag
		self.sp_batch_norm = sp_batch_norm

	def add_image_input(self, model, reader, batch_size, image_size, dtype):
		data, label, additional_outputs = brew.image_input(
			model,
			reader, ["data", "label"],
			batch_size= batch_size,
			output_type= dtype,
			use_gpu_transform=True if model._device_type == 1 else False,
			use_caffe_datum= True,
			# mean= 128,
			# std= 128,
			color= 3,
			crop= image_size,
			is_test= self.test_flag, 
			)

	def add_conv_layer(self, input_dims, output_dims, name, kernel, stride= 1, pad= 0, prev_blob= None):
		if prev_blob == None:
			self.prev_blob = brew.conv(
				self.model,
				self.prev_blob,
				name,
				input_dims,
				output_dims,
				kernel=kernel,
				stride=stride,
				pad=pad,
				no_bias=1,
				)
		else:
			self.prev_blob = brew.conv(
				self.model,
				prev_blob,
				name,
				input_dims,
				output_dims,
				kernel=kernel,
				stride=stride,
				pad=pad,
				no_bias=1,
				)

		return self.prev_blob

	def add_relu_activ(self):
		self.prev_blob = brew.relu(
			self.model,
			self.prev_blob,
			self.prev_blob,
			)
		return self.prev_blob

	def add_max_pool(self, prev_blob, name, kernel, stride):
		self.prev_blob = brew.max_pool(
			self.model,
			prev_blob,
			name,
			kernel= kernel,
			stride= stride,
			)
		return prev_blob

	def concat_layers(self, *args, axis=1):
		self.prev_blob, split_info = brew.concat(
			args,
			axis,
			)

		print(split_info)
		return self.prev_blob

	def add_sp_batch_norm(self, filters, name):
		self.prev_blob = brew.spatial_bn(
			self.model,
			self.prev_blob,
			name,
			filters,
			epsilon=1e-3,
			momentum=self.sp_batch_norm,
			is_test= self.test_flag,
			)
		return self.prev_blob

	def Inception_Stem(self, model):

		self.add_conv_layer(1, 32, 'conv1_stem', 3, stride= 2)
		self.add_conv_layer(32, 32, 'conv2_stem', 3)
		self.add_conv_layer(32, 64, 'conv3_stem', 3)

		# creating conv layer first so we can utilize the prev_blob
		self.add_conv_layer(64, 96, 'conv4_stem', 3, stride= 2)

		self.add_max_pool('conv3_stem', 'pool1_stem', 3, 2)

		concat1 = self.concat_layers('pool1_stem', 'conv4_stem')

		# can possibly make one layer as opposed to two

		self.add_conv_layer(160, 64, 'conv5_stem', 1)
		self.add_conv_layer(64, 96, 'conv6_stem', 3)

		self.add_conv_layer(160, 64, 'conv7_stem', 1, prev_blob= 'pool1_stem')
		self.add_conv_layer(64, 64, 'conv8_stem', (7, 1))
		self.add_conv_layer(64, 64, 'conv9_stem', (1, 7))
		self.add_conv_layer(64, 96, 'conv10_stem', 3)

		concat2 = self.concat_layers('conv6_stem', 'conv10_stem')

		self.add_max_pool('pool2_stem', self.prev_blob, 1, 2)
		self.add_conv_layer(192, 192, 'conv11_stem', 3, prev_blob= concat2)

		self.concat_layers('pool2_stem', 'conv11_stem')

	# def Inception_A(model):


	# def Inception_B(model):


	# def Inception_C(model):


	# def Reduction_A(model):


	# def Reduction_B(model):


# def create_Inception(model):

if __name__ == '__main__':
	core.GlobalInit(['caffe2', '--caffe2_log_level=2'])
	arg_scope = {"order" : "NCHW"}

	X = np.random.rand(299, 299, 3).astype(np.float32)

	train_model = model_helper.ModelHelper(name="inceptionv4_train", arg_scope=arg_scope)

	inception = Inceptionv4(train_model, X, 0,)

	inception.Inception_Stem(train_model)

	print(workspace.Blobs())