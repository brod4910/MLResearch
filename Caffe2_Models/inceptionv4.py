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
		self.layer_num = 1
		self.block_name = ''

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
			('%s_sp_batch_norm_%d', self.block_name, self.layer_num),
			filters,
			epsilon=1e-3,
			momentum=self.sp_batch_norm,
			is_test= self.test_flag,
			)
		return self.prev_blob

	def add_conv_layer(self, input_filters, output_filters, kernel, padding, stride= 1, prev_blob= None):
		if prev_blob == None:
			if pad == 'same':
				self.prev_blob = brew.conv(
					self.model,
					self.prev_blob,
					('%s_conv_%d', self.block_name, self.layer_num),
					input_filters,
					output_filters,
					kernel= kernel,
					stride= stride,
					no_bias=1,
					)
			else:
				self.prev_blob = brew.conv(
					self.model,
					self.prev_blob,
					('%s_conv_%d', self.block_name, self.layer_num),
					input_filters,
					output_filters,
					kernel= kernel,
					stride= stride,
					pad= 0,
					no_bias=1,
					)

		else:
			if pad == 'same':
				self.prev_blob = brew.conv(
					self.model,
					prev_blob,
					('%s_conv_%d', self.block_name, self.layer_num),
					input_filters,
					output_filters,
					kernel= kernel,
					stride= stride,
					no_bias=1,
					)
			else:
				self.prev_blob = brew.conv(
					self.model,
					self.prev_blob,
					('%s_conv_%d', self.block_name, self.layer_num),
					input_filters,
					output_filters,
					kernel= kernel,
					stride= stride,
					pad= 0,
					no_bias=1,
					)

		self.add_sp_batch_norm(output_filters)
		self.add_relu_activ()
		self.layer_num += 1

		return self.prev_blob

	def add_max_pool(self, prev_blob, kernel, stride, pad= 0):
		self.prev_blob = brew.max_pool(
			self.model,
			prev_blob,
			('%s_max_pool_%d', self.block_name, self.layer_num),
			kernel= kernel,
			stride= stride,
			pad= pad,
			)
		return prev_blob

	def add_avg_pool(self, prev_blob, kernel= 3,stride= 1,global_pool= False):
		# no stride or kernel. I will trust the Caffe ConvBaseOp class.
		self.prev_blob = brew.average_pool(
			self.model,
			prev_blob,
			kernel= kernel,
			stride= stride,
			('%s_avg_pool_%d', self.block_name, self.layer_num),
			global_pooling= global_pool,
			)

	def concat_layers(self, *args, axis=1):
		self.prev_blob, split_info = brew.concat(
			args,
			axis,
			)

		print(split_info)
		return self.prev_blob

	def Inception_Stem(self, model):
		self.block_name = 'stem'

		self.add_conv_layer(3, 32, 3, 'valid', stride= 2)
		self.add_conv_layer(32, 32, 3, 'valid')
		local_prev = self.add_conv_layer(32, 64, 3, 'same')

		# creating conv layer first so we can utilize the prev_blob
		self.add_conv_layer(64, 96, 3, 'valid', stride= 2)

		local_prev = self.add_max_pool(local_prev, 3, 2)

		concat1 = self.concat_layers(local_prev, self.prev_blob)

		# utilize prev_blob here
		self.add_conv_layer(160, 64, 1)
		local_prev = self.add_conv_layer(64, 96, 3, 'valid')

		self.add_conv_layer(160, 64, 1, prev_blob= concat1)
		self.add_conv_layer(64, 64, (1, 7))
		self.add_conv_layer(64, 64, (7, 1))
		self.add_conv_layer(64, 96, 3, 'valid')

		concat2 = self.concat_layers(local_prev, self.prev_blob)

		local_prev = self.add_max_pool(self.prev_blob, 1, 2)
		self.add_conv_layer(192, 192, 3, 'valid', prev_blob= concat2)

		self.concat_layers(local_prev, self.prev_blob)

		return self.prev_blob

	def Inception_A(model, input):
		self.layer_num = 1
		self.block_name = 'block_A'

		self.add_avg_pool(input)
		self.add_conv_layer()

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