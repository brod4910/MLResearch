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
	    data = model.StopGradient(data, data)

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
			is_test= self.test_flag,
			)
		return self.prev_blob

	def add_conv_layer(self, input_filters, output_filters, kernel, padding, stride= 1, prev_blob= None):
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
					no_bias=1,
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
					no_bias=1,
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
					no_bias=1,
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
			'%s_max_pool_%d' % (self.block_name, self.layer_num),
			kernel= kernel,
			stride= stride,
			pad= pad,
			)
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
		return self.prev_blob

	def concat_layers(self, *args, axis=1):
		self.prev_blob, split_info = brew.concat(
			args,
			axis,
			)

		print(split_info)
		return self.prev_blob

	def Inception_Stem(self, model):
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
		self.add_conv_layer(64, 64, (1, 7), 'same')
		self.add_conv_layer(64, 64, (7, 1), 'same')
		self.add_conv_layer(64, 96, 3, 'valid')

		concat2 = self.concat_layers(local_prev, self.prev_blob)

		local_prev = self.add_max_pool(self.prev_blob, 1, 2)
		self.add_conv_layer(192, 192, 3, 'valid', prev_blob= concat2)

		self.concat_layers(local_prev, self.prev_blob)

		return self.prev_blob

	def Inception_A(self, input):

		self.add_avg_pool(input)
		layer_1 = self.add_conv_layer(384, 96, 1, 'same')

		layer_2 = self.add_conv_layer(384, 96, 1, 'same', prev_blob= input)

		self.add_conv_layer(384, 64, 1, 'same', prev_blob = input)
		layer_3 = self.add_conv_layer(64, 96, 3, 'same')

		self.add_conv_layer(384, 64, 1, 'same', prev_blob= input)
		self.add_conv_layer(64, 96, 3, 'same')
		layer_4 = self.add_conv_layer(64, 96, 3, 'same')

		return self.concat_layers(layer_1, layer_2, layer_3, layer_4)

	def Inception_B(self, input):

		self.add_avg_pool(input)
		layer_1 = self.add_conv_layer(1024, 128, 1, 'same')

		layer_2 = self.add_conv_layer(1024, 384, 1, 'same', prev_blob= input)

		self.add_conv_layer(1024, 192, 1, 'same', prev_blob= input)
		self.add_conv_layer(192, 224, (7, 1), 'same')
		layer_3 = self.add_conv_layer(224, 256, (1, 7), 'same')

		self.add_conv_layer(1024, 192, 1, 'same', prev_blob= input)
		self.add_conv_layer(192, 192, (7, 1), 'same')
		self.add_conv_layer(192, 224, (1, 7), 'same')
		self.add_conv_layer(224, 224, (7, 1), 'same')
		layer_4 = self.add_conv_layer(224, 256, (1, 7), 'same')

		return self.concat_layers(layer_1, layer_2, layer_3, layer_4)

	def Inception_C(self, input):

		self.add_avg_pool(input)
		layer_1 = self.add_conv_layer(1536, 256, 1, 'same')

		layer_2 = self.add_conv_layer(1536, 256, 1, 'same', prev_blob= input)

		sub_layer_1 = self.add_conv_layer(1536, 384, 1, 'same', prev_blob= input)
		layer_3 = self.add_conv_layer(384, 256, (3, 1), 'same', prev_blob= sub_layer_1)
		layer_4 = self.add_conv_layer(384, 256. (1, 3), 'same', prev_blob= sub_layer_1)

		self.add_conv_layer(1536, 384, 1, 'same', prev_blob= input)
		self.add_conv_layer(384, 448, (3, 1), 'same')
		sub_layer_2 = self.add_conv_layer(448, 512, (1, 3), 'same')
		layer_5 = self.add_conv_layer(512, 256, (3, 1), 'same', prev_blob= sub_layer_2)
		layer_6 = self.add_conv_layer(512, 256, (1, 3), 'same', prev_blob= sub_layer_2)

		return self.concat_layers(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6)

	def Reduction_A(self, input):
		layer_1 = self.add_max_pool(input, 3, 2)

		layer_2 = self.add_conv_layer(384, 384, 3, 'valid', stride= 2, prev_blob= input)

		self.add_conv_layer(384, 192, 1, 'same', prev_blob= input)
		self.add_conv_layer(192, 224, 3, 'same',)
		layer_3 = self.add_conv_layer(224, 256, 3, 'valid', stride= 2)

		return self.concat_layers(layer_1, layer_2, layer_3)

	def Reduction_B(self, input):
		layer_1 = self.add_max_pool(input, 3, 2)

		self.add_conv_layer(1024, 192, 1, 'same', prev_blob= input)
		layer_2 = self.add_conv_layer(192, 192, 3, 'valid', stride= 2)

		self.add_conv_layer(1024, 256, 1, 'same', prev_blob= input)
		self.add_conv_layer(256, 256, (1, 7), 'same')
		self.add_conv_layer(256, 320, (7, 1), 'same')
		layer_3 = self.add_conv_layer(320, 320, 3, 'valid', stride= 2)

		return self.concat_layers(layer_1, layer_2, layer_3)

def create_Inceptionv4()
