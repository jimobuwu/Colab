import tensorflow as tf
import numpy as np
import scipy.io

vgg_layers = []

def get_weights(vgg_layers, i):
  weights = vgg_layers[i][0][0][2][0][0]
  W = tf.constant(weights)
  return W


def get_bias(vgg_layers, i):
  bias = vgg_layers[i][0][0][2][0][1]
  # (1, out_channels) -> (out_channels,1)
  b = tf.constant(np.reshape(bias, bias.size))
  return b

def conv_layer(layer_name, layer_input, W):
    conv = tf.nn.conv2d(layer_input, W, strides=[1,1,1,1], padding='SAME')
    

def build_model(input_img, model_weights_file):
    _, h, w, d = input_img.shape
    intput = np.zeros((1, h, w, d))

    global vgg_layers
    vgg_rawnet = scipy.io.loadmat(model_weights_file)
    vgg_layers = vgg_rawnet['layers'][0]
