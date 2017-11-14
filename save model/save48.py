import time
import random
import os
import tensorflow as tf
import numpy as np
import time
from skimage import io, transform, exposure, color
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
time.sleep(2)


np.set_printoptions(precision=2)
import argparse

from tensorflow.python.tools import freeze_graph

# from skimage import io, transform, exposure, color

parser = argparse.ArgumentParser()
parser.add_argument("--ngf", type=int, default=48, help="number of generator filters in first conv layer")
a = parser.parse_args()


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)



def batchnorm(input,channels):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

       # channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

#import tensorflow.contrib.layers as ly

#ly.conv2d_transpose
def deconv(batch_input, out_channels, cha):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [d for d in batch_input.get_shape()]
        s = tf.shape(batch_input)
        s *= tf.constant([1, 2, 2, 1])
        s -= tf.constant([0, 0, 0, cha])
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, s,
                                      [1, 2, 2, 1], padding="SAME")
        return conv

def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

  #  layer_specs = [
    #    a.ngf * 2,
   #     a.ngf * 4,
   #     a.ngf * 8,
 #      a.ngf * 8,
    #    a.ngf * 8,
    #    a.ngf * 8,
     #   a.ngf * 8,
 #   ]

    layer_specs = [
        a.ngf * 2,
        a.ngf * 4,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        a.ngf * 8,
        ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved, out_channels)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5, 0),
        (a.ngf * 8, 0.5, 384),
        (a.ngf * 8, 0.5, 384),
        (a.ngf * 8, 0.0, 384),
        (a.ngf * 4, 0.0, 576),
        (a.ngf * 2, 0.0, 288),
        (a.ngf, 0.0, 144),
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout, cha) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels,cha)

            output = batchnorm(output, out_channels)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels, 93)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]




def main():
    im_test = io.imread('/home/bo718.wang/irfan/pacling/train_2_out/images/u1025-inputs.png')
    im_test = transform.resize(im_test, [512, 256,3])
    in_dir = "/home/bo718.wang/irfan/pacling/train_2/"
    out_dir = "./out"
    tf.reset_default_graph()
    im = tf.placeholder(tf.float32, [None, None, 3])
    # im1 = tf.placeholder(tf.float32, [None, None, 3])
    # im1 = tf.placeholder(tf.float32, [None, None, 3])
    im1 = tf.expand_dims(im, 0)
    im1 *= 2
    im1 -= 1
    # im1 *= 256
    # im1 -= 128
    with tf.variable_scope('generator'):
        g_out = create_generator(im1, 3)
    # g_out *= 128
    g_out /= 2
    g_out += 0.5
    # out = g_out + 128.
    out = tf.identity(g_out, 'out')
    # print(out.name)
    # print('--------------------')
    # print("type of out",type(g_out))
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pretrained = tf.train.latest_checkpoint(in_dir)
        saver.restore(sess, pretrained)
        out = sess.run(out, feed_dict={im: im_test})
        io.imsave('out.jpg', out[0])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pretrained = tf.train.latest_checkpoint(in_dir)
        saver.restore(sess, pretrained)
        #   TF SAVER
        checkpoint_path = in_dir + 'model-881800'
         #  FREEZE GRAPH
        tf.train.write_graph(sess.graph_def, in_dir,
                             'model.pb')

        freeze_graph.freeze_graph(in_dir + '/model.pb', '', False,
                                  checkpoint_path, 'out',
                                  'save/restore_all', 'save/Const:0', out_dir + '/119_cartoon_size_sketch.pb', False, "")
        print('==================')


main()
