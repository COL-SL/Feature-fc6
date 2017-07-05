import os
import os.path
import glob
import cv2
import caffe
import lmdb
import glob
import numpy as np
import sys
import time
from caffe.proto import caffe_pb2
from itertools import *

caffe.set_device(0)
caffe.set_mode_gpu()

#caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
#sys.path.insert(0, caffe_root + 'python')

JUMP_FRAMES = 0
model_def = '/media/aceballos/alberto.ceballos/code/caffe/models/bvlc_alexnet/deploy.prototxt'
model_weights = '/media/aceballos/alberto.ceballos/code/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

path = sys.argv[1]
normal_or_abnormal = sys.argv[2]
discard_frames = 0
total_frames = 0

date_actual = time.strftime("%c")
date_actual = date_actual.replace(' ', '_')
date_actual = date_actual.replace(':', '_')

if normal_or_abnormal == 'N':
    file_exit = "/media/aceballos/alberto.ceballos/code/caffe/examples/feature_fc6/" + date_actual + "_" + "data_feature_fc6_normal.txt"
else:
    file_exit = "/media/aceballos/alberto.ceballos/code/caffe/examples/feature_fc6/" + date_actual + "_" + "data_feature_fc6_abnormal.txt"

os.chdir(path)
list = glob.glob("*.jpg")
ordered_list = sorted(list)
#ordered_list  = list.sort()

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/media/aceballos/alberto.ceballos/code/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# print '\n','mean-subtracted values:', zip('BGR', mu),'\n'


# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,  # batch size
                          3,  # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

print '\n'

initial_time = time.time()

for frame in ordered_list:

    if discard_frames < JUMP_FRAMES:
        discard_frames = discard_frames + 1
    else:
        print frame
        image = caffe.io.load_image(frame)
        transformed_image = transformer.preprocess('data', image)

        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        ### perform classification
        output = net.forward()

        # output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

        # print 'predicted class is:', output_prob.argmax()

        # labels_file = '/media/aceballos/alberto.ceballos/code/caffe/data/ilsvrc12/synset_words.txt'
        # labels = np.loadtxt(labels_file, str, delimiter='\t')

        # print 'output label:', labels[output_prob.argmax()]

        # sort top five predictions from softmax output
        # top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

        # print 'probabilities and labels:'
        # result = zip(output_prob[top_inds], labels[top_inds])

        # print result, '\n'

        feature_6 = net.blobs['fc6'].data[0]

        with open(file_exit, 'a') as file:
            np.savetxt(file, feature_6, fmt='%.4f', delimiter='\n')

        discard_frames = 0
        total_frames = total_frames + 1

final_time = time.time()
total_time = final_time - initial_time
frames_per_second = total_frames / total_time

print '\n\n',"Execution Time: %f seconds" % (total_time)
print "Number of frames saved:", total_frames
print "Frames per second:", frames_per_second
print "Save in:", file_exit, '\n'
