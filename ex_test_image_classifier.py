from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import cv2
from PIL import Image


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'test_list', '', 'Test image list.')
tf.app.flags.DEFINE_string(
    'test_dir', '.', 'Test image directory.')
tf.app.flags.DEFINE_string(
    'output_pred_file', 'output_pred.txt', 'Output Testing result.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'Batch size.')
tf.app.flags.DEFINE_integer(
    'num_classes', 128, 'Number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

def central_crop_image(image_path, height, width, central_fraction=0.875):
    image = cv2.imread(image_path, 1) # B G R 
    h, w, _ = image.shape
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    if central_fraction:
        central_y = h /2
        central_x = w /2
        crop_h = h * central_fraction
        crop_w = w * central_fraction
        miny = int(central_y - crop_h / 2)
        minx = int(central_x - crop_w / 2)
        maxy = int(central_y + crop_h / 2)
        maxx = int(central_x + crop_w / 2)
        image = image[miny:maxy, minx:maxx, :]
        
    if height and width:
        image = cv2.resize(image, [height, width])

    image = image - 0.5
    image = image * 2.0
    return image     
   

def main(_):
    if not FLAGS.test_list:
        raise ValueError('You must supply the test list with --test_list')
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()
        
    with tf.Graph().as_default() as g:
        tf_global_step = slim.get_or_create_global_step()
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        # using model_name to set the preprocessing op
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        
        test_image_size = FLAGS.test_image_size or network_fn.default_image_size

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        batch_size = FLAGS.batch_size
        
        img_placeholder = tf.placeholder(tf.uint8, [None, None, 3])
        processedimage = image_preprocessing_fn(img_placeholder, test_image_size, test_image_size)
    
        
        tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
        # get score

        logits, _ = network_fn(tensor_input)

        probs = tf.nn.softmax(logits)
        # logits = tf.nn.top_k(logits, 5)
        
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        test_ids = [line.strip() for line in open(FLAGS.test_list)]
        tot = len(test_ids)
        results = list()
        test_image_h = test_image_size
        test_image_w = test_image_size

        #result_dict = {}
        count = 0
        with tf.Session(config=config, graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

            fb = open(FLAGS.output_pred_file, 'w')
            time_start = time.time()
            for test_id in test_ids:
                images = [] # list()
                test_path = os.path.join(FLAGS.test_dir, test_id)
                #img = cv2.imread(test_path, 1)
                img = np.array(Image.open(test_path), dtype=np.uint8)
                #h, w, _ = image.shape
                #img = cv2.resize(img, [600, 600])
                #image = open(test_path, 'rb').read()
                #image = tf.image.decode_jpeg(image, channels=3)
                # processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)

                processed_image = sess.run(processedimage, feed_dict={img_placeholder:img}) #, test_image_size:test_image_h, test_image_size:test_image_w})
                #processed_image = sess.run(processedimage, feed_dict={img_placeholder:img}, test_image_size, test_image_size)

                images.append(processed_image)
                images = np.array(images) 
                
                predictions = sess.run(probs, feed_dict = {tensor_input : images}) # .indices
                probility = predictions[0].tolist()
                #result_dict[test_id] = probilit
                fb.write(test_id + ' '+ ' '.join(str(pred_cls) for pred_cls in probility) + '\n')
                count += 1
                if count%100==0:
                    time_count = time.time()-time_start
                    print ('{} images predicted. Escaped Time : {}s'.format(count, time_count))
            fb.close() 

'''
            for idx in range(0, tot, batch_size):
                images = list()
                idx_end = min(tot, idx + batch_size)
                # print(idx)
                for i in range(idx, idx_end):
                    image_id = test_ids[i]
                    test_path = os.path.join(FLAGS.test_dir, image_id)
                    image = open(test_path, 'rb').read()
                    image = tf.image.decode_jpeg(image, channels=3)
                    processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
                    processed_image = sess.run(processed_image)
                    images.append(processed_image)
                images = np.array(images)
                # predictions = sess.run(logits, feed_dict = {tensor_input : images}) # .indices
                predictions = sess.run(probs, feed_dict = {tensor_input : images}) # .indices
                for i in range(idx, idx_end):
                    # print('{} {}'.format(image_id, predictions[i - idx].tolist()))
                    result_dict[image_id]=predictions[i - idx].tolist()


            time_total=time.time()-time_start
            print('total time: {}, total images: {}, average time: {}'.format(
                time_total, len(test_ids), time_total / len(test_ids)))
            with open(FLAGS.output_pred_file, 'w') as fb:
                for key in result_dict:
                    fb.write(key + ' '+ ' '.join(str(pred_cls) for pred_cls in result_dict[key])+ '\n')
'''


if __name__ == '__main__':
    tf.app.run()
