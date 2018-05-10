# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   @Author: Kong Haiyang
   @Date: 2018-05-10 17:15:07
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

MEAN = [123.68, 116.779, 103.939]  # RGB
csv_file = 'train.csv'
batch_size = 1


def gen_image_label_old(csv_file, image_size=224, skip_header_lines=None):
  '''Processing csv as original string.
  '''
  csv_queue = tf.train.string_input_producer([csv_file])
  reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

  _, value = reader.read(csv_queue)
  value_raw = tf.reshape(value, [1])
  split_values = tf.string_split(value_raw, delimiter=',')
  filename = split_values.values[0]
  label = split_values.values[1]

  image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
  image = tf.cast(image, tf.float32)

  image = tf.image.resize_images(image, (image_size, image_size))

  return image, filename, label


def gen_image_label(csv_file, image_size=224, skip_header_lines=None):
  csv_queue = tf.train.string_input_producer([csv_file])
  reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

  _, value = reader.read(csv_queue)
  filename, label = tf.decode_csv(value, [["path"], ["label"]])

  image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
  image = tf.cast(image, tf.float32)

  initial_height, initial_width = tf.shape(image)[0], tf.shape(image)[1]

  def width_greater_than_height(initial_height, initial_width):
    ratio = tf.to_float(initial_width)/tf.constant(image_size, tf.float32)
    new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
    new_height = tf.to_int32(tf.to_float(initial_height) / ratio)
    offset_width = 0
    offset_height = tf.to_int32((image_size - new_height)/2)
    return new_height, new_width, offset_height, offset_width

  def height_greater_than_width(initial_height, initial_width):
    ratio = tf.to_float(initial_height)/tf.constant(image_size, tf.float32)
    new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
    new_height = tf.to_int32(tf.to_float(initial_height) / ratio)
    offset_width = tf.to_int32((image_size - new_width)/2)
    offset_height = 0
    return new_height, new_width, offset_height, offset_width

  new_height, new_width, offset_height, offset_width = \
      tf.cond(tf.greater_equal(initial_height, initial_width),
              lambda: height_greater_than_width(initial_height, initial_width),
              lambda: width_greater_than_height(initial_height, initial_width))

  image = tf.image.resize_images(image, [new_height, new_width])
  image_slices = []
  for i in range(3):
    img = image[..., i, tf.newaxis]
    img = tf.image.pad_to_bounding_box(img, offset_height, offset_width,
                                       image_size, image_size, constant_values=MEAN[i])
    image_slices.append(img)

  image = tf.concat(image_slices, 2)

  return image, filename, label


def gen_images_labels_patch(csv_file, shuffle=False, batch_size=16, num_preprocess_threads=4):
  image, filename, label = gen_image_label(csv_file)

  min_after_dequeue = num_preprocess_threads*batch_size

  if shuffle:
    ims, fns, lbls = tf.train.shuffle_batch(
        [image, filename, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_after_dequeue + 3*batch_size,
        min_after_dequeue=min_after_dequeue,
        allow_smaller_final_batch=True
    )
  else:
    ims, fns, lbls = tf.train.batch(
        [image, filename, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_after_dequeue + 3*batch_size,
        allow_smaller_final_batch=True
    )

  return ims, fns, lbls


ims, fns, lbls = gen_images_labels_patch(csv_file, shuffle=True, batch_size=batch_size)

import cv2

with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  try:
    while not coord.should_stop():
      for _ in range(7):
        im, fn, lbl = sess.run([ims, fns, lbls])
        for bs in range(batch_size):
          print(fn[bs])
          print(lbl[bs])
          cv2.imwrite('test.jpg', im[bs][..., ::-1])
      coord.request_stop()
      # for step in range(total_steps):
      #   train_data, train_label = sess.run([im, lbl])
      #   feed_dict = {data_node: train_data, labels_node: train_label}
      #   _, l, lr = sess.run([optimizer, loss, learning_rate], feed_dict=feed_dict)
      # coord.request_stop()
  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    pass
  coord.join(threads)
