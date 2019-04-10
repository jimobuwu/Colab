import tensorflow as tf
import os
import numpy as np


def preprocess(img):
    imgpre = np.copy(img)
    imgpre = imgpre[..., ::-1]
    imgpre = imgpre[np.newaxis, :, :, :]
    # subtract mean value
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return imgpre


def get_content_image(img_dir, img_path, max_size):
    path = os.path.join(img_dir, img_path)
    img_content = tf.io.read_file(path)
    img = tf.image.decode_png(img_content, channels=3)
    h, w, d = img.shape
    mx = max_size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = tf.image.resize_images(img, size=(int(w), mx), method=tf.image.ResizeMethod.AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = tf.image.resize_images(img, size=(mx, int(h)), method=tf.image.ResizeMethod.AREA)
    img = preprocess(img)
    return img


def get_style_image(content_img, style_img_dir, style_img_paths):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in style_img_paths:
        path = os.path.join(style_img_dir, style_fn)
        img_content = tf.io.read_file(path)
        img = tf.image.decode_png(img_content, channels=3)
        img = tf.image.resize_images(img, size=(cw, ch), method=tf.image.ResizeMethod.AREA)
        img = preprocess(img)
        style_imgs.append(img)
    return style_imgs

