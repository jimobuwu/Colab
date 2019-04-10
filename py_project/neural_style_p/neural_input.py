import tensorflow as tf
import cv2
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
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img返回的是个np.array？
    img = img.astype(np.float32)
    h,w,d = img.shape
    mx = max_size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        # resize,图像缩放，不截断
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img


def get_style_image(content_img, style_img_dir, style_img_paths):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in style_img_paths:
        path = os.path.join(style_img_dir, style_fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interploation=cv2.INTER_AREA)
        img = preprocess(img)
        style_imgs.append(img)
    return style_imgs



