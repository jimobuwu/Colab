# a = [1,2,3,4,5,6]
# b = a[-1:-7:-1]
# # c = a[::-1]
# # d = [[1,2,3],[1,2,3]]
# # c = d[..., ::-1]
# print(a[..., None])
# print(b)
# # print(c)

import numpy as np
import tensorflow as tf
import os
import cv2

a = np.array([[1,2,3,4],[1,2,3,4]])
print(a[..., ::-1])

tf.enable_eager_execution()

# img_dir = '.'
img_path = "1035.png"
img_content = tf.io.read_file('1035.png')
img = tf.image.decode_png(img_content, channels=3)
print(img.shape)
img = tf.image.resize_images(img, size=(100, 100))
h,w,d = img.shape
imga = np.copy(img)
print(img.shape)

# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img = img.astype(np.float32)
# print(img.shape)

def test(a,b):
    if b:
        print("has b")

test(1)