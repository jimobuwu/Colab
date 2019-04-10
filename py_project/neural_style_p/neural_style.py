import tensorflow as tf

# 开始的时候找不到。在pycharm里设置mark directory as sources root
import neural_input as ni

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('content_img', None, 'Filename of the content image (example: lion.jpg)')
tf.app.flags.DEFINE_string('content_img_dir', './image_input', 'Directory path to the content image')
tf.app.flags.DEFINE_string('style_img', None, 'Filename of the content image (example: lion.jpg)')
tf.app.flags.DEFINE_string('style_img_dir', './image_input', 'Directory path to the content image')
tf.app.flags.DEFINE_integer('max_size', 512, 'Maximum width or height of the input images')
tf.app.flags.DEFINE_string('image_format', 'png', 'image format')
tf.app.flags.DEFINE_string('model_weights', 'imagenet-vgg-verydeep-19.mat', "Weights and biases of the VGG-19 network.")


def render_single_image():
    # 原图
    content_img = ni.get_content_image(FLAGS.content_img_dir, FLAGS.content_img)
    # 风格图
    style_imgs = ni.get_style_image(FLAGS.style_img_dir, FLAGS.style_img)




def main(_):
    pass


if __name__ == '__main__':
    tf.app.run()
