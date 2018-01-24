"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils
import matplotlib.image as mp

media_dir = "/media/data/tarek/"

FLAGS = tf.flags.FLAGS

# tf.flags.DEFINE_string('model', 'pretrained/scannerA2scannerH.pb', 'model path (.pb)')

# tf.flags.DEFINE_string('input', media_dir + "mitosis@20x/evaluation/40_img_test_dataset/mitosis_scannerA/38.png",
#                        'input image path (.jpg)')
# tf.flags.DEFINE_string('output',
#                        media_dir + "mitosis@20x/evaluation/40_img_test_dataset/mitosis_scannerH/tinyexp/38_mod.png",
#                        'output image path (.jpg)')

tf.flags.DEFINE_string('input_dir', "/media/data/tarek/camelyon16/eval/patches_Tumor_110.tif/",
                       'input image path (.png)')
tf.flags.DEFINE_string('output_dir', "/media/data/tarek/camelyon16/eval/stained_patches_Tumor_110/",
                       'output image path (.png)')
# tf.flags.DEFINE_string('output_dir',"/media/data/tarek/mitosis@20x/evaluation/80_img_top_left_crop/", 'output image path (.png)')
# tf.flags.DEFINE_string('output_dir',"/media/data/tarek/mitosis@20x/evaluation/80_img_top_left_crop/", 'output image path (.png)')
# tf.flags.DEFINE_string('output_dir',"/media/data/tarek/mitosis@20x/evaluation/80_img_top_left_crop/", 'output image path (.png)')

tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

# this checkpoint 20171121-0129 was trained for nealy 35K epoch and with wiht  10*cycleloss
# tf.flags.DEFINE_string('checkpoint_dir', '/media/data/tarek/checkpoints/split20171128-1036',
#                        'checkpoints directory path')

tf.flags.DEFINE_string('checkpoint_dir', '/media/data/tarek/checkpoints/camelyon16_center2_to_center320180109-2324',
                       'checkpoints directory path')


def inference():
    graph = tf.Graph()

    with graph.as_default():
        with tf.gfile.FastGFile(FLAGS.input, 'rb') as f:
            image_data = f.read()
            input_image = tf.image.decode_jpeg(image_data, channels=3)
            input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
            input_image = utils.convert2float(input_image)
            input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

        with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
        [output_image] = tf.import_graph_def(graph_def,
                                             input_map={'input_image': input_image},
                                             return_elements=['output_image:0'],
                                             name='output')
    # print("output image ", output_image)

    with tf.Session(graph=graph) as sess:
        generated = output_image.eval()
        # print("output image eval ", generated)

        with open(FLAGS.output, 'wb') as f:
            f.write(generated)


def get_result(XtoY=True):
    graph = tf.Graph()
    try:
        os.mkdir(FLAGS.output_dir)
    except:
        print('dir already exist!')
    with tf.Session(graph=graph) as sess:
        cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)
        input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
        cycle_gan.model()

        if XtoY:
            output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
        else:
            output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, latest_ckpt)
        # print("listing dir ",os.listdir(FLAGS.input_dir))
        for index, fname in enumerate(os.listdir(FLAGS.input_dir)):
            img = mp.imread(FLAGS.input_dir + fname)
            feed = {input_image: img}
            gen_img = sess.run(output_image, feed_dict=feed)
            image_dir = FLAGS.output_dir + fname
            # print("output file name", image_dir)
            # print(gen_img)

            with open(image_dir, 'wb') as f:
                f.write(gen_img)

            # mp.imsave(image_dir, gen_img)
            if index % 25 == 0:
                print(index)


def main(unused_argv):
    # inference()
    get_result()


if __name__ == '__main__':
    tf.app.run()
