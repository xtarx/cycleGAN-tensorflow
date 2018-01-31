from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from sklearn.feature_extraction import image
import cv2
from glob import glob
import tifffile as tiff
from shutil import copyfile
from PIL import Image

rng = np.random.RandomState(0)
patch_size = (256, 256)
global_counter = 0
media_dir = "/media/data/tarek"


# Function for obtaining center crops from an image
def crop_center(x, crop_w, crop_h):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    print(h, w)
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    print(j, i)
    return x[j:j + crop_h, i:i + crop_w]


def crop_patches(input_image, output_dir):
    global global_counter

    print("in generate patch ", global_counter)
    patches = image.extract_patches_2d(input_image, patch_size, max_patches=50,
                                       random_state=rng)
    for counter, i in enumerate(patches):

        if np.any(i):
            # print("saving image")
            cv2.imwrite(output_dir + str(global_counter) + '.png', cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
            global_counter += 1


def generate_patches(dir, output_dir):
    pattern = "*.tiff"
    images = []
    resized_images = []
    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))

    print(len(images))
    images.sort()
    images = images[100:]
    print("First Index is ", images[0])
    for img in images:

        resized_images.append((tiff.imread(img)))

    for counter, img in enumerate(resized_images):

        if counter % 100 == 0:
            print('-----------Step %d:-------------' % counter)
        crop_patches(img, output_dir)


def tiff_to_png(dir, output_dir):
    pattern = "*.tiff"
    images = []
    resized_images = []
    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))

    # print(len(images))
    images.sort()
    for img in images:
        resized_images.append((tiff.imread(img)))
        # # soora = (cv2.resize(soora, (1539, 1376), interpolation=cv2.INTER_AREA))
        # resized_images.append(soora)

    for counter, img in enumerate(images):
        img_name = (img.split('/')[-1][:-5])
        img_file = (tiff.imread(img))
        # exit()
        if counter % 100 == 0:
            print('-----------Step %d:-------------' % counter)
        cv2.imwrite(output_dir + img_name + '.png', cv2.cvtColor(img_file, cv2.COLOR_RGB2BGR))


def resize_png_folder(dir, output_dir):
    pattern = "*.tiff"
    images = []
    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))

    print(len(images))
    # images=images[0:3]
    for counter, img in enumerate(images):
        img_name = (img.split('/')[-1][:-5])
        img_file = (tiff.imread(img))
        # exit()
        if counter % 100 == 0:
            print('-----------Step %d:-------------' % counter)
        img_file = (cv2.resize(img_file, (1539, 1376), interpolation=cv2.INTER_AREA))
        cv2.imwrite(output_dir + img_name + '.png', cv2.cvtColor(img_file, cv2.COLOR_RGB2BGR))


# split into folds

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    shuffle(data_files)
    return data_files


from random import shuffle
from math import floor


def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing


def move_to_fold(list, fold):
    srcA = media_dir + "/mitosis@20x/24jan2018/H/"
    dstA = media_dir + "/mitosis@20x/24jan2018/5fold/H/"
    # srcH=media_dir + "/mitosis@20x/24jan2018/H/"
    # dstH=media_dir + "/mitosis@20x/24jan2018/5fold/H/"

    for img in list:
        # print(srcA+img)
        # print(dstA+fold)
        copyfile(srcA + img, dstA + fold + img)
        # copyfile(srcH+img, dstH+fold+img)


def split_into_k_folds(datadir):
    files_shuffled = get_file_list_from_dir(datadir)
    # print(len(files_shuffled))
    fold_1 = files_shuffled[:16960]
    fold_2 = files_shuffled[16960:33920]
    fold_3 = files_shuffled[33920:50880]
    fold_4 = files_shuffled[50880:67840]
    fold_5 = files_shuffled[67840:84800]
    move_to_fold(fold_1, "1/")
    move_to_fold(fold_2, "2/")
    move_to_fold(fold_3, "3/")
    move_to_fold(fold_4, "4/")
    move_to_fold(fold_5, "5/")

    return 0


def crop_2(src_img, h, w, output_dir, counter):
    im = Image.open(src_img)
    im_w, im_h = im.size
    print('Image width:%d height:%d  will split into (%d %d) ' % (im_w, im_h, w, h))
    w_num, h_num = int(im_w / w), int(im_h / h)

    for wi in range(0, w_num):
        for hi in range(0, h_num):
            box = (wi * w, hi * h, (wi + 1) * w, (hi + 1) * h)
            piece = im.crop(box)
            piece.save(output_dir + str(counter) + "_" + str(hi) + '.png')


def generate_Scanner_eval_images(dir, output_dir):
    pattern = "*.png"
    images = []
    # read directory of images
    for _, _, _ in os.walk(dir):
        images.extend(glob(os.path.join(dir, pattern)))

    print(len(images))
    images.sort()
    # images = images[0:2]

    for counter, img in enumerate(images):
        # print("Image path is ", output_dir, img)
        crop_2(img, 256, 256, output_dir, counter)


# for counter, image_full in enumerate(images):
#     generate_patches(image_full)
if __name__ == '__main__':
    # dir = media_dir + "/mitosis@20x/24jan2018/png/H/"

    dir = media_dir + "/mitosis@20x/TIFF_original_dataset/A-scanner/"
    output_dir = media_dir + "/mitosis@20x/30jan2018/A/"
    generate_patches(dir,output_dir);


    # dir = media_dir + "/mitosis@20x/24jan2018/png_resized/A/"
    # output_dir = media_dir + "/mitosis@20x/24jan2018/eval/A/"
    # generate_Scanner_eval_images(dir, output_dir)
    #
    # dir = media_dir + "/mitosis@20x/24jan2018/png_resized/H/"
    # output_dir = media_dir + "/mitosis@20x/24jan2018/eval/H/"
    # generate_Scanner_eval_images(dir, output_dir)

    # resize_png_folder(dir, output_dir)
    # split_into_k_folds(media_dir + "/mitosis@20x/24jan2018/H/")
