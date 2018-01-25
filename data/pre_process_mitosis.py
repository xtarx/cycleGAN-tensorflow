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

rng = np.random.RandomState(0)
patch_size = (256, 256)
global_counter = 0
media_dir = "/media/data/tarek"


def crop_patches(input_image, output_dir):
    global global_counter

    print("in generate patch ", global_counter)
    patches = image.extract_patches_2d(input_image, patch_size, max_patches=200,
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
    # images = images[100:]
    for img in images:
        resized_images.append((tiff.imread(img)))

    for counter, img in enumerate(resized_images):

        if counter % 100 == 0:
            print('-----------Step %d:-------------' % counter)
        crop_patches(img, output_dir)


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


def move_to_fold(list,fold):
    srcA=media_dir + "/mitosis@20x/24jan2018/H/"
    dstA=media_dir + "/mitosis@20x/24jan2018/5fold/H/"
    # srcH=media_dir + "/mitosis@20x/24jan2018/H/"
    # dstH=media_dir + "/mitosis@20x/24jan2018/5fold/H/"

    for img in list:
        # print(srcA+img)
        # print(dstA+fold)
        copyfile(srcA+img, dstA+fold+img)
        # copyfile(srcH+img, dstH+fold+img)


def split_into_k_folds(datadir):
    files_shuffled = get_file_list_from_dir(datadir)
    # print(len(files_shuffled))
    fold_1=files_shuffled[:16960]
    fold_2=files_shuffled[16960:33920]
    fold_3=files_shuffled[33920:50880]
    fold_4=files_shuffled[50880:67840]
    fold_5=files_shuffled[67840:84800]
    move_to_fold(fold_1,"1/")
    move_to_fold(fold_2,"2/")
    move_to_fold(fold_3,"3/")
    move_to_fold(fold_4,"4/")
    move_to_fold(fold_5,"5/")

    return 0


# for counter, image_full in enumerate(images):
#     generate_patches(image_full)
if __name__ == '__main__':
    # dir = media_dir + "/mitosis@20x/TIFF_original_dataset/A-scanner/"
    # output_dir = media_dir + "/mitosis@20x/24jan2018/A/"
    split_into_k_folds(media_dir + "/mitosis@20x/24jan2018/H/")
