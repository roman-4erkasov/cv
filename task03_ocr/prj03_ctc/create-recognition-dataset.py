"""Create recognition crops from train.json (using car plates coordinates & texts)."""

import logging

import cv2
import numpy as np
import torch

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import json
from argparse import ArgumentParser

import cv2
import numpy as np
import tqdm

# from contest.common import get_box_points_in_correct_order, apply_normalization


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("data_dir", help="Path to dir containing 'train/', 'test/', 'train.json'.")
    return parser.parse_args()



def get_box_points_in_correct_order(box):
    """
    Permute the points in box in following order: Top left -> Top right -> Bottom right -> Bottom Left.
    :return: np.ndarray box shaped (4, 2)
    """
    box_sorted_by_x = sorted(box.tolist(), key=lambda x: x[0])
    if box_sorted_by_x[0][1] < box_sorted_by_x[1][1]:
        top_left = box_sorted_by_x[0]
        bottom_left = box_sorted_by_x[1]
    else:
        top_left = box_sorted_by_x[1]
        bottom_left = box_sorted_by_x[0]
    if box_sorted_by_x[2][1] < box_sorted_by_x[3][1]:
        top_right = box_sorted_by_x[2]
        bottom_right = box_sorted_by_x[3]
    else:
        top_right = box_sorted_by_x[3]
        bottom_right = box_sorted_by_x[2]
    return np.asarray([top_left, top_right, bottom_right, bottom_left])


def apply_normalization(image, keypoints, to_size=(320, 64)):
    """
    Apply perspective transform to crop such that keypoints are moved to corners of rectangle with fit_size.
    :return np.ndarray image of shape (to_size, 3) (result crop)
    """
    dest_points = np.asarray([[0., 0.],
                              [to_size[0], 0.],
                              [to_size[0], to_size[1]],
                              [0., to_size[1]]])
    h, _ = cv2.findHomography(keypoints, dest_points)
    crop_normalized = cv2.warpPerspective(image, h, dsize=to_size, flags=cv2.INTER_LANCZOS4)
    return crop_normalized




def get_crop(image, box):
    # TODO TIP: Maybe useful to crop using corners
    # See cv2.findHomography & cv2.warpPerspective for more
    box = get_box_points_in_correct_order(box)

    # TODO TIP: Maybe adding some margin could help.
    x_min = np.clip(min(box[:, 0]), 0, image.shape[1])
    x_max = np.clip(max(box[:, 0]), 0, image.shape[1])
    y_min = np.clip(min(box[:, 1]), 0, image.shape[0])
    y_max = np.clip(max(box[:, 1]), 0, image.shape[0])

    crop = image[y_min: y_max, x_min: x_max]

    return crop


def main(args):
    config_filename = os.path.join(args.data_dir, "train.json")
    with open(config_filename, "rt") as fp:
        config = json.load(fp)

    config_recognition = []
    for i, item in enumerate(tqdm.tqdm(config)):

        image_filename = item["file"]
        image = cv2.imread(os.path.join(args.data_dir, image_filename))
        if image is None:
            continue

        image_base, ext = os.path.splitext(image_filename)

        nums = item["nums"]
        for j, num in enumerate(nums):
            text = num["text"]
            box = np.asarray(num["box"])
            crop_filename = image_base + ".box." + str(j).zfill(2) + ext
            new_item = {"file": crop_filename, "text": text}
            if os.path.exists(crop_filename):
                config_recognition.append(new_item)
                continue

            crop = get_crop(image, box)
            cv2.imwrite(os.path.join(args.data_dir, crop_filename), crop)
            config_recognition.append(new_item)

    output_config_filename = os.path.join(args.data_dir, "train_recognition.json")
    with open(output_config_filename, "wt") as fp:
        json.dump(config_recognition, fp)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
