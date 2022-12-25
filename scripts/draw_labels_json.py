#!/usr/bin/python
"""
draws boxes and corners on image based on json info
"""

import json
import cv2
import numpy as np
import sys

from zerdax2 import COLORS, CLASSES


def draw_boxes(canvas, pieces):
    print("Pieces:")
    thick = round(2 * (canvas.shape[0] / 1280))
    for piece in pieces:
        name = piece['piece']
        number = CLASSES[name]
        color = COLORS[number]
        left = piece['box'][0]
        top = piece['box'][1]
        right = left + piece['box'][2]
        bottom = top + piece['box'][3]
        cv2.rectangle(canvas, (left, top), (right, bottom),
                      color=color, thickness=thick)
        cv2.putText(canvas, name, (left-10, top-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        print(piece)
    return canvas


def draw_corners(canvas, corners):
    print("Board corners:")
    for i, corner in enumerate(corners):
        canvas = cv2.circle(canvas, corner,
                            radius=7, color=(0, 200, 170-i*40), thickness=-1)
        print(corner)
    return canvas


def draw_board_box(canvas, box):
    print("Board:")
    thick = round(3 * (canvas.shape[0] / 1280))
    color = (200, 100, 50)

    left = round(box[0] - box[1]/2)
    right = round(box[0] + box[1]/2)
    top = round(box[2] - box[3]/2)
    bottom = round(box[2] + box[3]/2)

    cv2.rectangle(canvas, (left, top), (right, bottom),
                  color=color, thickness=thick)

    print(box)
    return canvas


if __name__ == "__main__":
    files = sys.argv[1:]
    for imgname in files:
        print(f"drawing {imgname}...")
        basename = imgname.rsplit(".", 1)[0]

        img = cv2.imread(imgname)
        canvas = np.empty(img.shape, dtype='uint8') * 0

        data_file = open(f'{basename}.json')
        data = json.load(data_file)

        canvas = draw_boxes(canvas, data['pieces'])
        canvas = draw_corners(canvas, data['corners'])
        canvas = draw_board_box(canvas, data['board_box'])

        output = cv2.addWeighted(img, 1, canvas, 0.6, 1)
        cv2.imwrite(f"{basename}_json.png", output)

        data_file.close()
