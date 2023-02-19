#!/usr/bin/python
"""
converts json to yolo plain txt label format
"""

import json
import sys
import cv2
import numpy as np

from zerdax2_misc import CLASSES


def json2yolo(filename):
    basename = filename.rsplit(".", 1)[0]
    jsonname = f'{basename}.json'
    imgname = f'{basename}.png'
    txtname = f'{basename}.txt'

    data_file = open(jsonname)
    data = json.load(data_file)
    txt = open(txtname, 'w')
    img = cv2.imread(imgname)

    width = img.shape[1]
    heigth = img.shape[0]

    corners = np.array(data['corners'], dtype='int32')
    x0 = np.min(corners[:, 0])
    x1 = np.max(corners[:, 0])
    y0 = np.min(corners[:, 1])
    y1 = np.max(corners[:, 1])

    x = ((x0 + x1)/2) / width
    y = ((y0 + y1)/2) / heigth
    dx = (x1 - x0) / width
    dy = (y1 - y0) / heigth
    print("0", x, y, dx, dy, file=txt)

    for piece in data['pieces']:
        name = piece['piece']
        left = piece['box'][0]
        top = piece['box'][1]
        dx = piece['box'][2]
        dy = piece['box'][3]
        right = left + dx
        bottom = top + dy

        x = ((left + right)/2) / width
        y = ((top + bottom)/2) / heigth
        dx = dx / width
        dy = dy / heigth
        print(CLASSES[name], x, y, dx, dy, file=txt)

    data_file.close()
    txt.close()
    return


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        json2yolo(filename)
