#!/usr/bin/python
"""
draws labels and boxes around chess pieces based on yolo txt
"""

import cv2
import numpy as np
import sys


def draw_corners(canvas, corners):
    for i, corner in enumerate(corners):
        canvas = cv2.circle(canvas, corner,
                            radius=10, color=(200, 0, 100-i*10), thickness=-1)
        print(corner)
    return canvas


if __name__ == "__main__":
    for img_name in sys.argv[1:]:
        basename = img_name.rsplit(".", 1)[0]
        out_name = f"{basename}_json.png"

        img = cv2.imread(img_name)
        width = img.shape[1]
        heigth = img.shape[0]

        corners = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        corners = corners.reshape(4, 2)
        for c in corners:
            c[0] *= width
            c[1] *= heigth

        corners = (np.rint(corners)).astype(int)
        print(corners)

        canvas = np.zeros(img.shape, dtype='uint8')
        draw_corners(canvas, corners)
        output = cv2.addWeighted(img, 1, canvas, 0.6, 1)
        cv2.imwrite(out_name, output)
