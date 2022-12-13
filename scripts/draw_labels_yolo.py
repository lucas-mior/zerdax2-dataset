#!/usr/bin/python
"""
draws labels and boxes around chess pieces
"""
import cv2
import numpy as np
import sys

from zerdax2 import COLORS, NAMES


def get_piece(line, width, heigth):
    a = line.strip().split(" ")
    xc, yc = float(a[1]), float(a[2])
    dx, dy = float(a[3]), float(a[4])
    x0, y0 = xc - dx/2, yc - dy/2
    x1, y1 = x0 + dx, y0 + dy
    x0 = round(x0*width)
    x1 = round(x1*width)
    y0 = round(y0*heigth)
    y1 = round(y1*heigth)
    piece = {
        "number": a[0],
        "name": NAMES[str(a[0])]
    }
    return (piece, x0, y0, x1, y1)


def get_pieces(img, txt_name):
    pieces = []
    with open(txt_name) as f:
        Lines = f.readlines()
        for line in Lines:
            piece = get_piece(line, img.shape[1], img.shape[0])
            pieces.append(piece)
    return pieces


def draw_boxes(img_name, txt_name):
    img = cv2.imread(img_name)
    pieces = get_pieces(img, txt_name)
    basename = txt_name.rsplit(".", 1)[0]
    output = f'{basename}_yolo.png'
    canvas = np.zeros(img.shape, dtype='uint8')
    for piece in pieces:
        p, x0, y0, x1, y1 = piece
        color = COLORS[p['number']]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, p['name'], (x0-10, y0-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        canvas2 = cv2.addWeighted(img, 1, canvas, 0.6, 1)
    cv2.imwrite(output, canvas2)


if __name__ == "__main__":
    img_name = sys.argv[1]
    txt_name = sys.argv[2]
    draw_boxes(img_name, txt_name)
