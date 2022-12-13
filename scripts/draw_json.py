#!/usr/bin/python

import json
import cv2
import numpy as np
import sys

imgname = sys.argv[1]
basename = imgname.rsplit(".", 1)[0]
data_file = open(f'{basename}.json')
data = json.load(data_file)

img = cv2.imread(imgname)

canvas = np.empty(img.shape, dtype='uint8') * 0

print("Pieces:")
for piece in data['pieces']:
    left = piece['box'][0]
    top = piece['box'][1]
    right = left + piece['box'][2]
    bottom = top + piece['box'][3]
    cv2.rectangle(canvas, (left, top), (right, bottom),
                  color=(255, 0, 0), thickness=2)
    print(piece)

print("Board corners:")
i = 0
for corner in data['corners']:
    i += 1
    canvas = cv2.circle(canvas, corner,
                        radius=7, color=(0, 200, 170-i*40), thickness=-1)
    print(corner)

output = cv2.addWeighted(img, 1, canvas, 0.6, 1)
cv2.imwrite(f"{basename}_draw.png", output)

# Closing file
data_file.close()
