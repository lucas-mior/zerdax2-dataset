#!/usr/bin/python
"""
Adds normalized corners coordinates
to json file
"""

import json
import sys
import cv2


def add_norm_corners(data, img):
    width = img.shape[1]
    heigth = img.shape[0]

    corners = data['corners']
    norm_coners = []
    for c in corners:
        norm_coners.append([c[0]/width, c[1]/heigth])

    data['corners_normalized'] = norm_coners
    return data


def sort_corners(imgname):
    corner = data['corners']
    ncorn = sorted(corner, key=lambda x: x[0])
    data['corners'] = ncorn

    # Closing file
    json.dump(data, out_file, indent=4)
    data_file.close()
    return data


if __name__ == "__main__":
    files = sys.argv[1:]
    for imgname in files:
        basename = imgname.rsplit(".", 1)[0]
        jsonname = f'{basename}.json'
        imgname = f'{basename}.png'

        img = cv2.imread(imgname)
        data_file = open(jsonname)
        data = json.load(data_file)

        data = sort_corners(data)
        data = add_norm_corners(data, img)

        out_file = open(f'{basename}_2.json', 'w')
        json.dump(data, out_file, indent=4)
        data_file.close()
