#!/usr/bin/python

import json
import sys


def json2yolo(filename):
    data_file = open(filename)
    data = json.load(data_file)

    name_to_class = {
        "K": 0,
        "Q": 1,
        "R": 2,
        "B": 3,
        "N": 4,
        "P": 5,
        "k": 0,
        "q": 1,
        "r": 2,
        "b": 3,
        "n": 4,
        "p": 5,
    }

    # width = data['width']
    # heigth = data['heigth']
    width = 1280
    heigth = 800

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
        print(name_to_class[name], x, y, dx, dy)

    # Closing file
    data_file.close()
    return


if __name__ == "__main__":
    filename = sys.argv[1]
    json2yolo(filename)
