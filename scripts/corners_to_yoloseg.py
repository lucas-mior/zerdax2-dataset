#!/usr/bin/python
"""
converts json to yolo plain txt label segmentation format
"""

import json
import sys


def json2yolo(filename):
    basename = filename.rsplit(".", 1)[0]
    jsonname = f'{basename}.json'
    txtname = f'{basename}.txt'

    data_file = open(jsonname)
    data = json.load(data_file)
    txt = open(txtname, 'w')

    cs = data['corners_normalized']
    corners = [str(item) for sublist in cs for item in sublist]
    corners = ' '.join(corners)
    print(f"0 {corners}", file=txt)

    # Closing file
    data_file.close()
    txt.close()
    return


if __name__ == "__main__":
    files = sys.argv[1:]
    for filename in files:
        json2yolo(filename)
