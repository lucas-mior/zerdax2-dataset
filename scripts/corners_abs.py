#!/usr/bin/python

import json
import sys

imgname = sys.argv[1]
basename = imgname.rsplit(".", 1)[0]
data_file = open(f'{basename}.json')
data = json.load(data_file)

width = 1280
heigth = 800
corners = data['corners']
norm_coners = []
for c in corners:
    norm_coners.append([c[0]/width, c[1]/heigth])

data['corners_normalized'] = norm_coners

# Closing file
out_file = open(f'{basename}_2.json', 'w')
json.dump(data, out_file, indent=4)
data_file.close()
