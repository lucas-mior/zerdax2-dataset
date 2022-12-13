#!/usr/bin/python

import json
import sys

imgname = sys.argv[1]
basename = imgname.rsplit(".", 1)[0]
data_file = open(f'{basename}.json')
data = json.load(data_file)

corner = data['corners']
ncorn = sorted(corner, key=lambda x: x[0])
data['corners'] = ncorn
data['width'] = 1280
data['heigth'] = 800

# Closing file
out_file = open(f'{basename}_2.json', 'w')
json.dump(data, out_file, indent=4)
data_file.close()
