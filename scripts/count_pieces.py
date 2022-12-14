#!/usr/bin/python

import sys
import json

if __name__ == "__main__":
    jsons = sys.argv[1:]
    for jsonname in jsons:
        data_file = open(jsonname)
        basename = jsonname.rsplit(".", 1)[0]
        out_file = f'{basename}_2.json'
        data = json.load(data_file)

        npieces = 0
        for piece in data['pieces']:
            npieces += 1

        data['piece_amount'] = npieces
        json.dump(data, out_file, indent=4)
        data_file.close()
