#!/usr/bin/python

import sys

INPUT = sys.argv[1]

fens_in = open(INPUT, 'r')
out = open(f"{INPUT}.2", 'w')

while line := fens_in.readline():
    count = 0

    if not line:
        break
    for c in line:
        if c.isalpha():
            count += 1
    a = f"{count} {line}"
    out.write(a)

fens_in.close()
out.close()
