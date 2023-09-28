#!/usr/bin/python
import sys

INPUT = sys.argv[1]

fens_in = open(INPUT, 'r')

for i in range(3, 33, 1):
    out = open(f"{INPUT}.{i:02}", 'w')

    while line := fens_in.readline():
        s = line.split(" ")
        npieces = int(s[0])

        if npieces == i:
            out.write(s[1])
        elif npieces > i:
            break

    out.close()

fens_in.close()
