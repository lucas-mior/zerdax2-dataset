#!/bin/sh

i=0
increment=1
nfens="$(wc -l fens.txt | awk '{print $1}')"
nfens=1
while [ $i -lt $nfens ]; do
    echo "i = $i"
    blender -b zerdax2_models.blend -P blender.py -- "$i" "$increment"
    i=$((i+increment))
done
