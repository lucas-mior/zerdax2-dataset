#!/bin/sh

i=0
increment=100
nfens="$(wc -l fens.txt | awk '{print $1}')"
while [ $i -lt $nfens ]; do
    echo "================================================"
    echo "contagem = $i"
    echo "================================================"
    blender -b zerdax2_models.blend -P blender.py -- "$i" "$increment"
    i=$((i+increment))
    echo "================================================"
done
