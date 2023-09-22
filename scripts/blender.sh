#!/bin/bash

# Input .blend file and output .png file
INPUT_BLEND="$1"
OUTPUT_PNG="${1/.blend/.png}"

# Render the image preview
blender -b "$INPUT_BLEND" -o "$OUTPUT_PNG" -f 1 -F PNG -x 1

# Check if rendering was successful
if [ $? -eq 0 ]; then
    echo "Image preview rendered and saved as $OUTPUT_PNG"
else
    echo "Rendering failed"
    exit 1
fi
