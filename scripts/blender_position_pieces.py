#!/usr/bin/python

import bpy
import re

PIECES = list(reversed(['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']))


def get_piece_group(piece_name):
    print(f"get_piece_group(\"{piece_name}\"")
    match = re.search(r'[0-9]+', piece_name)
    if match:
        return int(match.group())
    return None


def get_piece_color(piece_name):
    print(f"get_piece_color(\"{piece_name}\"")
    match = re.search(r'^(B|W)', piece_name)
    if match:
        return match.group()
    return None


def get_piece_type(piece_name):
    print(f"get_piece_type(\"{piece_name}\"")
    match = re.search(r'(King|Queen|Rook|Bishop|Knight|Pawn)', piece_name)
    if match:
        return match.group()
    return None


objects_dict = {}


for obj in bpy.context.scene.objects:
    obj_collection_name = obj.users_collection[0].name
    if "Pieces" not in obj_collection_name:
        continue
    print(f"{obj_collection_name=}")

    # Extract the relevant information from the object name
    color = get_piece_color(obj.name)
    group = get_piece_group(obj.name)

    if color is None:
        color = 'B'
    if group is None:
        print("Could not find group.")
        print("Check your objects")
        continue
    
    print(f"{color=}")
    print(f"{group=}")
    # If the color is not already in the dictionary, create a new entry
    if color not in objects_dict:
        objects_dict[color] = {}

    # If the group is not already in the dictionary, create a new entry
    if group not in objects_dict[color]:
        objects_dict[color][group] = []

    # Add the object to the corresponding group
    objects_dict[color][group].append(obj)

# Set the X and Y positions based on the object groups
for color, color_groups in objects_dict.items():
    for group, group_objects in color_groups.items():
        # Sort the objects by their piece type in the specified order
        sorted_objects = sorted(group_objects,
                                key=lambda obj:
                                PIECES.index(get_piece_type(obj.name)))

        piece_width = 0.03
        piece_height = 0.04
        x_spacing = 0.05
        y_spacing = 0.05
        color_offset = 0.02

        x_offset = -((len(sorted_objects) - 1) * piece_width
                     + (len(sorted_objects) - 1) * x_spacing) / 2.0
        y_offset = group * (piece_height + y_spacing)

        for i, obj in enumerate(sorted_objects):
            x = x_offset + i * (piece_width + x_spacing)
            if color == "W":
                y = y_offset + color_offset
            else:
                y = y_offset - color_offset

            obj.location = (-y, -x, 0)
