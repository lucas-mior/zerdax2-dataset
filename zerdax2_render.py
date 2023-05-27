#!/usr/bin/python
"""Blender script used to generate the synthetic dataset.
"""

import bpy
import bpy_extras
import os
import sys
from pathlib import Path
import numpy as np
import random
import gc
import mathutils

d = os.path.dirname(bpy.data.filepath)
if d not in sys.path:
    sys.path.append(d)
from zerdax2_misc import CLASSES, PIECES
import util
from util import print


DO_RENDER = False
MIN_BOARD_CORNER_PADDING = 30  # pixels
SQUARE_LENGTH = 0.039934  # m
COLLECTION_NAME = "ChessPosition"

TABLE_STYLES = 2
BOARD_STYLES = 7
PIECE_STYLES = 7

WIDTH = 960
HEIGHT = 600
ADD_TABLE = True
ADD_BOARD = True
ADD_PIECES = True
ADD_CAPTURED = True


def set_configs():
    global WIDTH, HEIGHT, ADD_TABLE, ADD_BOARD, ADD_PIECES, ADD_CAPTURED

    if np.random.rand() < 0.5:
        WIDTH = 960
        HEIGHT = 600
    else:
        WIDTH = 600
        HEIGHT = 960

    rand_num = np.random.rand()
    if rand_num < 0.5:
        ADD_TABLE = True
    else:
        ADD_TABLE = False

    rand_num = np.random.rand()
    if rand_num < 0.05:
        ADD_BOARD = False
        ADD_PIECES = False
    elif rand_num < 0.1:
        ADD_BOARD = True
        ADD_PIECES = False
    else:
        ADD_BOARD = True
        ADD_PIECES = True

    rand_num = np.random.rand()
    if rand_num < 0.5 and ADD_PIECES and ADD_TABLE:
        ADD_CAPTURED = True
    else:
        ADD_CAPTURED = False
    ADD_TABLE = True
    ADD_PIECES = True
    ADD_BOARD = True
    ADD_CAPTURED = True
    return


def setup_world():
    if bpy.context.scene.world.use_nodes:
        world = bpy.context.scene.world
        world.node_tree.nodes.clear()
        for image in bpy.data.images:
            if image.name.endswith(".hdr"):
                bpy.data.images.remove(image)

    hdr_files = [f for f in os.listdir("backgrounds/") if f.endswith(".hdr")]

    hdr_file = "backgrounds/" + random.choice(hdr_files)
    bpy.context.scene.world.use_nodes = True

    world = bpy.context.scene.world

    world.node_tree.nodes.clear()

    env_tex_node = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
    env_tex_node.image = bpy.data.images.load(hdr_file)

    output_node = world.node_tree.nodes.new('ShaderNodeOutputWorld')

    world.node_tree.links.new(env_tex_node.outputs['Color'],
                              output_node.inputs['Surface'])
    return


def setup_camera(board):
    camera = bpy.context.scene.camera
    angle = 90
    while angle >= 60 or angle <= 40:
        z = np.random.uniform(10*SQUARE_LENGTH, 15*SQUARE_LENGTH)
        x = np.random.uniform(-9*SQUARE_LENGTH, 9*SQUARE_LENGTH)
        dy = np.random.normal(9*SQUARE_LENGTH, SQUARE_LENGTH)
        dy = np.clip(dy, 8.5*SQUARE_LENGTH, 9.5*SQUARE_LENGTH)
        y = 0.7*abs(x) + 0.1*abs(z) + dy
        if np.random.rand() < 0.5:
            y = -y

        camera.location = (x, y, z)
        if board is not None:
            util.point_to(camera, board.location)
        else:
            util.point_to(camera, mathutils.Vector((0, 0, 0)))

        v = np.array([x, y, z])
        w = np.array([0, 0, 1])
        dot = np.dot(v, w)
        modulo = np.sqrt(x**2 + y**2 + z**2)
        angle = np.degrees(np.arcsin(dot/modulo))

    rot_x = np.random.uniform(-0.01, -0.05)
    rot_y = np.random.uniform(-0.02, +0.02)
    rot_z = np.random.uniform(-0.02, +0.02)

    bpy.context.view_layer.update()
    camera.rotation_euler[0] += rot_x
    camera.rotation_euler[1] += rot_y
    camera.rotation_euler[2] += rot_z

    bpy.context.view_layer.update()
    return


def setup_spotlight(light):
    z = np.uniform(13*SQUARE_LENGTH, 20*SQUARE_LENGTH)
    if light.name == "Spot0":
        x = np.random.uniform(-18*SQUARE_LENGTH, 0)
    else:
        x = np.random.uniform(0, 18*SQUARE_LENGTH)
    y = np.random.uniform(-18*SQUARE_LENGTH, 18*SQUARE_LENGTH)
    location = mathutils.Vector((x, y, z))
    light.location = location
    z = 0.0
    x = np.random.uniform(-5*SQUARE_LENGTH, 5*SQUARE_LENGTH)
    y = np.random.uniform(-5*SQUARE_LENGTH, 5*SQUARE_LENGTH)
    focus = mathutils.Vector((x, y, z))
    util.point_to(light, focus)
    return


def setup_table(table_style, board, collection):
    if ADD_TABLE:
        source_obj = bpy.data.objects[f"Table{table_style}"]
        table = source_obj.copy()
        table.data = source_obj.data.copy()
        table.animation_data_clear()
        table.hide_render = False
        table.hide_viewport = False
        table.hide_set(False)
        table.location[0] = 0
        table.location[1] = 0

        s = (0.9, 1.4)
        scale = util.create_scale(x=s, y=s, z=(1, 1))
        nscale = mathutils.Vector(scale[1])
        nscale *= scale[0]
        table.scale = nscale

        collection.objects.link(table)

        if board is not None:
            vertices = board.data.vertices
            board_zs = [(board.matrix_world @ v.co).z for v in vertices]
            table.location[2] = min(board_zs)
        else:
            table.location[2] = 0
    else:
        table = None

    bpy.context.view_layer.update()
    return table


def setup_board(board_style, collection):
    if ADD_BOARD:
        source_obj = bpy.data.objects[f"Board{board_style}"]
        board = source_obj.copy()
        board.data = source_obj.data.copy()
        board.animation_data_clear()
        board.hide_render = False
        board.hide_viewport = False
        board.hide_set(False)
        board.location[0] = 0
        board.location[1] = 0
        collection.objects.link(board)
    else:
        board = None

    bpy.context.view_layer.update()
    return board


def setup_sun():
    strength = np.random.uniform(0.4, 0.5)
    bpy.data.lights['Sun'].energy = strength
    return


def setup_lighting():
    flash = bpy.data.objects["LightCameraFlash"]
    spot0 = bpy.data.objects["LightSpot0"]
    spot1 = bpy.data.objects["LightSpot1"]
    sun = bpy.data.objects["LightSun"]

    modes = {
        "flash": {
            flash: True,
            spot0: False,
            spot1: False,
            sun: True,
        },
        "spotlights": {
            flash: False,
            spot0: True,
            spot1: True,
            sun: True,
        }
    }
    which = np.random.randint(len(modes))
    mode, visibilities = list(modes.items())[which]

    for obj, visibility in visibilities.items():
        obj.hide_render = not visibility
        obj.hide_set(not visibility)
        obj.hide_viewport = not visibility
    return


def dump_yolo_txt(txtpath, objects):
    print(f"dumping txt {txtpath}...")
    with txtpath.open("w") as txt:
        for obj in objects:
            name = obj['piece']
            number = CLASSES[name]

            left = obj['box'][0]
            top = obj['box'][1]
            dx = obj['box'][2]
            dy = obj['box'][3]

            right = left + dx
            bottom = top + dy
            xc = round((left + right)/2)
            yc = round((top + bottom)/2)

            xc, dx = xc/WIDTH, dx/WIDTH
            yc, dy = yc/HEIGHT, dy/HEIGHT

            yolo = f"{number} {xc} {yc} {dx} {dy}\n"
            print(yolo, end="")
            txt.write(yolo)
        txt.close()
    return


def add_piece(piece, collection, piece_style, scale_pieces):
    piece_name = PIECES[piece["name"]]
    name = piece_name + str(piece_style)

    offsets = np.random.normal((.5,)*2, (.1,)*2)
    offsets = np.clip(offsets, .32, .68)
    rank_offset, file_offset = offsets
    rank = piece["square"][1] + rank_offset
    file = piece["square"][0] + file_offset

    # Translate to coordinate system
    # where the origin is in the middle of the board
    rank -= 4
    file -= 4

    location = mathutils.Vector((file, rank, 0)) * SQUARE_LENGTH
    rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))
    scale = mathutils.Vector(scale_pieces[1])
    scale *= scale_pieces[0]

    source_obj = bpy.data.objects[name]
    obj = source_obj.copy()
    obj.data = source_obj.data.copy()
    obj.animation_data_clear()
    obj.location = location
    obj.rotation_euler = rotation
    obj.scale = scale
    collection.objects.link(obj)
    return obj


def add_extra(source_obj, collection, table, scale):
    vertices = table.data.vertices
    z_vertices = [(table.matrix_world @ v.co).z for v in vertices]
    x_vertices = [(table.matrix_world @ v.co).x for v in vertices]
    y_vertices = [(table.matrix_world @ v.co).y for v in vertices]
    z = max(z_vertices)
    xmin = min(x_vertices) + SQUARE_LENGTH/2
    xmax = max(x_vertices) - SQUARE_LENGTH/2
    ymin = min(y_vertices) + SQUARE_LENGTH/2
    ymax = max(y_vertices) - SQUARE_LENGTH/2

    distance = 10000
    distance_factor = 6
    i = 0
    while True:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        j = 0
        limit = distance_factor*SQUARE_LENGTH
        while abs(x) < limit and abs(y) < limit:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            j += 1
            if j >= 20:
                break
        if j >= 20:
            i = 20
            break

        for other in collection.objects:
            if "Table" in other.name:
                continue
            d = util.min_distance_point(other, (x, y, z))
            if d < distance:
                distance = d
        if distance > SQUARE_LENGTH:
            break
        i += 1
        if i >= 20:
            break

    obj = None
    if i < 20:
        obj = source_obj.copy()
        obj.data = source_obj.data.copy()
        obj.animation_data_clear()
        obj.location = (x, y, z)
        rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))
        nscale = mathutils.Vector(scale[1])
        nscale *= scale[0]
        obj.rotation_euler = rotation
        obj.scale = nscale
        collection.objects.link(obj)

        obj.hide_render = False
        obj.hide_set(False)
        obj.hide_viewport = False

    return obj


def board_box(corners):
    x = [c[0] for c in corners]
    y = [c[1] for c in corners]
    cornersx = sorted(x)
    cornersy = sorted(y)

    x0, x1 = cornersx[0], cornersx[3]
    dx = x1 - x0

    y0, y1 = cornersy[0], cornersy[3]
    dy = y1 - y0

    box = [x0, y0, dx, dy]
    return box


def setup_shot(position, output_file):
    scene = bpy.context.scene

    # Setup rendering
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "JPEG"
    scene.render.filepath = str(output_file)
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT

    if COLLECTION_NAME not in bpy.data.collections:
        collection = bpy.data.collections.new(COLLECTION_NAME)
        scene.collection.children.link(collection)
    collection = bpy.data.collections[COLLECTION_NAME]

    for obj in bpy.data.objects:
        obj.select_set(False)
    for obj in collection.objects:
        obj.select_set(True)
        bpy.ops.object.delete()

    styles = {
        "table": np.random.randint(0, TABLE_STYLES),
        "board": np.random.randint(0, BOARD_STYLES),
        "piece": np.random.randint(0, PIECE_STYLES),
    }
    board = setup_board(styles['board'], collection)
    table = setup_table(styles['table'], board, collection)

    corner_coords = None
    while not corner_coords:
        setup_camera(board)
        if ADD_BOARD:
            corner_coords = get_corner_coordinates(scene)
        else:
            break

    setup_world()
    setup_lighting()

    objects = []

    if ADD_BOARD:
        corner_coords = sorted(corner_coords, key=lambda x: x[0])
        objects.append({
            "piece": "Board",
            "box": board_box(corner_coords),
        })

    pieces = parse_position(fen)
    captured_pieces = get_missing_pieces(fen)
    scale_pieces = util.create_scale()
    if ADD_PIECES:
        for piece in pieces:
            obj = add_piece(piece, collection, styles['piece'], scale_pieces)
            objects.append({
                "piece": piece["name"],
                "box": util.get_bounding_box(scene, obj)
            })

    if ADD_TABLE:
        if ADD_CAPTURED:
            for piece in captured_pieces:
                name = PIECES[piece] + str(styles['piece'])
                source_obj = bpy.data.objects[name]
                obj = add_extra(source_obj, collection, table, scale_pieces)
                # if obj is not None:
                #     box = util.get_bounding_box(scene, obj)
                #     if box is not None:
                #         objects.append({
                #             "piece": piece,
                #             "box": box,
                #         })
        if np.random.rand() < 0.5:
            scale = util.create_scale()
            source_obj = bpy.data.objects["RedCup"]
            add_extra(source_obj, collection, table, scale)
        if np.random.rand() < 0.5:
            scale = util.create_scale()
            source_obj = bpy.data.objects["CoffeCup"]
            add_extra(source_obj, collection, table, scale)

    return objects


def get_corner_coordinates(scene):
    corner_points = np.array([[-1., -1], [-1, 1], [1, 1], [1, -1]])
    corner_points *= 4*SQUARE_LENGTH
    corner_points = np.concatenate((corner_points, np.zeros((4, 1))), axis=-1)
    render = scene.render

    def _surpass_padding(resolution, p):
        dp = resolution - MIN_BOARD_CORNER_PADDING
        return not (MIN_BOARD_CORNER_PADDING <= p <= dp)

    def _get_coords_corners():
        for corner in corner_points:
            x, y, z = bpy_extras.object_utils.world_to_camera_view(
                scene, scene.camera, mathutils.Vector(corner)).to_tuple()
            y = 1. - y
            x *= render.resolution_x * render.resolution_percentage * .01
            y *= render.resolution_y * render.resolution_percentage * .01
            x, y = round(x), round(y)

            if _surpass_padding(render.resolution_x, x) or \
               _surpass_padding(render.resolution_y, y):
                raise ValueError

            yield x, y
    try:
        return list(_get_coords_corners())
    except ValueError:
        return None


def get_missing_pieces(fen):
    pieces = list("KkQqBbBbNnNnRrRrPPPPPPPPpppppppp")
    board = list(''.join(filter(str.isalpha, fen)))
    for piece in board:
        try:
            pieces.remove(piece)
        except ValueError:
            pass
    return pieces


def parse_position(fen):
    pieces = []

    fen_parts = fen.split(' ')
    piece_positions = fen_parts[0]

    ranks = piece_positions.split('/')

    for rank_idx, rank in enumerate(ranks):
        file_idx = 0

        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                piece_name = char
                square = (file_idx, 7 - rank_idx)  # (column, row)
                piece = {'name': piece_name, 'square': square}
                pieces.append(piece)
                file_idx += 1

    return pieces


if __name__ == "__main__":
    argv = sys.argv
    print("="*30, f"{argv[0]}.py", "="*30)

    fens_path = Path("fens.txt")
    with fens_path.open("r") as f:
        for i, fen in enumerate(map(str.strip, f)):
            if i != 6000:
                continue
            print(f"FEN #{i} = {fen}")
            print(f"FEN #{i} = {fen}", file=sys.stderr)

            set_configs()

            filename = Path("renders") / f"{i:05d}.png"
            objects = setup_shot(fen, filename)
            if DO_RENDER:
                print(f"rendering {filename}...")
                bpy.ops.render.render(write_still=1)
                if ADD_BOARD:
                    txtpath = filename.parent / (filename.stem + ".txt")
                    dump_yolo_txt(txtpath, objects)

            if i % 100 == 0:
                bpy.ops.outliner.orphans_purge()
                gc.collect()
            break
    print("="*60)
