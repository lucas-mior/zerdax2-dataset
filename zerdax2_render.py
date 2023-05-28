#!/usr/bin/python

import bpy
import bpy_extras
import os
import sys
from pathlib import Path
import numpy as np
import random
import gc
import mathutils

pwd = os.path.dirname(bpy.data.filepath)
if pwd not in sys.path:
    sys.path.append(pwd)
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

    if np.random.rand() < 0.5:
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

    if np.random.rand() < 0.5 and ADD_PIECES and ADD_TABLE:
        ADD_CAPTURED = True
    else:
        ADD_CAPTURED = False
    ADD_TABLE = True
    ADD_CAPTURED = True
    return


def setup_world():
    world = bpy.context.scene.world
    if world.use_nodes:
        world.node_tree.nodes.clear()
        for image in bpy.data.images:
            if image.name.endswith(".hdr"):
                bpy.data.images.remove(image)

    hdr_files = [f for f in os.listdir("backgrounds/") if f.endswith(".hdr")]

    hdr_file = "backgrounds/" + random.choice(hdr_files)
    world.use_nodes = True
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
        z = np.random.uniform(11*SQUARE_LENGTH, 14*SQUARE_LENGTH)
        x = np.random.uniform(-9*SQUARE_LENGTH, 9*SQUARE_LENGTH)
        dy = np.random.uniform(8.5*SQUARE_LENGTH, 9.5*SQUARE_LENGTH)
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

    rot_x = np.random.uniform(-0.02, -0.02)
    rot_y = np.random.uniform(-0.02, +0.02)
    rot_z = np.random.uniform(-0.02, +0.02)

    bpy.context.view_layer.update()
    camera.rotation_euler[0] += rot_x
    camera.rotation_euler[1] += rot_y
    camera.rotation_euler[2] += rot_z

    bpy.context.view_layer.update()
    return camera


def setup_spotlight(spotlight):
    if spotlight.name == "Spot0":
        x = np.random.uniform(-18*SQUARE_LENGTH, 0)
    else:
        x = np.random.uniform(0, 18*SQUARE_LENGTH)

    y = np.random.uniform(-18*SQUARE_LENGTH, 18*SQUARE_LENGTH)
    z = np.uniform(13*SQUARE_LENGTH, 20*SQUARE_LENGTH)
    spotlight.location = mathutils.Vector((x, y, z))

    z = 0.0
    x = np.random.uniform(-5*SQUARE_LENGTH, 5*SQUARE_LENGTH)
    y = np.random.uniform(-5*SQUARE_LENGTH, 5*SQUARE_LENGTH)
    focus = mathutils.Vector((x, y, z))
    util.point_to(spotlight, focus)
    return


def object_copy(name, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    source_obj = bpy.data.objects[name]
    obj = source_obj.copy()
    obj.data = source_obj.data.copy()
    obj.animation_data_clear()

    obj.hide_render = False
    obj.hide_viewport = False
    obj.hide_set(False)

    obj.location = location
    obj.rotation_euler = rotation
    obj.scale = scale
    return obj


def setup_table(table_style, board, collection):
    if ADD_TABLE:
        s = (0.9, 1.4)
        scale_table = util.create_scale(x=s, y=s, z=(1, 1))
        scale = mathutils.Vector(scale_table["coords"])
        scale *= scale_table["global"]

        table = object_copy(f"Table{table_style}", scale=scale)
        collection.objects.link(table)

        if board is not None:
            vertices = board.data.vertices
            board_zs = [(board.matrix_world @ v.co).z for v in vertices]
            table.location[2] = min(board_zs)
        else:
            table.location[2] = 0
        bpy.context.view_layer.update()
    else:
        table = None

    return table


def setup_board(board_style, collection):
    if ADD_BOARD:
        board = object_copy(f"Board{board_style}")
        collection.objects.link(board)
        bpy.context.view_layer.update()
    else:
        board = None

    return board


def setup_sun():
    strength = np.random.uniform(0.1, 0.9)
    bpy.data.lights['Sun'].energy = strength
    bpy.context.view_layer.update()
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

    bpy.context.view_layer.update()
    return


def board_box(corners):
    x = [c[0] for c in corners]
    y = [c[1] for c in corners]
    cornersx = sorted(x)
    cornersy = sorted(y)

    x0, x1 = cornersx[0], cornersx[3]
    dx = x1 - x0

    y0, y1 = cornersy[0], cornersy[3]
    dy = y1 - y0

    box = [x0/WIDTH, y0/HEIGHT, dx/WIDTH, dy/HEIGHT]
    return box


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
            xc = (left + right)/2
            yc = (top + bottom)/2

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
    scale = mathutils.Vector(scale_pieces["coords"])
    scale *= scale_pieces["global"]

    piece = object_copy(name, location, rotation, scale)
    collection.objects.link(piece)
    return piece


def add_extra(source_name, collection, xlim, ylim, z, table, scale_obj):
    distance = 10000
    not_piece = "Black" not in source_name and "White" not in source_name
    if not_piece:
        tolerance = 4*SQUARE_LENGTH
        absolute = 8*SQUARE_LENGTH
    else:
        tolerance = 1*SQUARE_LENGTH
        absolute = 6*SQUARE_LENGTH

    limits = [xlim[0]-SQUARE_LENGTH, xlim[1]+SQUARE_LENGTH,
              ylim[0]-SQUARE_LENGTH, ylim[1]+SQUARE_LENGTH]
    choice = np.random.randint(4)
    limits[choice] = absolute if choice % 2 == 0 else -absolute
    xlim = limits[0:2]
    ylim = limits[2:4]

    i = 0
    while True:

        x = np.random.uniform(*xlim)
        y = np.random.uniform(*ylim)
        if "Table1" in table.name:
            elipsis = table.dimensions / 2
            j = 0
            while (x / elipsis[0])**2 + (y / elipsis[1])**2 >= 0.95:
                x = np.random.uniform(*limits[0:2])
                y = np.random.uniform(*limits[2:4])
                j += 1
                if j >= 10:
                    return None

        for other in collection.objects:
            if "Table" in other.name or "Board" in other.name:
                continue
            print(f"Checking {other.name}")
            d = util.distance_points(other.location, (x, y, z))
            if d < distance:
                distance = d
            if distance <= tolerance:
                break
        if distance > tolerance:
            break
        i += 1
        if i >= 10:
            return None

    location = (x, y, z)
    rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))
    scale = mathutils.Vector(scale_obj["coords"])
    scale *= scale_obj["global"]

    obj = object_copy(source_name, location, rotation, scale)
    collection.objects.link(obj)
    return obj


def setup_shot(position, output_file):
    scene = bpy.context.scene

    # Setup rendering
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
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

    setup_world()
    setup_lighting()

    styles = {
        "table": np.random.randint(0, TABLE_STYLES),
        "board": np.random.randint(0, BOARD_STYLES),
        "piece": np.random.randint(0, PIECE_STYLES),
    }
    board = setup_board(styles['board'], collection)
    table = setup_table(styles['table'], board, collection)

    corners = None
    while not corners:
        camera = setup_camera(board)
        corners = get_corner_coordinates(scene, camera)

    objects = []

    if ADD_BOARD:
        corners = sorted(corners, key=lambda x: x[0])
        objects.append({
            "piece": "Board",
            "box": board_box(corners),
        })

    if ADD_PIECES:
        position_pieces = parse_position(fen)
        captured_pieces = get_missing_pieces(fen)
        scale_pieces = util.create_scale()
        for piece in position_pieces:
            obj = add_piece(piece, collection, styles['piece'], scale_pieces)
            box = util.get_bounding_box(scene, obj)
            if box is None:
                print("Bounding box error. Check the camera view.")
                return None
            objects.append({
                "piece": piece["name"],
                "box": box
            })

    if ADD_TABLE:
        vertices = table.data.vertices
        z_vertices = [(table.matrix_world @ v.co).z for v in vertices]
        x_vertices = [(table.matrix_world @ v.co).x for v in vertices]
        y_vertices = [(table.matrix_world @ v.co).y for v in vertices]
        xlim = [min(x_vertices), max(x_vertices)]
        ylim = [min(y_vertices), max(y_vertices)]
        z = max(z_vertices)
        misc = ["RedCup", "CoffeCup"]
        for source_name in misc:
            if np.random.rand() < 0.5:
                scale = util.create_scale()
                add_extra(source_name, collection, xlim, ylim, z, table, scale)
        if ADD_PIECES and ADD_CAPTURED:
            for piece in captured_pieces:
                source_name = PIECES[piece] + str(styles['piece'])
                obj = add_extra(source_name, collection,
                                xlim, ylim, z, table, scale_pieces)
                if is_object_hiding(obj):
                    print("Hiding!")
    return objects


def is_object_hiding(obj):
    if obj is None:
        return False
    scene = bpy.context.scene
    camera = scene.camera

    ray_origin = camera.location
    ray_direction = obj.location - camera.location
    ray_direction.normalize()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    ray = scene.ray_cast(depsgraph, ray_origin, ray_direction)
    return ray[0] and ray[4] != obj


def get_corner_coordinates(scene, camera):
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
                scene, camera, mathutils.Vector(corner)).to_tuple()
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
                square = (file_idx, 7 - rank_idx)
                piece = {'name': piece_name, 'square': square}
                pieces.append(piece)
                file_idx += 1

    return pieces


if __name__ == "__main__":
    argv = sys.argv
    print("="*30, f"{argv[0]}.py", "="*30)

    gc.disable()
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
            if objects is None:
                break
            if DO_RENDER:
                print(f"rendering {filename}...")
                bpy.ops.render.render(write_still=1)
                if ADD_BOARD:
                    txtpath = filename.parent / (filename.stem + ".txt")
                    dump_yolo_txt(txtpath, objects)

            if i % 20 == 0:
                bpy.ops.outliner.orphans_purge()
                gc.collect()
            break
    print("="*60)
