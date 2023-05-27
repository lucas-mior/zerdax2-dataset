#!/usr/bin/python
"""Blender script used to generate the synthetic dataset.
"""

import bpy
import bpy_extras.object_utils
import os
import sys
from pathlib import Path
import numpy as np
import random
import builtins as __builtin__
import gc
import chess
import mathutils

d = os.path.dirname(bpy.data.filepath)
if d not in sys.path:
    sys.path.append(d)
from zerdax2_misc import CLASSES, PIECES


DO_DEBUG = True
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


def console_print(*args, **kwargs):
    for area in bpy.context.screen.areas:
        if area.type != 'CONSOLE':
            continue

        with bpy.context.temp_override(window=bpy.context.window,
                                       area=area,
                                       region=area.regions[-1],
                                       space_data=area.spaces.active,
                                       screen=bpy.context.screen):
            s = " ".join([str(arg) for arg in args])
            for line in s.split("\n"):
                bpy.ops.console.scrollback_append(text=line)
    return


def print(*args, **kwargs):
    console_print(*args, **kwargs)
    __builtin__.print(*args, **kwargs)
    return


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
    return


def point_to(obj, focus, roll=0):
    # Based on https://blender.stackexchange.com/a/127440
    location = obj.location
    direction = focus - location
    quat = direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    roll_matrix = mathutils.Matrix.Rotation(roll, 4, "Z")
    location = location.to_tuple()
    obj.matrix_world = quat @ roll_matrix
    obj.location = location
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
    print(f"setup_camera({board})")
    camera = bpy.context.scene.camera
    angle = 90
    while angle >= 60 or angle <= 40:
        z = np.random.normal(13*SQUARE_LENGTH, 2*SQUARE_LENGTH)
        z = np.clip(z, 10*SQUARE_LENGTH, 15*SQUARE_LENGTH)
        x = np.random.uniform(-9*SQUARE_LENGTH, 9*SQUARE_LENGTH)
        dy = np.random.normal(9*SQUARE_LENGTH, SQUARE_LENGTH)
        dy = np.clip(dy, 8.5*SQUARE_LENGTH, 9.5*SQUARE_LENGTH)
        y = 0.7*abs(x) + 0.1*abs(z) + dy
        if np.random.randint(0, 2) == 1:
            y = -y

        camera.location = (x, y, z)
        if board is not None:
            point_to(camera, board.location)
        else:
            point_to(camera, mathutils.Vector((0, 0, 0)))

        v = np.array([x, y, z])
        w = np.array([0, 0, 1])
        dot = np.dot(v, w)
        modulo = np.sqrt(x**2 + y**2 + z**2)
        angle = np.degrees(np.arcsin(dot/modulo))
        print(f"Camera to table angle: {angle:.2f}")

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
    print(f"setup_spotlight(light={light.name})")
    z = np.random.normal(16*SQUARE_LENGTH, 2*SQUARE_LENGTH)
    z = np.clip(z, 13*SQUARE_LENGTH, 22*SQUARE_LENGTH)
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
    point_to(light, focus)
    return


def setup_table(table_style, board, collection):
    print(f"setup_table(style={table_style}")
    for i in range(0, TABLE_STYLES):
        source_obj = bpy.data.objects[f"Table{i}"]
        if i == table_style and ADD_TABLE:
            obj = source_obj.copy()
            obj.data = source_obj.data.copy()
            obj.animation_data_clear()
            obj.hide_render = False
            obj.hide_viewport = False
            obj.hide_set(False)
            obj.location[0] = 0
            obj.location[1] = 0
            collection.objects.link(obj)
        else:
            source_obj.hide_render = False
            source_obj.hide_viewport = False
            source_obj.hide_set(False)

    if ADD_TABLE:
        if board is not None:
            vertices = board.data.vertices
            board_zs = [(board.matrix_world @ v.co).z for v in vertices]
            obj.location[2] = min(board_zs)
        else:
            obj.location[2] = 0
    else:
        obj = None

    bpy.context.view_layer.update()
    return obj


def setup_board(board_style, collection):
    print(f"setup_board(board_style={board_style})")
    for i in range(0, BOARD_STYLES):
        source_obj = bpy.data.objects[f"Board{i}"]
        if i == board_style and ADD_BOARD:
            obj = source_obj.copy()
            obj.data = source_obj.data.copy()
            obj.animation_data_clear()
            obj.hide_render = False
            obj.hide_viewport = False
            obj.hide_set(False)
            obj.location[0] = 0
            obj.location[1] = 0
            collection.objects.link(obj)
        else:
            source_obj.hide_render = False
            source_obj.hide_viewport = False
            source_obj.hide_set(False)

    if not ADD_BOARD:
        obj = None
    bpy.context.view_layer.update()
    return obj


def setup_sun():
    print("setup_sun()")
    strength = np.random.uniform(0.4, 0.5)
    bpy.data.lights['Sun'].energy = strength
    return


def setup_lighting():
    print("setup_lighting()")
    flash = bpy.data.objects["CameraFlashLight"]
    spot0 = bpy.data.objects["Spot0"]
    spot1 = bpy.data.objects["Spot1"]
    sun = bpy.data.objects["Sun"]

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
    mode, visibilities = list(modes.items())[np.random.randint(len(modes))]

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


def add_piece(piece, square, collection, piece_style):
    color = {
        chess.WHITE: "White",
        chess.BLACK: "Black"
    }[piece.color]
    piece = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King"
    }[piece.piece_type]
    name = color + piece + str(piece_style)

    # Position the piece in the middle of the square
    offsets = np.random.normal((.5,)*2, (.1,)*2)
    offsets = np.clip(offsets, .32, .68)
    rank_offset, file_offset = offsets
    rank = chess.square_rank(square) + rank_offset
    file = chess.square_file(square) + file_offset

    # Translate to coordinate system where the origin is in the middle of the
    # board
    rank -= 4
    file -= 4

    location = mathutils.Vector((file, rank, 0)) * SQUARE_LENGTH
    rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))

    source_obj = bpy.data.objects[name]
    obj = source_obj.copy()
    obj.data = source_obj.data.copy()
    obj.animation_data_clear()
    obj.location = location
    obj.rotation_euler = rotation
    collection.objects.link(obj)
    return obj


def place_group(group, xmin, xmax, ymin, ymax, dist_factor=6):
    print(f"place_group(group={group},",
          f"xmin={xmin/SQUARE_LENGTH:.2f}, xmax={xmax/SQUARE_LENGTH:.2f},",
          f"ymin={ymin/SQUARE_LENGTH:.2f}, ymax={ymax/SQUARE_LENGTH:.2f})")
    pieces_loc = []
    xc = np.random.uniform(xmin, xmax)
    yc = np.random.uniform(ymin, ymax)
    print(f"center = ({xc/SQUARE_LENGTH}, {yc/SQUARE_LENGTH})")
    limit = dist_factor*SQUARE_LENGTH
    while abs(xc) < limit and abs(yc) < limit:
        xc = np.random.uniform(xmin, xmax)
        yc = np.random.uniform(ymin, ymax)

    for piece in group:
        x = 1000
        y = 1000
        dist = 0
        i = 0
        if abs(xc) > abs(yc):
            limit = dist_factor*SQUARE_LENGTH
            while abs(x) < limit and abs(y) < limit or dist < SQUARE_LENGTH/2:
                dist = 1000
                x = np.random.normal(xc, 2*SQUARE_LENGTH)
                y = np.random.normal(yc, 4*SQUARE_LENGTH)
                x = np.clip(x, xmin, xmax)
                y = np.clip(y, ymin, ymax)
                for p in pieces_loc:
                    d = dist_point((x, y, 0), (p[1][0], p[1][1], 0))
                    if d < dist:
                        dist = d
                i += 1
                if i >= 20:
                    break
        else:
            limit = dist_factor*SQUARE_LENGTH
            while abs(x) < limit and abs(y) < limit or dist < SQUARE_LENGTH/2:
                dist = 1000
                x = np.random.normal(xc, 4*SQUARE_LENGTH)
                y = np.random.normal(yc, 2*SQUARE_LENGTH)
                x = np.clip(x, xmin, xmax)
                y = np.clip(y, ymin, ymax)
                for p in pieces_loc:
                    d = dist_point((x, y, 0), (p[1][0], p[1][1], 0))
                    if d < dist:
                        dist = d
                i += 1
                if i >= 20:
                    break
        if i < 20:
            pieces_loc.append((piece, (x, y)))
    return (xc, yc, 0), pieces_loc


def place_captured(captured_pieces, table, piece_style, collection):
    captured_black = [c for c in captured_pieces if c.islower()]
    captured_white = [c for c in captured_pieces if c.isupper()]

    vertices = table.data.vertices
    x_vertices = [(table.matrix_world @ v.co).x for v in vertices]
    y_vertices = [(table.matrix_world @ v.co).y for v in vertices]
    xmin = min(x_vertices) + SQUARE_LENGTH/2
    xmax = max(x_vertices) - SQUARE_LENGTH/2
    yminblack = min(y_vertices) + SQUARE_LENGTH/2
    ymaxblack = -2*SQUARE_LENGTH
    yminwhite = +2*SQUARE_LENGTH
    ymaxwhite = max(y_vertices) - SQUARE_LENGTH/2
    dist_factor = 6

    bcenter, captured_black_loc = place_group(captured_black,
                                              xmin, xmax,
                                              yminblack, ymaxblack,
                                              dist_factor)

    while True:
        wcenter, captured_white_loc = place_group(captured_white,
                                                  xmin, xmax,
                                                  yminwhite, ymaxwhite,
                                                  dist_factor)
        if dist_point(wcenter, bcenter) > 6*SQUARE_LENGTH:
            break

    for piece in captured_black_loc:
        name = PIECES[piece[0]] + str(piece_style)
        add_to_table(name, collection, table, x=piece[1][0], y=piece[1][1])
    for piece in captured_white_loc:
        name = PIECES[piece[0]] + str(piece_style)
        add_to_table(name, collection, table, x=piece[1][0], y=piece[1][1])
    return


def add_to_table(name, collection, table, dist_factor=6, x=0, y=0):

    vertices = table.data.vertices
    z_vertices = [(table.matrix_world @ v.co).z for v in vertices]
    x_vertices = [(table.matrix_world @ v.co).x for v in vertices]
    y_vertices = [(table.matrix_world @ v.co).y for v in vertices]
    z = max(z_vertices)
    xmin = min(x_vertices) + SQUARE_LENGTH/2
    xmax = max(x_vertices) - SQUARE_LENGTH/2
    ymin = min(y_vertices) + SQUARE_LENGTH/2
    ymax = max(y_vertices) - SQUARE_LENGTH/2

    dist = 1000*SQUARE_LENGTH
    i = 0
    if x == 0 and y == 0:
        while True:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            j = 0
            limit = dist_factor*SQUARE_LENGTH
            while abs(x) < limit and abs(y) < limit:
                x = np.random.uniform(xmin, xmax)
                y = np.random.uniform(ymin, ymax)
                j += 1
                if j >= 20:
                    break
            if j >= 20:
                i = 20
                break

            for obj in collection.objects:
                d = dist_point(obj.location, (x, y, z))
                if d < dist:
                    dist = d
            if dist > SQUARE_LENGTH:
                break
            i += 1
            if i >= 20:
                break

    if i < 20:
        source_obj = bpy.data.objects[name]
        obj = source_obj.copy()
        obj.data = source_obj.data.copy()
        obj.animation_data_clear()
        obj.location = (x, y, z)
        rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))
        obj.rotation_euler = rotation
        collection.objects.link(obj)

        obj.hide_render = False
        obj.hide_set(False)
        obj.hide_viewport = False

    return


def dist_obj(obj1, obj2):
    a = obj1.location
    b = obj2.location

    return (a - b).length


def dist_point(P1, P2):
    a = (P1[0] - P2[0])**2 + (P1[1] - P2[1])**2 + (P1[2] - P2[2])**2
    return np.sqrt(a)


def board_box(corners):
    print(f"corners = {corners}")
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


def setup_shot(position, output_file, captured_pieces):
    print(f"setup_shot(position={position.board_fen()},\n",
          f"output_file={output_file},\n",
          f"captured_pieces={captured_pieces})")
    scene = bpy.context.scene

    # Setup rendering
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "JPEG"
    scene.render.filepath = str(output_file)
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT

    styles = {
        "table": np.random.randint(0, TABLE_STYLES),
        "board": np.random.randint(0, BOARD_STYLES),
        "piece": np.random.randint(0, PIECE_STYLES),
    }

    # Create a collection to store the position
    if COLLECTION_NAME not in bpy.data.collections:
        collection = bpy.data.collections.new(COLLECTION_NAME)
        scene.collection.children.link(collection)
    collection = bpy.data.collections[COLLECTION_NAME]

    for obj in bpy.data.objects:
        obj.select_set(False)

    for obj in collection.objects:
        obj.select_set(True)
        bpy.ops.object.delete()

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

    if ADD_PIECES:
        for square, piece in position.piece_map().items():
            obj = add_piece(piece, square, collection, styles['piece'])
            objects.append({
                "piece": piece.symbol(),
                "box": get_bounding_box(scene, obj)
            })

    if ADD_TABLE:
        if ADD_CAPTURED:
            place_captured(captured_pieces, table, styles['piece'], collection)
        if np.random.rand() < 0.5:
            add_to_table("RedCup", collection, table)
        if np.random.rand() < 0.5:
            add_to_table("CoffeCup", collection, table)

    return objects


def get_corner_coordinates(scene):
    corner_points = np.array([[-1., -1],
                              [-1, 1],
                              [1, 1],
                              [1, -1]]) * 4 * SQUARE_LENGTH
    corner_points = np.concatenate((corner_points, np.zeros((4, 1))), axis=-1)
    sr = bpy.context.scene.render

    def _surpass_padding(resolution, p):
        dp = resolution - MIN_BOARD_CORNER_PADDING
        return not (MIN_BOARD_CORNER_PADDING <= p <= dp)

    def _get_coords_corners():
        for corner in corner_points:
            x, y, z = bpy_extras.object_utils.world_to_camera_view(
                scene, scene.camera, mathutils.Vector(corner)).to_tuple()
            y = 1. - y
            x *= sr.resolution_x * sr.resolution_percentage * .01
            y *= sr.resolution_y * sr.resolution_percentage * .01
            x, y = round(x), round(y)

            if _surpass_padding(sr.resolution_x, x) or \
               _surpass_padding(sr.resolution_y, y):
                raise ValueError

            yield x, y
    try:
        return list(_get_coords_corners())
    except ValueError:
        return None


def get_bounding_box(scene, obj):
    """Obtain the bounding box of an object.
    Args:
        scene: the scene
        obj: the object
    Returns:
        the box coordinates in the form (x, y, width, height)
    """
    # adapted from https://blender.stackexchange.com/a/158236
    camera_obj = scene.camera
    mat = camera_obj.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = obj.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(obj.matrix_world)
    me.transform(mat)

    camera = camera_obj.data

    def _get_coords_bounding_box():
        frame = [-v for v in camera.view_frame(scene=scene)[:3]]
        for v in me.vertices:
            co_local = v.co
            z = -co_local.z

            if z <= 0.0:
                print("===========", z, obj, file=sys.stderr)
                continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            yield x, y

    xs, ys = np.array(list(_get_coords_bounding_box())).T

    min_x = np.clip(min(xs), 0.0, 1.0)
    max_x = np.clip(max(xs), 0.0, 1.0)
    min_y = np.clip(min(ys), 0.0, 1.0)
    max_y = np.clip(max(ys), 0.0, 1.0)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    assert round((max_x - min_x) *
                 dim_x) != 0 and round((max_y - min_y) * dim_y) != 0

    corners = (
        round(min_x * dim_x),
        round(dim_y - max_y * dim_y),
        round((max_x - min_x) * dim_x),
        round((max_y - min_y) * dim_y)
    )
    return corners


def get_missing_pieces(fen):
    print(f"get_missing_pieces(fen={fen})")
    pieces = list("KkQqBbBbNnNnRrRrPPPPPPPPpppppppp")
    board = list(''.join(filter(str.isalpha, fen)))
    for piece in board:
        try:
            pieces.remove(piece)
        except ValueError:
            pass
    return pieces


if __name__ == "__main__":
    argv = sys.argv
    print("="*30, f"{argv[0]}.py", "="*30)

    fens_path = Path("fens.txt")
    with fens_path.open("r") as f:
        for i, fen in enumerate(map(str.strip, f)):
            print(f"FEN #{i} = {fen}")
            print(f"FEN #{i} = {fen}", file=sys.stderr)

            set_configs()

            filename = Path("renders") / f"{i:05d}.png"
            position = chess.Board("".join(fen))
            captured_pieces = get_missing_pieces(fen)
            objects = setup_shot(position, filename, captured_pieces)
            if DO_RENDER:
                print(f"rendering {filename}...")
                bpy.ops.render.render(write_still=1)
                if ADD_BOARD:
                    txtpath = filename.parent / (filename.stem + ".txt")
                    dump_yolo_txt(txtpath, objects)

            if i % 100 == 0:
                gc.collect()
                bpy.ops.outliner.orphans_purge()
            break
    print("="*60)
