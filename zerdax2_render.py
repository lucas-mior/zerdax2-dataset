#!/usr/bin/python
"""Blender script used to generate the synthetic dataset.
"""

import bpy
import bpy_extras.object_utils
import os
import sys
from pathlib import Path
import numpy as np
import builtins as __builtin__
import gc

sys.path.append("/home/lucas/.local/lib/python3.10/site-packages")
import mathutils
import chess
import json

d = os.path.dirname(bpy.data.filepath)
if d not in sys.path:
    sys.path.append(d)
from zerdax2_misc import CLASSES


DEBUG = False
DO_RENDER = True
DO_JSON = False
MIN_BOARD_CORNER_PADDING = 30  # pixels
SQUARE_LENGTH = 0.259
COLLECTION_NAME = "ChessPosition"
BOARD_STYLES = 7
TABLE_STYLES = 5
PIECE_STYLES = 7
table_stuff = []


def console_print(*args, **kwargs):
    for a in bpy.context.screen.areas:
        if a.type == 'CONSOLE':
            with bpy.context.temp_override(window=bpy.context.window,
                                           area=a,
                                           region=a.regions[-1],
                                           space_data=a.spaces.active,
                                           screen=bpy.context.screen):
                s = " ".join([str(arg) for arg in args])
                for line in s.split("\n"):
                    bpy.ops.console.scrollback_append(text=line)
    return


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def print(*args, **kwargs):
    console_print(*args, **kwargs)
    __builtin__.print(*args, **kwargs)
    return


def point_to(obj, focus, roll=0):
    debug_print(f"point_to(obj={obj.name}, focus={focus}, roll={roll})")
    # Based on https://blender.stackexchange.com/a/127440
    loc = obj.location
    direction = focus - loc
    quat = direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    roll_matrix = mathutils.Matrix.Rotation(roll, 4, "Z")
    loc = loc.to_tuple()
    obj.matrix_world = quat @ roll_matrix
    obj.location = loc
    return


def setup_camera(board_style):
    print(f"setup_camera(board_style={board_style})")
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
        board = bpy.data.objects[f"Board{board_style}"]
        point_to(camera, board.location)

        v = np.array([x, y, z])
        w = np.array([0, 0, 1])
        dot = np.dot(v, w)
        modulo = np.sqrt(x**2 + y**2 + z**2)
        angle = np.degrees(np.arcsin(dot/modulo))
        print(f"Camera to table angle: {angle:.2f}")

    if x <= -4*SQUARE_LENGTH:
        perspective = "left" if y >= 0 else "right"
    elif -4*SQUARE_LENGTH < x < +4*SQUARE_LENGTH:
        perspective = "center"
    else:
        perspective = "right" if y <= 0 else "left"

    rx = np.random.uniform(-0.01, -0.05)
    ry = np.random.uniform(-0.02, +0.02)
    rz = np.random.uniform(-0.02, +0.02)

    bpy.context.view_layer.update()
    camera.rotation_euler[0] += rx
    camera.rotation_euler[1] += ry
    camera.rotation_euler[2] += rz

    bpy.context.view_layer.update()
    data = {
        "perspective": perspective,
        "angle_variation": (rx, ry, rz),
        "location": (x, y, z),
    }
    return data


def setup_spotlight(light):
    print(f"setup_spotlight(light={light.name})")
    z = np.random.normal(16*SQUARE_LENGTH, 2*SQUARE_LENGTH)
    z = np.clip(z, 13*SQUARE_LENGTH, 22*SQUARE_LENGTH)
    if light.name == "Spot1":
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
    data = {
        "focus": focus.to_tuple(),
        "location": location.to_tuple()
    }
    return data


def setup_table(table_style, board_style):
    print(f"setup_table(style={table_style}, board_style={board_style})")
    for i in range(1, TABLE_STYLES):
        obj = bpy.data.objects[f"Table{i}"]
        if i == table_style:
            obj.hide_render = False
            obj.hide_viewport = False
            obj.hide_set(False)
        else:
            obj.hide_render = True
            obj.hide_viewport = True
            obj.hide_set(True)
    table = bpy.data.objects[f'Table{table_style}']
    board = bpy.data.objects[f'Board{board_style}']
    bmin = min([(board.matrix_world @ v.co).z for v in board.data.vertices])
    table.location[2] = bmin

    bpy.context.view_layer.update()
    return


def setup_board(board_style):
    print(f"setup_board(board_style={board_style})")
    for i in range(1, BOARD_STYLES):
        obj = bpy.data.objects[f"Board{i}"]
        if i == board_style and ADD_BOARD:
            obj.hide_render = False
            obj.hide_viewport = False
            obj.hide_set(False)
        else:
            obj.hide_render = True
            obj.hide_viewport = True
            obj.hide_set(True)

    bpy.context.view_layer.update()
    return obj


def setup_sun():
    print("setup_sun()")
    strength = np.random.uniform(0.4, 0.5)
    bpy.data.lights['Sun'].energy = strength
    return strength


def setup_lighting():
    print("setup_lighting()")
    flash = bpy.data.objects["CameraFlashLight"]
    spot1 = bpy.data.objects["Spot1"]
    spot2 = bpy.data.objects["Spot2"]
    sun = bpy.data.objects["Sun"]

    modes = {
        "flash": {
            flash: True,
            spot1: False,
            spot2: False,
            sun: True,
        },
        "spotlights": {
            flash: False,
            spot1: True,
            spot2: True,
            sun: True,
        }
    }
    mode, visibilities = list(modes.items())[np.random.randint(len(modes))]

    for obj, visibility in visibilities.items():
        obj.hide_render = not visibility
        obj.hide_set(not visibility)
        obj.hide_viewport = not visibility

    return


def in_square(x, y, d):
    if abs(x) < (d*SQUARE_LENGTH) and abs(y) < (d*SQUARE_LENGTH):
        return True
    else:
        return False


def dump_yolo_txt(txtpath):
    print(f"dumping txt {txtpath}...")
    with txtpath.open("w") as txt:
        for obj in data['pieces']:
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
            yc, dy = yc/HEIGTH, dy/HEIGTH

            yolo = f"{number} {xc} {yc} {dx} {dy}\n"
            print(yolo, end="")
            txt.write(yolo)
        txt.close()
    return


def add_piece(piece, square, collection, piece_style):
    debug_print(f"add_piece(piece={piece}, square={square},",
                f"collection={collection.name}, piece_style={piece_style})")
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

    src_obj = bpy.data.objects[name]
    obj = src_obj.copy()
    obj.data = src_obj.data.copy()
    obj.animation_data_clear()
    obj.location = location
    obj.rotation_euler = rotation
    collection.objects.link(obj)
    return obj


def place_group(group, xmin, xmax, ymin, ymax, dfact=6):
    print(f"place_group(group={group},",
          f"xmin={xmin/SQUARE_LENGTH:.2f}, xmax={xmax/SQUARE_LENGTH:.2f},",
          f"ymin={ymin/SQUARE_LENGTH:.2f}, ymax={ymax/SQUARE_LENGTH:.2f})")
    pieces_loc = []
    xcenter = np.random.uniform(xmin, xmax)
    ycenter = np.random.uniform(ymin, ymax)
    print(f"center = ({xcenter/SQUARE_LENGTH}, {ycenter/SQUARE_LENGTH})")
    while abs(xcenter) < dfact*SQUARE_LENGTH and abs(ycenter) < dfact*SQUARE_LENGTH:
        xcenter = np.random.uniform(xmin, xmax)
        ycenter = np.random.uniform(ymin, ymax)

    for piece in group:
        x = 1000
        y = 1000
        dist = 0
        i = 0
        if abs(xcenter) > abs(ycenter):
            while (abs(x) < dfact*SQUARE_LENGTH and abs(y) < dfact*SQUARE_LENGTH) or dist < SQUARE_LENGTH/2:
                dist = 1000
                x = np.random.normal(xcenter, 2*SQUARE_LENGTH)
                y = np.random.normal(ycenter, 4*SQUARE_LENGTH)
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
            while (abs(x) < dfact*SQUARE_LENGTH and abs(y) < dfact*SQUARE_LENGTH) or dist < SQUARE_LENGTH/2:
                dist = 1000
                x = np.random.normal(xcenter, 4*SQUARE_LENGTH)
                y = np.random.normal(ycenter, 2*SQUARE_LENGTH)
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
    return (xcenter, ycenter, 0), pieces_loc


def place_captured(captured_pieces, piece_style, collection, table_style, board_style):
    print(f"place_captured(captured_pieces={captured_pieces},",
          f"piece_style={piece_style}, collection={collection.name},",
          f"table_style={table_style})")
    piece_names = {
        "K": "WhiteKing",
        "Q": "WhiteQueen",
        "B": "WhiteBishop",
        "N": "WhiteKnight",
        "R": "WhiteRook",
        "P": "WhitePawn",
        "k": "BlackKing",
        "q": "BlackQueen",
        "b": "BlackBishop",
        "n": "BlackKnight",
        "r": "BlackRook",
        "p": "BlackPawn",
    }
    cap_black = [c for c in captured_pieces if c.islower()]
    cap_white = [c for c in captured_pieces if c.isupper()]
    table = bpy.data.objects[f'Table{table_style}']

    xvertices = [(table.matrix_world @ v.co).x for v in table.data.vertices]
    yvertices = [(table.matrix_world @ v.co).y for v in table.data.vertices]
    xmin = min(xvertices) + SQUARE_LENGTH/2
    xmax = max(xvertices) - SQUARE_LENGTH/2
    yminblack = min(yvertices) + SQUARE_LENGTH/2
    ymaxblack = -2*SQUARE_LENGTH
    yminwhite = +2*SQUARE_LENGTH
    ymaxwhite = max(yvertices) - SQUARE_LENGTH/2
    if board_style == 3:
        dfact = 7
    else:
        dfact = 6

    bcenter, cap_black_loc = place_group(cap_black,
                                         xmin=xmin, xmax=xmax,
                                         ymin=yminblack, ymax=ymaxblack,
                                         dfact=dfact)

    while True:
        wcenter, cap_white_loc = place_group(cap_white,
                                             xmin=xmin, xmax=xmax,
                                             ymin=yminwhite, ymax=ymaxwhite,
                                             dfact=dfact)
        if dist_point(wcenter, bcenter) > 6*SQUARE_LENGTH:
            break

    for piece in cap_black_loc:
        name = piece_names[piece[0]] + str(piece_style)
        add_to_table(name, collection, table_style,
                     dfact=7, x=piece[1][0], y=piece[1][1])
    for piece in cap_white_loc:
        name = piece_names[piece[0]] + str(piece_style)
        add_to_table(name, collection, table_style,
                     dfact=7, x=piece[1][0], y=piece[1][1])
    return


def add_to_table(name, collection, table_style, dfact=6, x=0, y=0):
    debug_print(f"add_to_table(name={name}, collection={collection.name},",
                f"table_style={table_style}, dfact={dfact}, x={x:.2f}, y={y:.2f})")

    rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))

    table = bpy.data.objects[f'Table{table_style}']

    zvertices = [(table.matrix_world @ v.co).z for v in table.data.vertices]
    xvertices = [(table.matrix_world @ v.co).x for v in table.data.vertices]
    yvertices = [(table.matrix_world @ v.co).y for v in table.data.vertices]
    z = max(zvertices)
    xmin = min(xvertices) + SQUARE_LENGTH/2
    xmax = max(xvertices) - SQUARE_LENGTH/2
    ymin = min(yvertices) + SQUARE_LENGTH/2
    ymax = max(yvertices) - SQUARE_LENGTH/2

    dist = 1000*SQUARE_LENGTH
    i = 0
    if x == 0 and y == 0:
        while True:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            j = 0
            while abs(x) < dfact*SQUARE_LENGTH and abs(y) < dfact*SQUARE_LENGTH:
                x = np.random.uniform(xmin, xmax)
                y = np.random.uniform(ymin, ymax)
                j += 1
                if j >= 20:
                    break
            if j >= 20:
                i = 20
                break

            for obj_name in table_stuff:
                d = dist_point(bpy.data.objects[obj_name].location, (x, y, z))
                if d < dist:
                    dist = d
            if dist > SQUARE_LENGTH:
                break
            i += 1
            if i >= 20:
                break

    if i < 20:
        src_obj = bpy.data.objects[name]
        obj = src_obj.copy()
        obj.data = src_obj.data.copy()
        obj.animation_data_clear()
        obj.location = (x, y, z)
        obj.rotation_euler = rotation
        collection.objects.link(obj)

        obj.hide_render = False
        obj.hide_set(False)
        obj.hide_viewport = False

        table_stuff.append(obj.name)
    return


def dist_obj(obj1, obj2):
    debug_print(f"dist_obj({obj1.name}, {obj2.name})")
    a = obj1.location
    b = obj2.location

    return (a - b).length


def dist_point(P1, P2):
    debug_print("dist_point(", end=' ')
    debug_print(f"({P1[0]:.2f}, {P1[1]:.2f}, {P1[2]:.2f}), ",
                f"({P2[0]:.2f}, {P2[1]:.2f}, {P1[2]:.2f}))", sep='')
    a = (P1[0] - P2[0])**2 + (P1[1] - P2[1])**2 + (P1[2] - P2[2])**2
    return np.sqrt(a)


def board_box(corners):
    print(f"corners = {corners}")
    x = [c[0] for c in corners]
    y = [c[1] for c in corners]
    cornersx = sorted(x)
    cornersy = sorted(y)

    x0, x1 = cornersx[0], cornersx[3]
    xc = round((x0+x1)/2)
    dx = x1 - x0

    y0, y1 = cornersy[0], cornersy[3]
    yc = round((y0+y1)/2)
    dy = y1 - y0

    print(f"0 {xc} {yc} {dx} {dy}")
    box = [xc, dx, yc, dy]
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
    scene.render.resolution_y = HEIGTH

    board_style = np.random.randint(1, BOARD_STYLES)
    table_style = np.random.randint(1, TABLE_STYLES)
    corner_coords = None
    while not corner_coords:
        camera_params = setup_camera(board_style)
        corner_coords = get_corner_coordinates(scene)

    setup_lighting()
    board = setup_board(board_style)
    setup_table(table_style, board_style)

    corner_coords = sorted(corner_coords, key=lambda x: x[0])

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

    piece_data = []

    if ADD_BOARD:
        piece_data.append({
            "piece": "Board",
            "square": None,
            "box": get_bounding_box(scene, board)
        })

    table_stuff.clear()
    piece_amount = 0
    piece_style = np.random.randint(1, PIECE_STYLES)
    if ADD_PIECES:
        for square, piece in position.piece_map().items():
            obj = add_piece(piece, square, collection, piece_style)
            piece_data.append({
                "piece": piece.symbol(),
                "square": chess.square_name(square),
                "box": get_bounding_box(scene, obj)
            })
            piece_amount += 1

    # if np.random.randint(0, 2) == 1:
    place_captured(captured_pieces, piece_style, collection, table_style, board_style)
    if np.random.randint(0, 2) == 1:
        add_to_table("RedCup", collection, table_style, dfact=7)
    if np.random.randint(0, 2) == 1:
        add_to_table("CoffeCup", collection, table_style, dfact=8)

    # styles = {
    #     "table": table_style,
    #     "board": board_style,
    #     "piece": piece_style,
    # }
    if ADD_PIECES:
        fen_save = position.board_fen()
    else:
        fen_save = "8/8/8/8/8/8/8/8"

    # Write data output
    data = {
        "corners": corner_coords,
        "board_box": board_box(corner_coords),
        "pieces": piece_data,
        "fen": fen_save,
        "camera": camera_params,
    }
    return data


def get_corner_coordinates(scene):
    debug_print("get_corner_coordinates(scene)")
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
        debug_print("_get_coords_corners()")
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
    debug_print(f"get_bounding_box({scene.name}, {obj.name})")
    """Obtain the bounding box of an object.

    Args:
        scene: the scene
        obj: the object

    Returns:
        typing.Tuple[int, int, int, int]:
        the box coordinates in the form (x, y, width, height)
    """
    # adapted from https://blender.stackexchange.com/a/158236
    cam_ob = scene.camera
    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = obj.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(obj.matrix_world)
    me.transform(mat)

    camera = cam_ob.data

    def _get_coords_bounding_box():
        debug_print("_get_coords_bounding_box()")
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
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
        begin = int(argv[0])
        increment = int(argv[1])
    else:
        begin = 0
        increment = 1

    fens_path = Path("fens.txt")
    with fens_path.open("r") as f:
        for i, fen in enumerate(map(str.strip, f)):
            if i % 5 != 0:
                continue
            rand_num = np.random.randint(1, 1000)
            # rand_num = i
            if rand_num % 2 == 0:
                WIDTH = 960
                HEIGTH = 600
            else:
                WIDTH = 600
                HEIGTH = 960
            if rand_num % 11 == 0:
                ADD_BOARD = False
                ADD_PIECES = False
            elif rand_num % 23 == 0:
                ADD_BOARD = True
                ADD_PIECES = False
            else:
                ADD_BOARD = True
                ADD_PIECES = True

            print(f"FEN #{i} = {fen}")
            print(f"FEN #{i} = {fen}", file=sys.stderr)
            if ADD_BOARD and ADD_PIECES:
                mode = "board_and_pieces"
            elif ADD_BOARD:
                mode = "board_only"
            else:
                mode = "background_only"
            filename = Path("renders") / mode / f"{i:05d}.png"
            position = chess.Board("".join(fen))
            captured_pieces = get_missing_pieces(fen)
            data = setup_shot(position, filename, captured_pieces)
            if DO_RENDER:
                if DO_JSON:
                    jsonpath = filename.parent / (filename.stem + ".json")
                    print(f"dumping json {jsonpath}...")
                    with jsonpath.open("w") as f:
                        json.dump(data, f, indent=4)
                        f.close()

                if ADD_BOARD:
                    txtpath = filename.parent / (filename.stem + ".txt")
                    dump_yolo_txt(txtpath)

                print(f"rendering {filename}...")
                bpy.ops.render.render(write_still=1)
            if i % 100 == 0:
                gc.collect()
                bpy.ops.outliner.orphans_purge()
    print("="*60)
