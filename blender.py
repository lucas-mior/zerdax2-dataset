#!/usr/bin/python
"""Blender script used to generate the synthetic dataset.
"""

import bpy
import bpy_extras.object_utils
import mathutils
import chess
import numpy as np
from pathlib import Path
import typing
import json
import sys

MIN_BOARD_CORNER_PADDING = 25  # pixels
SQ_LEN = 0.25783
COLLECTION_NAME = "ChessPosition"
BOARD_STYLES = 4
TABLE_STYLES = 4
PIECE_STYLES = 4
table_stuff = []


def point_to(obj, focus, roll=0):
    print(f"point_to(obj={obj.name}, focus={focus}, roll={roll})")
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
    z = np.random.normal(14*SQ_LEN, 2*SQ_LEN)
    x = np.random.uniform(-10*SQ_LEN, 10*SQ_LEN)
    dy = min(max(np.random.normal(9*SQ_LEN, 1*SQ_LEN), 7*SQ_LEN), 12*SQ_LEN)
    y = 0.8*abs(x) + dy - 0.3*abs(z)
    if np.random.randint(0, 2) == 1:
        y = -y

    camera.location = (x, y, z)
    board = bpy.data.objects[f"Board{board_style}"]
    point_to(camera, board.location)

    rx = np.random.uniform(-0.00, -0.03)
    ry = np.random.uniform(-0.01, +0.01)
    rz = np.random.uniform(-0.01, +0.01)

    bpy.context.view_layer.update()
    camera.rotation_euler[0] += rx
    camera.rotation_euler[1] += ry
    camera.rotation_euler[2] += rz

    bpy.context.view_layer.update()
    data = {
        "angle_variation": (rx, ry, rz),
        "location": (x, y, z),
    }
    return data


def setup_spotlight(light):
    print(f"setup_spotlight(light={light.name})")
    z = np.random.normal(18*SQ_LEN, 2*SQ_LEN)
    x = np.random.uniform(-18*SQ_LEN, 18*SQ_LEN)
    y = np.random.uniform(-18*SQ_LEN, 18*SQ_LEN)
    location = mathutils.Vector((x, y, z))
    light.location = location
    z = 0.0
    x = np.random.uniform(-5*SQ_LEN, 5*SQ_LEN)
    y = np.random.uniform(-5*SQ_LEN, 5*SQ_LEN)
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
        if i == board_style:
            obj.hide_render = False
            obj.hide_viewport = False
            obj.hide_set(False)
        else:
            obj.hide_render = True
            obj.hide_viewport = True
            obj.hide_set(True)

    bpy.context.view_layer.update()
    return


def setup_sun():
    print("setup_sun()")
    strength = np.random.uniform(0.3, 0.4)
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
            sun: False,
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

    data = {
        "mode": mode,
        "flash": {
            "active": not flash.hide_render
        },
        "sun": {
            "active": not sun.hide_render,
            "strenth": setup_sun(),
        },
        **{
            key: {
                "active": not obj.hide_render,
                **setup_spotlight(obj)
            } for (key, obj) in {"spot1": spot1, "spot2": spot2}.items()
        }
    }
    return data


def add_piece(piece, square, collection, piece_style):
    print(f"add_piece(piece={piece}, square={square},",
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
    offsets = np.clip(offsets, .3, .7)
    rank_offset, file_offset = offsets
    rank = chess.square_rank(square) + rank_offset
    file = chess.square_file(square) + file_offset

    # Translate to coordinate system where the origin is in the middle of the
    # board
    rank -= 4
    file -= 4

    location = mathutils.Vector((file, rank, 0)) * SQ_LEN
    rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))

    src_obj = bpy.data.objects[name]
    obj = src_obj.copy()
    obj.data = src_obj.data.copy()
    obj.animation_data_clear()
    obj.location = location
    obj.rotation_euler = rotation
    collection.objects.link(obj)
    return obj


def place_group(group, xmin, xmax, ymin, ymax):
    print(f"place_group(group={group},",
          f"xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax})")
    pieces_loc = []
    xcenter = np.random.uniform(xmin, xmax)
    ycenter = np.random.uniform(ymin, ymax)
    while abs(xcenter) < 6*SQ_LEN and abs(ycenter) < 6*SQ_LEN:
        xcenter = np.random.uniform(xmin, xmax)
        ycenter = np.random.uniform(ymin, ymax)
    for piece in group:
        x = np.random.uniform(xcenter-1*SQ_LEN, xcenter+1*SQ_LEN)
        y = np.random.uniform(ycenter-6*SQ_LEN, ycenter+6*SQ_LEN)
        if xcenter > ycenter:
            while abs(x) < 6*SQ_LEN and abs(y) < 6*SQ_LEN:
                x = np.random.uniform(xcenter-1*SQ_LEN, xcenter+1*SQ_LEN)
                y = np.random.uniform(ycenter-6*SQ_LEN, ycenter+6*SQ_LEN)
        else:
            while abs(x) < 6*SQ_LEN and abs(y) < 6*SQ_LEN:
                y = np.random.uniform(ycenter-1*SQ_LEN, ycenter+1*SQ_LEN)
                x = np.random.uniform(xcenter-6*SQ_LEN, xcenter+6*SQ_LEN)
        pieces_loc.append((piece, (x, y)))
    return (xcenter, ycenter, 0), pieces_loc


def place_captured(cap_pieces, piece_style, collection, table_style):
    print(f"place_captured(cap_piece={cap_pieces},",
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
    cap_black = [c for c in cap_pieces if c.islower()]
    cap_white = [c for c in cap_pieces if c.isupper()]

    bcenter, cap_black_loc = place_group(cap_black,
                                         xmin=-9*SQ_LEN, xmax=9*SQ_LEN,
                                         ymin=-9*SQ_LEN, ymax=-4*SQ_LEN)
    while True:
        wcenter, cap_white_loc = place_group(cap_white,
                                             xmin=-9*SQ_LEN, xmax=9*SQ_LEN,
                                             ymin=4*SQ_LEN, ymax=+9*SQ_LEN)
        if dist_point(wcenter, bcenter) > 8*SQ_LEN:
            break

    for piece in cap_black_loc:
        name = piece_names[piece[0]] + str(piece_style)
        add_to_table(name, collection, table_style,
                     dfact=6, x=piece[1][0], y=piece[1][1])
    for piece in cap_white_loc:
        name = piece_names[piece[0]] + str(piece_style)
        add_to_table(name, collection, table_style,
                     dfact=6, x=piece[1][0], y=piece[1][1])
    return


def add_to_table(name, collection, table_style, dfact=6, x=0, y=0):
    print(f"add_to_table(name={name}, collection={collection.name},",
          f"table_style={table_style}, dfact={dfact}, x={x}, y={y})")
    src_obj = bpy.data.objects[name]
    obj = src_obj.copy()
    obj.data = src_obj.data.copy()
    obj.animation_data_clear()

    table = bpy.data.objects[f'Table{table_style}']
    z = max([(table.matrix_world @ v.co).z for v in table.data.vertices])

    dist = 1000*SQ_LEN
    if x == 0 and y == 0:
        while True:
            x = np.random.uniform(-12*SQ_LEN, 12*SQ_LEN)
            y = np.random.uniform(-12*SQ_LEN, 12*SQ_LEN)
            while abs(x) < dfact*SQ_LEN and abs(y) < dfact*SQ_LEN:
                x = np.random.uniform(-12*SQ_LEN, 12*SQ_LEN)
                y = np.random.uniform(-12*SQ_LEN, 12*SQ_LEN)
            for obj_name in table_stuff:
                d = dist_obj(bpy.data.objects[obj_name], obj)
                if d < dist:
                    dist = d
            if dist > SQ_LEN/2:
                break
    else:
        pass

    rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))
    obj.location = (x, y, z)
    obj.rotation_euler = rotation
    collection.objects.link(obj)
    table_stuff.append(obj.name)
    return


def dist_obj(obj1, obj2):
    print(f"dist_obj({obj1.name}, {obj2.name})")
    a = obj1.location
    b = obj2.location

    return (a - b).length


def dist_point(P1, P2):
    print(f"dist_point({P1}, {P2})")
    a = (P1[0] - P2[0])**2 + (P1[1] - P2[1])**2 + (P1[2] - P2[2])**2
    return np.sqrt(a)


def render_board(board, output_file, cap_pieces, do_render):
    print(f"render_board(board={board}, output_file={output_file},",
          f"cap_pieces={cap_pieces}, do_render={do_render})")
    scene = bpy.context.scene

    # Setup rendering
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_file)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800

    board_style = np.random.randint(1, BOARD_STYLES)
    table_style = np.random.randint(1, TABLE_STYLES)
    corner_coords = None
    while not corner_coords:
        camera_params = setup_camera(board_style)
        lighting_params = setup_lighting()
        corner_coords = get_corner_coordinates(scene)
        setup_board(board_style)
        setup_table(table_style, board_style)

    corner_coords = sorted(corner_coords, key=lambda x: x[0])

    # Create a collection to store the position
    if COLLECTION_NAME not in bpy.data.collections:
        collection = bpy.data.collections.new(COLLECTION_NAME)
        scene.collection.children.link(collection)
    collection = bpy.data.collections[COLLECTION_NAME]

    # Remove all objects from the collection
    bpy.ops.object.delete({"selected_objects": collection.objects})

    piece_data = []
    table_stuff.clear()
    piece_amount = 0
    piece_style = np.random.randint(1, PIECE_STYLES)
    for square, piece in board.piece_map().items():
        obj = add_piece(piece, square, collection, piece_style)
        piece_data.append({
            "piece": piece.symbol(),
            "square": chess.square_name(square),
            "box": get_bounding_box(scene, obj)
        })
        piece_amount += 1

    place_captured(cap_pieces, piece_style, collection, table_style)
    add_to_table("RedCup", collection, table_style, dfact=7)

    # Write data output
    data = {
        "piece_amount": piece_amount,
        "fen": board.board_fen(),
        "camera": camera_params,
        "lighting": lighting_params,
        "corners": corner_coords,
        "pieces": piece_data,
        "table_stuff": table_stuff,
    }
    if do_render:
        print(f"rendering {output_file}...")
        jsonpath = output_file.parent / (output_file.stem + ".json")
        with jsonpath.open("w") as f:
            json.dump(data, f, indent=4)
        bpy.ops.render.render(write_still=1)
    return


def get_corner_coordinates(scene) -> typing.List[typing.List[int]]:
    print("get_corner_coordinates(scene) -> typing.List[typing.List[int]]:")
    corner_points = np.array([[-1., -1],
                              [-1, 1],
                              [1, 1],
                              [1, -1]]) * 4 * SQ_LEN
    corner_points = np.concatenate((corner_points, np.zeros((4, 1))), axis=-1)
    sr = bpy.context.scene.render

    def _get_coords():
        print("_get_coords()")
        for corner in corner_points:
            x, y, z = bpy_extras.object_utils.world_to_camera_view(
                scene, scene.camera, mathutils.Vector(corner)).to_tuple()
            y = 1. - y
            x *= sr.resolution_x * sr.resolution_percentage * .01
            y *= sr.resolution_y * sr.resolution_percentage * .01
            x, y = round(x), round(y)

            if not (MIN_BOARD_CORNER_PADDING <= x <= sr.resolution_x - MIN_BOARD_CORNER_PADDING) or \
                    not (MIN_BOARD_CORNER_PADDING <= y <= sr.resolution_y - MIN_BOARD_CORNER_PADDING):
                raise ValueError

            yield x, y
    try:
        return list(_get_coords())
    except ValueError:
        return None


def get_bounding_box(scene, obj) -> typing.Tuple[int, int, int, int]:
    print("get_bounding_box(scene, obj) -> typing.Tuple[int, int, int, int]:")
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

    def _get_coords():
        print("_get_coords()")
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

    xs, ys = np.array(list(_get_coords())).T

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
        int(round(min_x * dim_x)),
        int(round(dim_y - max_y * dim_y)),
        int(round((max_x - min_x) * dim_x)),
        int(round((max_y - min_y) * dim_y))
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
    print(f"running script {sys.argv[0]} ...")
    fens_path = Path("fens.txt")
    with fens_path.open("r") as f:
        for i, fen in enumerate(map(str.strip, f)):
            if 2002 <= i <= 2002:
                print(f"FEN #{i} = {fen}")
                print(f"FEN #{i} = {fen}", file=sys.stderr)
                filename = Path("render") / f"{i:05d}.png"
                board = chess.Board("".join(fen))
                cap_pieces = get_missing_pieces(fen)
                render_board(board, filename, cap_pieces, True)
            else:
                pass
