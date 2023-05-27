#!/usr/bin/python
"""Blender script used to generate the synthetic dataset.
"""

import bpy
import numpy as np
import builtins as __builtin__
import mathutils
import sys
from mathutils import Vector


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


def min_distance_point(obj_name, point):
    # Get reference to the object
    obj = bpy.data.objects.get(obj_name)

    if obj is None:
        print("Object not found.")
        return float('inf')

    # Get the mesh data of the object
    mesh = obj.data

    # Transform the vertices of the object into global coordinates
    obj_vertices = [obj.matrix_world @ Vector(v.co) for v in mesh.vertices]

    # Calculate the minimum distance
    min_distance = float('inf')
    point = Vector(point)
    for v in obj_vertices:
        distance = (v - point).length
        if distance < min_distance:
            min_distance = distance

    return min_distance


def min_distance_object(obj1, obj2):
    # Get references to the objects

    if obj1 is None or obj2 is None:
        print("One or both objects dont exist.")
        return float('inf')

    # Get the mesh data of the objects
    mesh1 = obj1.data
    mesh2 = obj2.data

    # Transform the vertices of the objects into global coordinates
    obj1_vertices = [obj1.matrix_world @ Vector(v.co) for v in mesh1.vertices]
    obj2_vertices = [obj2.matrix_world @ Vector(v.co) for v in mesh2.vertices]

    # Calculate the minimum distance
    min_distance = float('inf')
    for v1 in obj1_vertices:
        for v2 in obj2_vertices:
            distance = (v1 - v2).length
            if distance < min_distance:
                min_distance = distance

    return min_distance


def distance_points(P1, P2):
    dx = P1[0] - P2[0]
    dy = P1[1] - P2[1]
    dz = P1[2] - P2[2]
    return np.sqrt(dx*dx + dy*dy + dz*dz)


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