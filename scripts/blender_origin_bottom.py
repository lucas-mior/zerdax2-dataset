import bpy
import copy

for obj in bpy.context.selected_objects:
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
    oldlocation = copy.copy(obj.location)

    obj.location = (0, 0, 0)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True)

    bpy.context.view_layer.objects.active = obj
    min_z = max((obj.matrix_world @ v.co).z for v in obj.data.vertices)

    bpy.context.scene.cursor.location = (0, 0, min_z)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

    obj.location = oldlocation
    obj.location[2] = 0