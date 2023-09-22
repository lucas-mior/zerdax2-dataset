import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

stl_in = argv[0]
outfile = argv[1]

bpy.ops.import_mesh.stl(filepath=stl_in, axis_forward='-Z', axis_up='Y')
bpy.ops.wm.save_as_mainfile(filepath=outfile)
