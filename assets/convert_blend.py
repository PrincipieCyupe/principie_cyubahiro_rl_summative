"""Turns Blender files (.blend) into .glb files that the 3D browser view can load.

Browsers cannot open .blend files, so the crocodile and hippo models have to be
converted once. Put the .blend files anywhere inside assets/ and run:

    uv run --with bpy assets/convert_blend.py

"--with bpy" borrows Blender's Python module for this one command only, it is not
added to the project. The results are saved as assets/meshes/crocodile.glb and
assets/meshes/hippo.glb (a file is named after the animal in its file name).
"""

import sys
from pathlib import Path

import bpy

ASSETS_DIR = Path(__file__).resolve().parent
MESH_DIR = ASSETS_DIR / "meshes"


def output_name(path):
    name = path.stem.lower()
    if "croc" in name:
        return "crocodile"
    if "hip" in name:
        return "hippo"
    return name


def convert(path):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    target = MESH_DIR / f"{output_name(path)}.glb"
    bpy.ops.export_scene.gltf(filepath=str(target), export_format="GLB", export_apply=True)
    print(f"{path.name} -> {target.relative_to(ASSETS_DIR.parent)}")


if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]] or sorted(ASSETS_DIR.rglob("*.blend"))
    if not files:
        print("No .blend files found inside assets/.")
    MESH_DIR.mkdir(exist_ok=True)
    for blend in files:
        convert(blend)
