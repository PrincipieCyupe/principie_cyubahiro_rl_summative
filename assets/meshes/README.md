Optional 3D models for the browser view. These are artwork, not trained agents (those go in models/).

Drop in .glb files named guardian.glb, crocodile.glb or hippo.glb and refresh the page.
They are scaled to fit a grid cell and stood on the ground automatically, and the first
animation in the file is played while they walk. If a model faces the wrong way, change
MESH_TURN at the top of assets/web/world.js. Without a file the built in models are used.

Free CC0 models from OpenGameArt that would fit (they are .blend files, so open them in
Blender and use File > Export > glTF 2.0 (.glb)):
  https://opengameart.org/content/crocodile
  https://opengameart.org/node/17046  (hippo)
