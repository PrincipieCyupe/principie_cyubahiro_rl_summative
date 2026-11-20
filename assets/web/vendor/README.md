Third party code used by the 3D page. None of it was written for this project.

All files are from three.js version 0.170.0 (MIT licence, see LICENSE in this folder).
In the three.js package the extra modules live in a folder called "examples/jsm".
three.js calls them "addons", so here they are kept in a folder named addons/.

    file here                              where it is in the three.js package
    three.module.min.js                    build/three.module.min.js
    addons/objects/Sky.js                  examples/jsm/objects/Sky.js
    addons/objects/Water.js                examples/jsm/objects/Water.js
    addons/loaders/GLTFLoader.js           examples/jsm/loaders/GLTFLoader.js
    addons/controls/OrbitControls.js       examples/jsm/controls/OrbitControls.js
    addons/utils/BufferGeometryUtils.js    examples/jsm/utils/BufferGeometryUtils.js
    addons/utils/SkeletonUtils.js          examples/jsm/utils/SkeletonUtils.js
    ../textures/waternormals.jpg           examples/textures/waternormals.jpg (GitHub repo only)

How to get them yourself (any one way works):
1. Browse the package online and save each file:
       https://cdn.jsdelivr.net/npm/three@0.170.0/
   Click through the folders, open a file, and save it with Ctrl+S. For example:
       https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/objects/Sky.js
2. Download the whole three.js release as a zip and copy the files out of it:
       https://github.com/mrdoob/three.js/archive/refs/tags/r170.zip
3. With Node.js installed: npm install three@0.170.0, then copy them from node_modules/three/.

Why the files are copied here instead of loaded from the internet:
- the page works offline, so it runs for anyone who clones the repo, even without internet
- the version is fixed, so a new three.js release can never break the page
- no extra tools (npm, a build step) are needed, `uv run main.py` is enough
