# HandSimulator

We use the MANO hand model, the HTML texture model and hand-selected or randomized backgrounds to render images of two articulated hands.

We use OpenGL as rendering library and output raw images for simulating events as recorded by event cameras. Alongside these images, we also output data like segmentations, depth maps, normal maps and perhaps more in the future.

A sizeable portion of this project is a Python translation of this [C++ code base](https://github.com/opengl-tutorials/ogl).

## Usage

### Requirements
Requirements include but are not limited to:
* NumPy
* PyOpenGL
* GLFW
* PyGLM
* OpenCV
* the [MANO](https://mano.is.tue.mpg.de/) hand model and [SMPLX](https://github.com/vchoutas/smplx)

### Rendering
To render a sequence make sure the following components are set up as required:
* SMPLX is placed in model/ and the required mano.py as well as the MANO frame sequences are placed in model/examples
* objects/, textures/ uvs/ contain the required files
* your desired render settings are specified in settings.py
