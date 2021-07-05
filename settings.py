from OpenGL.GL import *

# list of rendering settings
buffer_data_usage = GL_DYNAMIC_DRAW
far = 1
fov = 45.0
LIMIT_FPS = False
NUM_FRAMES = 300000
near = 0.1
res = (240, 180)
max_xy = int(round(max(res[0], res[1])))
OUTPUT_MODE = 'monitor'
SHADING = 'basic'
USE_VBO_INDEXING = False
