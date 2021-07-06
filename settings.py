from OpenGL.GL import *

# list of rendering settings
buffer_data_usage = GL_DYNAMIC_DRAW     # for non-rigid objects
far = 1                                 # distance to far-plane in meters
fov = 45.0                              # field of view in degrees
LIMIT_FPS = False                       # double buffer VSync
NUM_FRAMES = float('inf')               # infinite -> number of frames in the sequence
near = 0.1                              # distance to near-plane in meters
res = (240, 180)                        # (width, height)
OUTPUT_MODE = 'files'                   # <'monitor', 'files', 'both'>
SHADING = 'basic'                       # <'plain', 'basic', 'shadow_mapping'>
USE_VBO_INDEXING = False                # only use for rigid objects
