from OpenGL.GL import *

# list of rendering settings
buffer_data_usage = GL_DYNAMIC_DRAW     # for non-rigid objects
coordinate_system = 'world'             # <'camera', 'world'>
far = 1                                 # distance to far-plane in meters
fov = 45.0                              # field of view in degrees
fps_in = 1000                           # input fps (for OUTPUT_DISK_FORMAT 'video')
fps_out = 1000                          # output fps (for OUTPUT_DISK_FORMAT 'video')
LIMIT_FPS = False                       # double buffer VSync
NUM_FRAMES = float('inf')               # infinite -> number of frames in the sequence
near = 0.1                              # distance to near-plane in meters
res = (240, 180)                        # (width, height)
OUTPUTS = 'ooxx'                        # o (yes) and x (no) for [RGB, segmentation, depth, normal]
OUTPUT_DISK_FORMAT = 'video'            # <'images', 'video'>
OUTPUT_MODE = 'both'                    # <'monitor', 'disk', 'both'>
OUTPUT_VIDEO_FORMAT = 'lossless'        # <'lossless', 'lossy'>
SHADING = 'basic'                       # <'plain', 'basic', 'shadow_mapping'>
USE_VBO_INDEXING = False                # only use for rigid objects
