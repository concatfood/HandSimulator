import glfw
import glm
from glm import *
from light import get_light_inv_dir
from light import get_light_pos
import numpy as np
import pickle
from PIL import Image
from PIL import ImageOps
from random import random
from settings import *
import time
from twohands import init_mano
from twohands import interpolate_sequence
from twohands import get_mano_hands
from utils.controls import compute_matrices_from_inputs
from utils.objloader import load_hole
from utils.object import Object
from utils.shader import load_shaders
from utils.texture import load_texture


# contains OpenGL IDs for the background
class Background(Object):
    def __init__(self):
        Object.__init__(self)

        self.element_buffer = None
        self.program_id = None
        self.texture_id = None
        self.texture_sampler_id = None
        self.uv_buffer = None
        self.vertex_array_object = None
        self.vertex_buffer = None


# contains OpenGL IDs for each hands
class Hand(Object):
    def __init__(self):
        Object.__init__(self)

        self.element_buffer = None
        self.id = None
        self.matrix_id = None
        self.model_matrix_id = None
        self.normal_buffer = None
        self.texture_id = None
        self.texture_sampler_id = None
        self.uv_buffer = None
        self.vertex_array_object = None
        self.vertex_buffer = None


# contains render settings and OpenGL IDs
class Scene:
    depth_bias_matrix_id = None
    depth_matrix_id = None
    depth_program_id = None
    id_near = None
    id_far = None
    light_id = None
    light_inv_dir_id = None
    program_id = None
    shadow_map_id = None
    view_matrix_id = None


# buffers background data for OpenGL
def buffer_data_background(background):
    background.vertex_array_object = GLuint(glGenVertexArrays(1))
    glBindVertexArray(background.vertex_array_object)

    background.vertex_buffer = GLuint(glGenBuffers(1))
    glBindBuffer(GL_ARRAY_BUFFER, background.vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, background.vertices.astype(GLfloat), GL_STATIC_DRAW)

    background.uv_buffer = GLuint(glGenBuffers(1))
    glBindBuffer(GL_ARRAY_BUFFER, background.uv_buffer)
    glBufferData(GL_ARRAY_BUFFER, background.uvs.astype(GLfloat), GL_STATIC_DRAW)

    background.element_buffer = GLuint(glGenBuffers(1))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, background.element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, background.indices.astype(GLushort), GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, background.vertex_buffer)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, GLvoidp(0))

    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, background.uv_buffer)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, GLvoidp(0))

    glBindVertexArray(0)


# buffers hand data for OpenGL
def buffer_data_hand(hand):
    if hand.vertex_array_object is None:
        hand.vertex_array_object = GLuint(glGenVertexArrays(1))

    glBindVertexArray(hand.vertex_array_object)

    if hand.vertex_buffer is None:
        hand.vertex_buffer = GLuint(glGenBuffers(1))

    glBindBuffer(GL_ARRAY_BUFFER, hand.vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, hand.vertices.astype(GLfloat), buffer_data_usage)

    if hand.uv_buffer is None:
        hand.uv_buffer = GLuint(glGenBuffers(1))

    glBindBuffer(GL_ARRAY_BUFFER, hand.uv_buffer)
    glBufferData(GL_ARRAY_BUFFER, hand.uvs.astype(GLfloat), buffer_data_usage)

    if hand.normal_buffer is None:
        hand.normal_buffer = GLuint(glGenBuffers(1))

    glBindBuffer(GL_ARRAY_BUFFER, hand.normal_buffer)
    glBufferData(GL_ARRAY_BUFFER, hand.normals.astype(GLfloat), buffer_data_usage)

    if USE_VBO_INDEXING:
        if hand.element_buffer is None:
            hand.element_buffer = GLuint(glGenBuffers(1))

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, hand.element_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, hand.indices.astype(GLushort), buffer_data_usage)

    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, hand.vertex_buffer)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, GLvoidp(0))

    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, hand.uv_buffer)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, GLvoidp(0))

    glEnableVertexAttribArray(2)
    glBindBuffer(GL_ARRAY_BUFFER, hand.normal_buffer)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, GLvoidp(0))

    glBindVertexArray(0)


# initializes GLFW and creates a window for display
def create_window():
    glfw.init()

    if not glfw.init():
        print('Failed to initialize GLFW')
        return None

    if OUTPUT_MODE == 'files':
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    glfw.window_hint(glfw.DOUBLEBUFFER, GL_TRUE if LIMIT_FPS else GL_FALSE)

    window = glfw.create_window(res[0], res[1], 'Hand simulator', None, None)

    if window is None:
        print('Failed to open GLFW window')
        glfw.terminate()
        return None

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

    return window


# deletes buffers and other OpenGL data
def delete_opengl(frame_buffers, render_buffers, depth_texture, background, hands):
    glDeleteBuffers(1, background.vertex_array_object)
    glDeleteBuffers(1, background.vertex_buffer)
    glDeleteBuffers(1, background.uv_buffer)
    glDeleteProgram(background.program_id)
    glDeleteTextures(background.texture_id)

    for hand in hands:
        glDeleteBuffers(1, hand.vertex_array_object)
        glDeleteBuffers(1, hand.vertex_buffer)
        glDeleteBuffers(1, hand.uv_buffer)
        glDeleteBuffers(1, hand.normal_buffer)

        if USE_VBO_INDEXING:
            glDeleteBuffers(1, hand.element_buffer)

        glDeleteTextures(hand.texture_id)

    glDeleteProgram(Scene.program_id)

    for frame_buffer in frame_buffers:
        glDeleteFramebuffers(1, frame_buffer)

    for render_buffer in render_buffers:
        glDeleteRenderbuffers(1, render_buffer)

    if depth_texture is not None:
        glDeleteTextures(1, depth_texture)

    glfw.terminate()


# initializes miscellaneous OpenGL settings
def init_opengl():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glEnable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


# prepares the scene for shading and shadow mapping
def init_scene():
    if Scene.program_id is None:
        if SHADING == 'basic':
            Scene.program_id = load_shaders('vert_basic.glsl', 'frag_basic.glsl')
            Scene.light_id = glGetUniformLocation(Scene.program_id, 'LightPosition_worldspace')
        elif SHADING == 'plain':
            Scene.program_id = load_shaders('vert_plain.glsl', 'frag_plain.glsl')
        elif SHADING == 'shadow_mapping':
            Scene.depth_program_id = load_shaders('vert_depth.glsl', 'frag_depth.glsl')
            Scene.program_id = load_shaders('vert_shadow.glsl', 'frag_shadow.glsl')

    if Scene.depth_matrix_id is None:
        if SHADING == 'shadow_mapping':
            Scene.depth_matrix_id = glGetUniformLocation(Scene.depth_program_id, 'depthMVP')

    if Scene.view_matrix_id is None:
        Scene.view_matrix_id = glGetUniformLocation(Scene.program_id, 'V')

    if Scene.depth_bias_matrix_id is None:
        if SHADING == 'shadow_mapping':
            Scene.depth_bias_matrix_id = glGetUniformLocation(Scene.program_id, 'DepthBiasMVP')

    if Scene.shadow_map_id is None:
        if SHADING == 'shadow_mapping':
            Scene.shadow_map_id = glGetUniformLocation(Scene.program_id, 'shadowMap')

    if Scene.light_inv_dir_id is None:
        if SHADING == 'shadow_mapping':
            Scene.light_inv_dir_id = glGetUniformLocation(Scene.program_id, 'LightInvDirection_worldspace')

    if Scene.id_near is None:
        Scene.id_near = glGetUniformLocation(Scene.program_id, 'near')

    if Scene.id_far is None:
        Scene.id_far = glGetUniformLocation(Scene.program_id, 'far')


# loads a specified background image
def load_background(path_texture):
    # stretch image to canvas
    vertices = np.array([[-1.0, -1.0, 0.0],
                         [1.0, -1.0, 0.0],
                         [-1.0,  1.0, 0.0],
                         [1.0,  1.0, 0.0]])
    uvs = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]])
    indices = np.array([0, 1, 2, 2, 1, 3])

    background = Background()
    background.vertices = vertices
    background.uvs = uvs
    background.indices = indices
    background.texture_id = load_texture(path_texture)
    background.program_id = load_shaders('vert_bg.glsl', 'frag_bg.glsl')
    background.texture_sampler_id = glGetUniformLocation(background.program_id, 'texture_sampler')

    buffer_data_background(background)

    return background


# loads two hands with textures, texture maps and precomputed neighbors
def load_hands():
    texture_id = load_texture('texture.png')

    right_hand = Hand()
    right_hand.texture_id = texture_id
    right_hand.id = glGetUniformLocation(Scene.program_id, 'id')
    right_hand.matrix_id = glGetUniformLocation(Scene.program_id, 'MVP')
    right_hand.model_matrix_id = glGetUniformLocation(Scene.program_id, 'M')
    right_hand.texture_sampler_id = glGetUniformLocation(Scene.program_id, 'texture_sampler')
    right_hand.faces_hole = load_hole('hands/right_hole.obj')

    with open('hands/uvs_right.pkl', 'rb') as f:
        pickle_uvs_right = pickle.load(f, encoding='latin')

    faces_uv_right = pickle_uvs_right['faces_uvs']
    verts_uv_right = pickle_uvs_right['verts_uvs']

    uvs_right = verts_uv_right[faces_uv_right.reshape(-1)]
    np.concatenate((uvs_right, 0.5 * np.ones((right_hand.faces_hole.shape[0], 2))))

    right_hand.uvs = uvs_right

    faces_vertices_right = np.load('hands/faces_vertices_right.npy')

    right_hand.faces_vertices = faces_vertices_right

    with open('hands/weights_faces_right.npy', 'rb') as f:
        right_hand.weights_faces = np.load(f)

    left_hand = Hand()
    left_hand.texture_id = texture_id
    left_hand.id = glGetUniformLocation(Scene.program_id, 'id')
    left_hand.matrix_id = glGetUniformLocation(Scene.program_id, 'MVP')
    left_hand.model_matrix_id = glGetUniformLocation(Scene.program_id, 'M')
    left_hand.texture_sampler_id = glGetUniformLocation(Scene.program_id, 'texture_sampler')
    left_hand.faces_hole = load_hole('hands/left_hole.obj')

    with open('hands/uvs_left.pkl', 'rb') as f:
        pickle_uvs_left = pickle.load(f, encoding='latin')

    faces_uv_left = pickle_uvs_left['faces_uvs']
    verts_uv_left = pickle_uvs_left['verts_uvs']

    uvs_left = verts_uv_left[faces_uv_left.reshape(-1)]
    np.concatenate((uvs_left, 0.5 * np.ones((left_hand.faces_hole.shape[0], 2))))

    left_hand.uvs = uvs_left

    faces_vertices_left = np.load('hands/faces_vertices_left.npy')

    left_hand.faces_vertices = faces_vertices_left

    with open('hands/weights_faces_left.npy', 'rb') as f:
        left_hand.weights_faces = np.load(f)

    return [right_hand, left_hand]


# generates a random (currently fixed) chessboard as a background
def load_random_chessboard():
    # stretch image to canvas
    vertices = np.array([[-1.0, -1.0, 0.0],
                         [1.0, -1.0, 0.0],
                         [-1.0,  1.0, 0.0],
                         [1.0,  1.0, 0.0]])
    uvs = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]])
    indices = np.array([0, 1, 2, 2, 1, 3])

    background = Background()
    background.vertices = vertices
    background.uvs = uvs
    background.indices = indices

    common_divisors = []

    for i in range(1, int(round(min(res[1], res[0]))) + 1):
        if res[1] % i == res[0] % i == 0:
            common_divisors.append(i)

    data = np.zeros((res[1], res[0], 4), GLubyte)
    data[:, :, 3] = 255

    # r = random()
    #
    # i_closest = -1
    # dist_closest = float('inf')
    #
    # for i, divisor in enumerate(common_divisors):
    #     dist = abs(common_divisors[i] - r * common_divisors[-1])
    #
    #     if dist < dist_closest:
    #         i_closest = i
    #         dist_closest = dist

    # length_side = common_divisors[i_closest]
    length_side = 10    # for now

    blocks_horizontal = int(round(res[0] / length_side))
    blocks_vertical = int(round(res[1] / length_side))

    np.random.seed(4)   # sequence number
    intensities = np.random.rand(blocks_vertical, blocks_horizontal) * 255

    for bv in range(blocks_vertical):
        for bh in range(blocks_horizontal):
            data[bv * length_side:(bv + 1) * length_side, bh * length_side:(bh + 1) * length_side, :3]\
                = GLubyte(int(round(intensities[bv, bh])))

    texture_id = GLuint(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res[0], res[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)

    background.texture_id = texture_id
    background.program_id = load_shaders('vert_bg.glsl', 'frag_bg.glsl')
    background.texture_sampler_id = glGetUniformLocation(background.program_id, 'texture_sampler')

    buffer_data_background(background)

    return background


# contains render loop
def loop(window, frame_buffers, background, hands, depth_texture, num_frames_sequence):
    color_attachment = GL_COLOR_ATTACHMENT0
    f = 0
    start_time = glfw.get_time()
    last_time = start_time
    nb_frames = 0

    # used for initializing a camera position
    mano_hands = get_mano_hands(0)
    hands_avg = np.zeros(3)
    hand_0 = mano_hands[0][0]
    hand_1 = mano_hands[1][0]
    hands_avg += np.mean(hand_0, axis=0)
    hands_avg += np.mean(hand_1, axis=0)
    hands_avg /= 2

    # render loop
    while glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and glfw.window_should_close(window) == 0:
        current_time = glfw.get_time()
        nb_frames += 1

        # print fps
        if current_time - last_time >= 1.0:
            print(str(nb_frames) + ' fps' + ' (frame ' + str(f) + ')')
            nb_frames = 0
            last_time += 1.0

        # choose displayed color attachment
        if glfw.get_key(window, glfw.KEY_1) == glfw.PRESS:
            color_attachment = GL_COLOR_ATTACHMENT0

        if glfw.get_key(window, glfw.KEY_2) == glfw.PRESS:
            color_attachment = GL_COLOR_ATTACHMENT1

        if glfw.get_key(window, glfw.KEY_3) == glfw.PRESS:
            color_attachment = GL_COLOR_ATTACHMENT2

        if glfw.get_key(window, glfw.KEY_4) == glfw.PRESS:
            color_attachment = GL_COLOR_ATTACHMENT3

        # 1. draw background
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffers[0])
        glViewport(0, 0, res[0], res[1])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(background.program_id)

        glBindVertexArray(background.vertex_array_object)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, background.texture_id)
        glUniform1i(background.texture_sampler_id, 0)

        glDrawElements(GL_TRIANGLES, len(background.indices), GL_UNSIGNED_SHORT, GLvoidp(0))

        glClear(GL_DEPTH_BUFFER_BIT)

        depth_mvp = None
        light_inv_dir = None

        # generate depth map for shadow mapping if selected
        if SHADING == 'shadow_mapping':
            glBindFramebuffer(GL_FRAMEBUFFER, frame_buffers[1])
            glViewport(0, 0, 1024, 1024)

            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUseProgram(Scene.depth_program_id)

            light_inv_dir = get_light_inv_dir('random')
            depth_projection_matrix = glm.ortho(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
            depth_view_matrix = glm.lookAt(light_inv_dir, glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
            depth_model_matrix = glm.mat4(1.0)
            depth_mvp = depth_projection_matrix * depth_view_matrix * depth_model_matrix

            glUniformMatrix4fv(Scene.depth_matrix_id, 1, GL_FALSE, value_ptr(depth_mvp))

        # load two MANO hands
        mano_hands = get_mano_hands(f)

        # 2. draw hands
        for i, hand in enumerate(hands):
            hand.faces = mano_hands[i][1]
            hand.faces = np.concatenate((hand.faces, hand.faces_hole))
            hand.vertices_raw = mano_hands[i][0]

            hand.compute_vertices()
            hand.compute_normals()
            buffer_data_hand(hand)

            glBindFramebuffer(GL_FRAMEBUFFER, frame_buffers[0])
            glViewport(0, 0, res[0], res[1])

            if SHADING == 'shadow_mapping':
                glEnable(GL_CULL_FACE)
                glCullFace(GL_BACK)

            glUseProgram(Scene.program_id)

            # used for a pose controlled by mouse and keyboard
            # projection_matrix, view_matrix, model_matrix = compute_matrices_from_inputs(window, res, fov, near, far)

            # matrices needed for a MVP matrix
            projection_matrix = glm.perspective(glm.radians(fov), res[0] / res[1], near, far)
            view_matrix = glm.mat4()

            if coordinate_system == 'world':
                view_matrix = glm.lookAt(glm.vec3(hands_avg) + glm.vec3(0.5 * far, 0, 0),
                                         glm.vec3(hands_avg) + glm.vec3(0.5 * far, 0, 0) +
                                         glm.vec3(-1, 0, 0), glm.vec3(0, 0, 1))
            elif coordinate_system == 'camera':
                view_matrix = glm.lookAt(glm.vec3(0), glm.vec3(-1, 0, 0), glm.vec3(0, 0, 1))

            model_matrix = glm.mat4()
            mvp = projection_matrix * view_matrix * model_matrix

            depth_bias_mvp = None

            # biased MVP matrix matrix needed by shadow mapping
            if SHADING == 'shadow_mapping':
                bias_matrix = glm.mat4(0.5, 0.0, 0.0, 0.0,
                                       0.0, 0.5, 0.0, 0.0,
                                       0.0, 0.0, 0.5, 0.0,
                                       0.5, 0.5, 0.5, 1.0)

                depth_bias_mvp = bias_matrix * depth_mvp

            # basic render settings
            glUniform1i(hand.id, i)
            glUniform1f(Scene.id_far, far)
            glUniform1f(Scene.id_near, near)
            glUniformMatrix4fv(hand.matrix_id, 1, GL_FALSE, value_ptr(mvp))
            glUniformMatrix4fv(hand.model_matrix_id, 1, GL_FALSE, value_ptr(model_matrix))
            glUniformMatrix4fv(Scene.view_matrix_id, 1, GL_FALSE, value_ptr(view_matrix))

            if SHADING == 'basic':
                # light_pos = get_light_pos('random', glm.vec3(hands_avg))
                light_pos = glm.vec3(hands_avg) + glm.vec3(0.5 * far, 0, 1)
                glUniform3f(Scene.light_id, light_pos.x, light_pos.y, light_pos.z)

            if SHADING == 'shadow_mapping':
                glUniformMatrix4fv(Scene.depth_bias_matrix_id, 1, GL_FALSE, value_ptr(depth_bias_mvp))
                glUniform3f(Scene.light_inv_dir_id, light_inv_dir.x, light_inv_dir.y, light_inv_dir.z)

            glBindVertexArray(hand.vertex_array_object)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, hand.texture_id)
            glUniform1i(hand.texture_sampler_id, 0)

            # depth map needed for shadow mapping
            if SHADING == 'shadow_mapping':
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, depth_texture)
                glUniform1i(Scene.shadow_map_id, 1)

            if USE_VBO_INDEXING:
                glDrawElements(GL_TRIANGLES, len(hand.indices), GL_UNSIGNED_SHORT, GLvoidp(0))
            else:
                glDrawArrays(GL_TRIANGLES, 0, len(hand.vertices))

        # display on screen
        if OUTPUT_MODE == 'monitor' or OUTPUT_MODE == 'both':
            glReadBuffer(color_attachment)

            glBindFramebuffer(GL_READ_FRAMEBUFFER, frame_buffers[0])
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            glBlitFramebuffer(0, 0, res[0], res[1], 0, 0, res[0], res[1], GL_COLOR_BUFFER_BIT, GL_NEAREST)

        # save to files
        if OUTPUT_MODE == 'files' or OUTPUT_MODE == 'both':
            glFlush()
            glFinish()
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

            num_digits = len(str(int(round(min(NUM_FRAMES, num_frames_sequence)))))

            glReadBuffer(GL_COLOR_ATTACHMENT0)

            data = glReadPixels(0, 0, res[0], res[1], GL_RGBA, GL_UNSIGNED_BYTE)
            image = ImageOps.flip(Image.frombytes("RGBA", (res[0], res[1]), data))
            image.save(('frames/rgb/frame_{f:0' + str(num_digits) + 'd}').format(f=f+1) + '.png', 'PNG')

            # glReadBuffer(GL_COLOR_ATTACHMENT1)
            #
            # data = glReadPixels(0, 0, res[0], res[1], GL_RGBA, GL_UNSIGNED_BYTE)
            # image = ImageOps.flip(Image.frombytes("RGBA", (res[0], res[1]), data))
            # image.save(('frames/segmentation/frame_{f:0' + str(num_digits) + 'd}').format(f=f+1) + '.png', 'PNG')
            #
            # glReadBuffer(GL_COLOR_ATTACHMENT2)
            #
            # data = glReadPixels(0, 0, res[0], res[1], GL_RGBA, GL_UNSIGNED_BYTE)
            # image = ImageOps.flip(Image.frombytes("RGBA", (res[0], res[1]), data))
            # image.save(('frames/depth/frame_{f:0' + str(num_digits) + 'd}').format(f=f+1) + '.png', 'PNG')
            #
            # glReadBuffer(GL_COLOR_ATTACHMENT3)
            #
            # data = glReadPixels(0, 0, res[0], res[1], GL_RGBA, GL_UNSIGNED_BYTE)
            # image = ImageOps.flip(Image.frombytes("RGBA", (res[0], res[1]), data))
            # image.save(('frames/normal/frame_{f:0' + str(num_digits) + 'd}').format(f=f+1) + '.png', 'PNG')

        glfw.swap_buffers(window)
        glfw.poll_events()

        f += 1

        if f >= NUM_FRAMES or f >= num_frames_sequence:
            break


# prepare frame buffers for single render pass
def setup_frame_buffers():
    frame_buffers = []

    frame_buffer_camera = GLuint(glGenFramebuffers(1))
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_camera)
    frame_buffers.append(frame_buffer_camera)

    render_buffers = []

    render_buffer_rgb = GLuint(glGenRenderbuffers(1))
    glBindRenderbuffer(GL_RENDERBUFFER, render_buffer_rgb)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, res[0], res[1])
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, render_buffer_rgb)
    render_buffers.append(render_buffer_rgb)

    render_buffer_seg = GLuint(glGenRenderbuffers(1))
    glBindRenderbuffer(GL_RENDERBUFFER, render_buffer_seg)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, res[0], res[1])
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, render_buffer_seg)
    render_buffers.append(render_buffer_seg)

    render_buffer_depth_color = GLuint(glGenRenderbuffers(1))
    glBindRenderbuffer(GL_RENDERBUFFER, render_buffer_depth_color)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, res[0], res[1])
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, render_buffer_depth_color)
    render_buffers.append(render_buffer_depth_color)

    render_buffer_normal = GLuint(glGenRenderbuffers(1))
    glBindRenderbuffer(GL_RENDERBUFFER, render_buffer_normal)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, res[0], res[1])
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_RENDERBUFFER, render_buffer_normal)
    render_buffers.append(render_buffer_normal)

    render_buffer_depth = GLuint(glGenRenderbuffers(1))
    glBindRenderbuffer(GL_RENDERBUFFER, render_buffer_depth)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, res[0], res[1])
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, render_buffer_depth)
    render_buffers.append(render_buffer_depth)

    glDrawBuffers([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3])

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print('Camera frame buffer is not complete')

    depth_texture = None

    if SHADING == 'shadow_mapping':
        frame_buffer_light = GLuint(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_light)
        frame_buffers.append(frame_buffer_light)

        max_xy = int(round(max(res[0], res[1])))

        depth_texture = GLuint(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, max_xy, max_xy, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                     GLvoidp(0))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)

        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_texture, 0)

        glDrawBuffer(GL_NONE)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print('Light frame buffer is not complete')

    return frame_buffers, render_buffers, depth_texture


# create window, load background and hands, prepare frame buffers, render, destroy
def render():
    window = create_window()

    if window is None:
        return

    init_opengl()
    # num_frames_sequence = init_mano('sequences/1000fps/raw_sequence0.pkl')
    num_frames_sequence = init_mano('sequences/output/4.pkl')
    # interpolate_sequence(30, 1000)

    init_scene()

    background = load_random_chessboard()
    hands = load_hands()

    frame_buffers, render_buffers, depth_texture = setup_frame_buffers()
    loop(window, frame_buffers, background, hands, depth_texture, num_frames_sequence)
    delete_opengl(frame_buffers, render_buffers, depth_texture, background, hands)
