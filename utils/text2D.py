import glm
import numpy as np
from OpenGL.GL import *
from utils.shader import load_shaders
from utils.texture import load_texture

text_2d_shader_id = None
text_2d_texture_id = None
text_2d_uniform_id = None
text_2d_uv_buffer_id = None
text_2d_vertex_buffer_id = None


# loads a texture and shader for 2D texts
def init_text_2d(texture_path):
    global text_2d_shader_id
    global text_2d_texture_id
    global text_2d_uniform_id
    global text_2d_uv_buffer_id
    global text_2d_vertex_buffer_id

    text_2d_texture_id = load_texture(texture_path)
    
    text_2d_vertex_buffer_id = glGenBuffers(1)
    text_2d_uv_buffer_id = glGenBuffers(1)
    
    text_2d_shader_id = load_shaders("vert_txt.glsl", "frag_txt.glsl")
    text_2d_uniform_id = glGetUniformLocation(text_2d_shader_id, "textureSampler")


# prints a text to a predefined location
def print_text_2d(text, x, y, size):
    length = len(text)

    vertices = []
    uvs = []

    for i in range(length):
        vertex_up_left = glm.vec2(x + i * size, y + size)
        vertex_up_right = glm.vec2(x + i * size + size, y + size)
        vertex_down_right = glm.vec2(x + i * size + size, y)
        vertex_down_left = glm.vec2(x + i * size, y)

        vertices.append(vertex_up_left)
        vertices.append(vertex_down_left)
        vertices.append(vertex_up_right)

        vertices.append(vertex_down_right)
        vertices.append(vertex_up_right)
        vertices.append(vertex_down_left)

        character = ord(text[i])
        uv_x = (character % 16) / 16.0
        uv_y = (character // 16) / 16.0

        uv_up_left = glm.vec2(uv_x, 1.0 - uv_y)
        uv_up_right = glm.vec2(uv_x + 1.0 / 16.0, 1.0 - uv_y)
        uv_down_right = glm.vec2(uv_x + 1.0 / 16.0, 1.0 - (uv_y + 1.0 / 16.0))
        uv_down_left = glm.vec2(uv_x, 1.0 - (uv_y + 1.0 / 16.0))
        uvs.append(uv_up_left)
        uvs.append(uv_down_left)
        uvs.append(uv_up_right)

        uvs.append(uv_down_right)
        uvs.append(uv_up_right)
        uvs.append(uv_down_left)

    glBindBuffer(GL_ARRAY_BUFFER, text_2d_vertex_buffer_id)
    glBufferData(GL_ARRAY_BUFFER, np.array([element for vertex in vertices for element in vertex], dtype=GLfloat),
                 GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, text_2d_uv_buffer_id)
    glBufferData(GL_ARRAY_BUFFER, np.array([element for uv in uvs for element in uv], dtype=GLfloat), GL_STATIC_DRAW)

    glUseProgram(text_2d_shader_id)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, text_2d_texture_id)
    glUniform1i(text_2d_uniform_id, 0)

    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, text_2d_vertex_buffer_id)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, GLvoidp(0))

    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, text_2d_uv_buffer_id)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, GLvoidp(0))

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glDrawArrays(GL_TRIANGLES, 0, len(vertices))

    glDisable(GL_BLEND)

    glDisableVertexAttribArray(0)
    glDisableVertexAttribArray(1)


# deletes buffers, the texture and the shader
def cleanup_text_2d():
    glDeleteBuffers(1, text_2d_vertex_buffer_id)
    glDeleteBuffers(1, text_2d_uv_buffer_id)
    glDeleteTextures(1, text_2d_texture_id)
    glDeleteProgram(text_2d_shader_id)
