import numpy as np
from OpenGL.GL import *
from PIL import Image, ImageOps


# load texture from specified path
def load_texture(image_path):
    image_path = 'textures/' + image_path
    print('Reading image {path}'.format(path=image_path))

    img = ImageOps.flip(Image.open(image_path).convert('RGBA'))
    data = np.array(list(img.getdata()),  GLubyte)

    texture_id = GLuint(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.size[0], img.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)

    return texture_id
