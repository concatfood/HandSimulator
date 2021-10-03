import numpy as np
from OpenGL.GL import *
from PIL import Image, ImageOps
from random import randrange


percentage_random = 0.1


# load texture from specified path
def load_texture(image_path, rand=False):
    image_path = 'textures/' + image_path
    print('Reading image {path}'.format(path=image_path))

    img = ImageOps.flip(Image.open(image_path).convert('RGBA'))

    if rand:
        rand_left = int(round(img.size[0] * percentage_random))
        rand_right = int(round(img.size[0] * percentage_random))
        rand_top = int(round(img.size[1] * percentage_random))
        rand_bottom = int(round(img.size[1] * percentage_random))
        left = randrange(rand_left) if rand_left != 0 else 0
        right = img.size[0] - randrange(rand_right) if rand_right != 0 else img.size[0]
        top = randrange(rand_top) if rand_top != 0 else 0
        bottom = img.size[1] - randrange(rand_bottom) if rand_bottom != 0 else img.size[1]

        img = img.crop((left, top, right, bottom))

    data = np.array(list(img.getdata()), GLubyte)

    texture_id = GLuint(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.size[0], img.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)

    return texture_id
