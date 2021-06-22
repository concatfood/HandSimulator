import glm
from utils.object import Object


# load the part with closes the wrist part of the hand as faces
def load_hole(path):
    faces = []

    try:
        f = open(path, 'r')
    except OSError:
        print("Could not open/read file {file}".format(file=path))

        return faces

    with f:
        for line in f:
            words = line.split()
            line_header = words[0]

            if line_header == 'f':
                words = [item for sublist in [word.split('/') for word in words] for item in sublist]

                if len(words) == 10:
                    vertex_index = (int(words[1]) - 1, int(words[4]) - 1, int(words[7]) - 1)
                else:
                    print('File {file} cannot be read. The faces are not in the correct format.'.format(file=path))

                    return faces

                faces.append(vertex_index)

    return faces


# load an OBJ file
def load_obj(path):
    path = 'objects/' + path
    print("Loading OBJ file {file}".format(file=path))

    out_vertices = []
    out_uvs = []
    out_normals = []

    vertex_indices = []
    uv_indices = []
    normal_indices = []
    temp_vertices = []
    temp_uvs = []
    temp_normals = []

    try:
        f = open(path, 'r')
    except OSError:
        print("Could not open/read file {file}".format(file=path))

        return out_vertices, out_uvs, out_normals

    with f:
        for line in f:
            words = line.split()
            line_header = words[0]

            if line_header == 'v':
                temp_vertices.append(glm.vec3(float(words[1]), float(words[2]), float(words[3])))
            elif line_header == 'vt':
                temp_uvs.append(glm.vec2(float(words[1]), float(words[2])))
            elif line_header == 'vn':
                temp_normals.append(glm.vec3(float(words[1]), float(words[2]), float(words[3])))
            elif line_header == 'f':
                words = [item for sublist in [word.split('/') for word in words] for item in sublist]

                if len(words) == 10:
                    vertex_index = (int(words[1]), int(words[4]), int(words[7]))
                    uv_index = (int(words[2]), int(words[5]), int(words[8]))
                    normal_index = (int(words[3]), int(words[6]), int(words[9]))
                else:
                    print('File {file} cannot be read. The faces are not in the correct format.'.format(file=path))

                    return out_vertices, out_uvs, out_normals

                vertex_indices.extend(vertex_index)
                uv_indices.extend(uv_index)
                normal_indices.extend(normal_index)

    for i in range(len(vertex_indices)):
        vertex_index = vertex_indices[i]
        uv_index = uv_indices[i]
        normal_index = normal_indices[i]

        vertex = temp_vertices[vertex_index - 1]
        uv = temp_uvs[uv_index - 1]
        normal = temp_normals[normal_index - 1]

        out_vertices.append(vertex)
        out_uvs.append(uv)
        out_normals.append(normal)

    faces = [tuple([j - 1 for j in vertex_indices[i:i+3]]) for i in range(0, len(vertex_indices), 3)]

    f.close()

    # store object in object data type
    obj = Object()
    obj.faces = faces
    obj.vertices = out_vertices
    obj.vertices_raw = temp_vertices
    obj.uvs = out_uvs
    obj.normals = out_normals

    return obj
