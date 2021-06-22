import math


# approximates equality constraint
def is_near(v1, v2):
    return math.fabs(v1 - v2) < 0.001


# gets the first similar vertex index
def get_similar_vertex_index(in_vertex, in_uv, in_normal, out_vertices, out_uvs, out_normals):
    for i in range(len(out_vertices)):
        if is_near(in_vertex[0], out_vertices[i][0]) \
                and is_near(in_vertex[1], out_vertices[i][1]) \
                and is_near(in_vertex[2], out_vertices[i][2]) \
                and is_near(in_uv[0], out_uvs[i][0]) \
                and is_near(in_uv[1], out_uvs[i][1]) \
                and is_near(in_normal[0], out_normals[i][0]) \
                and is_near(in_normal[1], out_normals[i][1]) \
                and is_near(in_normal[2], out_normals[i][2]):
            return i

    return None


# VBO indexing in O(n²)
def index_vbo_slow(in_vertices, in_uvs, in_normals):
    out_indices = []
    out_vertices = []
    out_uvs = []
    out_normals = []

    for i in range(len(in_vertices)):
        index = get_similar_vertex_index(in_vertices[i], in_uvs[i], in_normals[i], out_vertices, out_uvs, out_normals)

        if index is not None:
            out_indices.append(index)
        else:
            out_vertices.append(in_vertices[i])
            out_uvs.append(in_uvs[i])
            out_normals.append(in_normals[i])
            out_indices.append(len(out_vertices) - 1)

    return out_indices, out_vertices, out_uvs, out_normals


# auxiliary data type
class PackedVertex:
    def __init__(self, position, uv, normal):
        self.position = position
        self.uv = uv
        self.normal = normal


# return similar vertex index in O(1)
def get_similar_vertex_index_fast(packed, vertex_to_out_index):
    return vertex_to_out_index.get(packed)


# VBO indexing in O(n)
def index_vbo(in_vertices, in_uvs, in_normals):
    out_indices = []
    out_vertices = []
    out_uvs = []
    out_normals = []
    vertex_to_out_index = {}

    for i in range(len(in_vertices)):
        packed = PackedVertex(in_vertices[i], in_uvs[i], in_normals[i])
        index = get_similar_vertex_index_fast(packed, vertex_to_out_index)

        if index is not None:
            out_indices.append(index)
        else:
            out_vertices.append(in_vertices[i])
            out_uvs.append(in_uvs[i])
            out_normals.append(in_normals[i])
            new_index = len(out_vertices) - 1
            out_indices.append(new_index)
            vertex_to_out_index[packed] = new_index

    return out_indices, out_vertices, out_uvs, out_normals


# VBO indexing in O(n²) including tangent, bitangents and normals
def index_vbo_tbn(in_vertices, in_uvs, in_normals, in_tangents, in_bi_tangents):
    out_indices = []
    out_vertices = []
    out_uvs = []
    out_normals = []
    out_tangents = []
    out_bi_tangents = []

    for i in range(len(in_vertices)):
        index = get_similar_vertex_index(in_vertices[i], in_uvs[i], in_normals[i], out_vertices, out_uvs, out_normals)

        if index is not None:
            out_indices.append(index)

            out_tangents[index] += in_tangents[i]
            out_bi_tangents[index] += in_bi_tangents[i]
        else:
            out_vertices.append(in_vertices[i])
            out_uvs.append(in_uvs[i])
            out_normals.append(in_normals[i])
            out_tangents.append(in_tangents[i])
            out_bi_tangents.append(in_bi_tangents[i])
            out_indices.append(len(out_vertices) - 1)

    return out_indices, out_vertices, out_uvs, out_normals
