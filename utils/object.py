import glm


# generic object
class Object:
    def __init__(self):
        self.indices = None
        self.faces = None
        self.faces_hole = None
        self.faces_vertices = None
        self.normals = None
        self.normals_faces = None
        self.normals_vertices_raw = None
        self.uvs = None
        self.vertices = None
        self.vertices_raw = None

    # compute normals for smooth shading
    def compute_normals(self):
        if self.normals_faces is None:
            self.normals_faces = [None] * len(self.faces)
            self.normals_vertices_raw = [None] * len(self.faces_vertices)
            self.normals = [None] * 3 * len(self.faces)

        for f, face in enumerate(self.faces):
            v0 = self.vertices_raw[face[0]]
            v1 = self.vertices_raw[face[1]]
            v2 = self.vertices_raw[face[2]]

            a = v1 - v0
            b = v2 - v0

            a = glm.normalize(a)
            b = glm.normalize(b)

            normal = glm.cross(a, b)

            self.normals_faces[f] = normal

        for f, face_v in enumerate(self.faces_vertices):
            normal_vertex_raw = glm.vec3(0)

            for f_v in face_v:
                normal_vertex_raw += self.normals_faces[f_v]

            normal_vertex_raw /= len(face_v)

            self.normals_vertices_raw[f] = normal_vertex_raw

        for f, face in enumerate(self.faces):
            self.normals[3 * f] = self.normals_vertices_raw[face[0]]
            self.normals[3 * f + 1] = self.normals_vertices_raw[face[1]]
            self.normals[3 * f + 2] = self.normals_vertices_raw[face[2]]

    # compute vertices the order needed for OpenGL
    def compute_vertices(self):
        self.vertices = []

        for face in self.faces:
            self.vertices.append(self.vertices_raw[face[0]])
            self.vertices.append(self.vertices_raw[face[1]])
            self.vertices.append(self.vertices_raw[face[2]])

        for face in self.faces_hole:
            self.vertices.append(self.vertices_raw[face[0]])
            self.vertices.append(self.vertices_raw[face[1]])
            self.vertices.append(self.vertices_raw[face[2]])