import numpy as np


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
        self.weights_faces = None

    # compute normals for smooth shading
    def compute_normals(self):
        if self.normals_faces is None:
            self.normals_faces = np.zeros((len(self.faces), 3))
            self.normals_vertices_raw = np.zeros((len(self.faces_vertices), 3))
            self.normals = np.zeros((3 * len(self.faces), 3))

        v0 = self.vertices_raw[self.faces[:, 0]]
        v1 = self.vertices_raw[self.faces[:, 1]]
        v2 = self.vertices_raw[self.faces[:, 2]]

        a = v1 - v0
        b = v2 - v0

        a /= np.sqrt(np.einsum('...i,...i', a, a))[:, np.newaxis]
        b /= np.sqrt(np.einsum('...i,...i', b, b))[:, np.newaxis]

        normal = np.cross(a, b)

        self.normals_faces = normal
        self.normals_faces = np.concatenate((self.normals_faces, np.zeros((1, 3))))

        self.normals_vertices_raw = np.zeros((self.vertices_raw.shape[0], 3))

        for i in range(13):
            self.normals_vertices_raw += self.weights_faces[self.faces_vertices[:, i]][:, np.newaxis]\
                                         * self.normals_faces[self.faces_vertices[:, i]]

        self.normals_vertices_raw /= np.sqrt(np.einsum('...i,...i', self.normals_vertices_raw,
                                                       self.normals_vertices_raw))[:, np.newaxis]

        self.normals = self.normals_vertices_raw[self.faces.reshape(-1)]

    # compute vertices the order needed for OpenGL
    def compute_vertices(self):
        self.vertices = self.vertices_raw[self.faces.reshape(-1)]
        np.concatenate((self.vertices, self.vertices_raw[self.faces_hole.reshape(-1)]))
