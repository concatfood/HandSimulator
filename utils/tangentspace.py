# computes a tangent basis used for normal mapping (not used)
def compute_tangent_basis(vertices, uvs):
    tangents = []
    bi_tangents = []

    for i in range(0, len(vertices), 3):
        v0 = vertices[i + 0]
        v1 = vertices[i + 1]
        v2 = vertices[i + 2]

        uv0 = uvs[i + 0]
        uv1 = uvs[i + 1]
        uv2 = uvs[i + 2]

        delta_pos1 = v1 - v0
        delta_pos2 = v2 - v0

        delta_uv1 = uv1 - uv0
        delta_uv2 = uv2 - uv0

        r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x)
        tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r
        bi_tangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r

        tangents.append(tangent)
        tangents.append(tangent)
        tangents.append(tangent)

        bi_tangents.append(bi_tangent)
        bi_tangents.append(bi_tangent)
        bi_tangents.append(bi_tangent)

        return tangents, bi_tangents
