import torch
import torch.nn.functional as F
import math


def rotation_matrix(axis, angle, device):
    """
    Returns a 3x3 rotation matrix for rotating a vector around the given axis by the specified angle.

    Args:
        axis (torch.Tensor): A 3-dimensional PyTorch tensor representing the rotation axis.
        angle (float): The angle of rotation in radians.

    Returns:
        torch.Tensor: A 3x3 rotation matrix as a PyTorch tensor.
    """
    # Normalize the axis vector
    axis = axis / torch.norm(axis)

    # Extract axis components
    x, y, z = axis

    # Compute trigonometric values
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    # Construct the rotation matrix
    rotation_matrix = torch.tensor([
        [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ], device=device)

    return rotation_matrix

def rotate_mat(point, vector, angle, device):
    u, v, w = vector[0], vector[1], vector[2]
    a, b, c = point[0], point[1], point[2]
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [u * u + (v * v + w * w) * cos,
         u * v * (1 - cos) - w * sin,
         u * w * (1 - cos) + v * sin,
         (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - cos) + (b * w - c * v) * sin],
        [u * v * (1 - cos) + w * sin,
         v * v + (u * u + w * w) * cos,
         v * w * (1 - cos) - u * sin,
         (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - cos) + (c * u - a * w) * sin],
        [u * w * (1 - cos) - v * sin,
         v * w * (1 - cos) + u * sin,
         w * w + (u * u + v * v) * cos,
         (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - cos) + (a * v - b * u) * sin],
        [0, 0, 0, 1]
    ], device=device)


def get_vertex_normals(vertices, edges, radii, device):
    # this function gets the normals for each vertex connected by n_edges
    vertices_norms = torch.zeros((len(vertices), 3), device=device)
    for v in range(len(vertices)):
        point_edges = edges[(edges == v).any(1)]
        vectors = torch.zeros((len(point_edges), 3), device=device)
        i = 0
        for edg in point_edges:
            vectors[i] = vertices[v] - vertices[edg[edg != v].item()]
            i += 1
        unit_vectors = F.normalize(vectors, p=2, dim=1)
        sum_dirs = unit_vectors[0]
        for i in range(1, vectors.size(0)):
            sum_dirs += unit_vectors[i]
            sum_dirs = F.normalize(sum_dirs, p=2, dim=0)
        vertices_norms[v] = sum_dirs
    vertices_norms_points = vertices + (vertices_norms * radii.repeat(3, 1).T)
    return vertices_norms_points


def get_lines_norm_points(lines, vertices, radii, device, n_g = 1, n_d = 2):
    points_line = torch.empty((len(lines) * (n_d + 1) * n_g, 3), device=device)
    theta = 2 * math.pi / n_g
    pl = 0
    for line in lines:
        v1_i, v2_i = line
        v1 = vertices[v1_i]
        v2 = vertices[v2_i]
        r1 = radii[v1_i]
        r2 = radii[v2_i]
        c12 = v2 - v1
        perpendicular_vector = torch.tensor([1., 1., (-c12[0] - c12[1]) / c12[2]], device=device)
        d0 = F.normalize(perpendicular_vector, p=2, dim=0)
        for g in range(n_g):
            angle = theta * g
            rot_mat = rotation_matrix(c12, angle, device=device)
            nor_c12 = F.normalize(torch.matmul(rot_mat, d0), p=2, dim=0)
            p1 = v1 + nor_c12 * r1
            p2 = v2 + nor_c12 * r2

            points_line[pl] = p1
            points_line[pl + 1] = p2
            pl += 2
            for d in range(n_d - 1):
                points_line[pl] = p1 + ((p2 - p1) * (d + 1) / n_d)
                pl += 1
    return points_line

def plane_line_intersection(n, p, d, a):
    vpt = torch.dot(d, n)

    if vpt == 0:
        print("SLAB GENERATION::Parallel Error")
        return torch.zeros(3)

    t = torch.sum((p - a) * n) / vpt
    return a + d * t

def intersect_point_of_cones(v1, r1, v2, r2, v3, r3, norm, device):
    c12 = v2 - v1
    c13 = v3 - v1
    phi_12 = math.pi/2
    phi_13 = math.pi/2
    p12 = v1 + F.normalize(c12, dim=0) * math.cos(phi_12) * r1
    p13 = v1 + F.normalize(c13, dim=0) * math.cos(phi_13) * r1

    mat = rotate_mat(p12, c12, torch.tensor(math.pi / 2, device=device), device=device)
    dir_12 = torch.matmul(mat, torch.cat((norm, torch.tensor([0], device=device))))[:3]
    intersect_p = plane_line_intersection(c13, p13, dir_12, p12)

    v1p = intersect_p - v1
    scaled_n = torch.sqrt((r1 * r1 - torch.dot(v1p, v1p))) * norm
    return intersect_p, scaled_n, p12, p13


def compute_baricenter(points):
    return torch.sum(points, dim=0) / points.size(0)


def get_face_normals(faces, vertices, radii, device):
    points_face = torch.empty((len(faces) * 2 * 3, 3), device=device)
    #orthocenter = torch.empty((len(faces) * 2, 3))
    baricenter = torch.empty((len(faces) * 2, 3), device=device)
    #incenter = torch.empty((len(faces) * 2, 3))
    f = 0
    p = 0
    for face in faces:
        v1_i, v2_i, v3_i = face
        v1 = vertices[v1_i]
        v2 = vertices[v2_i]
        v3 = vertices[v3_i]
        r1 = radii[v1_i]
        r2 = radii[v2_i]
        r3 = radii[v3_i]
        norm = F.normalize(torch.cross(v1-v2, v1-v3), dim=0)
        intersect_p, scaled_n, p12, p13 = intersect_point_of_cones(v1, r1, v2, r2, v3, r3, norm, device=device)
        points_face[f] = intersect_p + scaled_n
        points_face[f + 3] = intersect_p - scaled_n
        intersect_p, scaled_n, p21, p23 = intersect_point_of_cones(v2, r2, v1, r1, v3, r3, norm, device=device)
        points_face[f + 1] = intersect_p + scaled_n
        points_face[f + 4] = intersect_p - scaled_n
        intersect_p, scaled_n, p31, p32 = intersect_point_of_cones(v3, r3, v1, r1, v2, r2, norm, device=device)
        points_face[f + 2] = intersect_p + scaled_n
        points_face[f + 5] = intersect_p - scaled_n
        #orthocenter[p] = compute_orthocenter(points_face[f:f + 3])
        #orthocenter[p + 1] = compute_orthocenter(points_face[f + 3:f + 6])
        baricenter[p] = compute_baricenter(points_face[f:f + 3])
        baricenter[p + 1] = compute_baricenter(points_face[f + 3:f + 6])
        #incenter[p] = compute_incenter(points_face[f:f + 3])
        #incenter[p + 1] = compute_incenter(points_face[f + 3:f + 6])
        p += 2
        f += 6

    return points_face, baricenter


def get_surface_points(vertices, radii, edges, faces, lines, device):
    points_norm_vtx = get_vertex_normals(vertices, edges, radii, device)
    points_lines = get_lines_norm_points(lines, vertices, radii, n_g=8, n_d=5, device=device)
    points_faces, points_baricenter = get_face_normals(faces, vertices, radii, device=device)
    surface_points = torch.cat((points_norm_vtx, points_lines, points_faces, points_baricenter), dim=0)
    return surface_points.unsqueeze(0)


