import torch
import torch.nn.functional as F
import math
import torch.nn as nn


def plane_line_intersection(n, p, d, a):
    vpt = torch.dot(d, n)

    if vpt == 0:
        print("SLAB GENERATION::Parallel Error")
        return torch.zeros(3)

    t = torch.sum((p - a) * n) / vpt
    return a + d * t


def compute_baricenter(points):
    return torch.sum(points, dim=0) / points.size(0)


class PerpendicularVector(nn.Module):
    def __init__(self):
        super(PerpendicularVector, self).__init__()
        self.perpendicular_vector = nn.Parameter(torch.tensor([1., 1., 0.]), requires_grad=False)

    def forward(self, vector):
        self.perpendicular_vector[2] = (-vector[0] - vector[1]) / vector[2]
        return self.perpendicular_vector


class RotateMatrix(nn.Module):
    def __init__(self):
        super(RotateMatrix, self).__init__()


        self.rotation_matrix = nn.Parameter(torch.tensor([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 1.]
        ]), requires_grad=False)
        self.angle = nn.Parameter(torch.tensor(math.pi / 2), requires_grad=False)
        self.cos, self.sin = torch.cos(self.angle), torch.sin(self.angle)


    def forward(self, point, vector):
        u, v, w = vector[0], vector[1], vector[2]
        a, b, c = point[0], point[1], point[2]

        self.rotation_matrix[0, 0] = u * u + (v * v + w * w) * self.cos
        self.rotation_matrix[0, 1] = u * v * (1 - self.cos) - w * self.sin
        self.rotation_matrix[0, 2] = u * w * (1 - self.cos) + v * self.sin
        self.rotation_matrix[0, 3] = (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - self.cos) + (b * w - c * v) * self.sin

        self.rotation_matrix[1, 0] = u * v * (1 - self.cos) + w * self.sin
        self.rotation_matrix[1, 1] = v * v + (u * u + w * w) * self.cos
        self.rotation_matrix[1, 2] = v * w * (1 - self.cos) - u * self.sin
        self.rotation_matrix[1, 3] = (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - self.cos) + (c * u - a * w) * self.sin

        self.rotation_matrix[2, 0] = u * w * (1 - self.cos) - v * self.sin
        self.rotation_matrix[2, 1] = v * w * (1 - self.cos) + u * self.sin
        self.rotation_matrix[2, 2] = w * w + (u * u + v * v) * self.cos
        self.rotation_matrix[2, 3] = (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - self.cos) + (a * v - b * u) * self.sin

        return self.rotation_matrix


class RotationMatrix(nn.Module):
    def __init__(self):
        super(RotationMatrix, self).__init__()
        self.rotation_matrix = nn.Parameter(torch.empty((3, 3)), requires_grad=False)

    def forward(self, axis, angle):
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

        # Update the rotation matrix tensor
        self.rotation_matrix[0, 0] = t*x*x + c
        self.rotation_matrix[0, 1] = t*x*y - z*s
        self.rotation_matrix[0, 2] = t*x*z + y*s
        self.rotation_matrix[1, 0] = t*x*y + z*s
        self.rotation_matrix[1, 1] = t*y*y + c
        self.rotation_matrix[1, 2] = t*y*z - x*s
        self.rotation_matrix[2, 0] = t*x*z - y*s
        self.rotation_matrix[2, 1] = t*y*z + x*s
        self.rotation_matrix[2, 2] = t*z*z + c

        return self.rotation_matrix


class MATMeshSurface(nn.Module):
    def __init__(self, edges, faces, lines):
        super(MATMeshSurface, self).__init__()
        self.edges = edges
        self.faces = faces
        self.lines = lines
        self.n_g = 8
        self.n_d = 5
        self.RotMat = RotationMatrix()
        self.PerVec = PerpendicularVector()
        self.points_lines = nn.Parameter(torch.empty((len(lines) * (self.n_d + 1) * self.n_g, 3)), requires_grad=False)
        self.points_faces = nn.Parameter(torch.empty((len(faces) * 2 * 3, 3)), requires_grad=False)
        self.points_baricenters = nn.Parameter(torch.empty((len(faces) * 2, 3)), requires_grad=False)
        self.RotMat4 = RotateMatrix()

    def get_norm_to_lines(self, skel_xyzr):
        theta = 2 * math.pi / self.n_g
        pl = 0
        for line in self.lines:
            v1_i, v2_i = line
            v1 = skel_xyzr[v1_i, :-1]
            v2 = skel_xyzr[v2_i, :-1]
            r1 = skel_xyzr[v1_i, -1]
            r2 = skel_xyzr[v2_i, -1]
            c12 = v2 - v1
            perpendicular_vector = self.PerVec(c12)
            d0 = F.normalize(perpendicular_vector, p=2, dim=0)
            for g in range(self.n_g):
                angle = theta * g
                rot_mat = self.RotMat(c12, angle)
                nor_c12 = F.normalize(torch.matmul(rot_mat, d0), p=2, dim=0)
                p1 = v1 + nor_c12 * r1
                p2 = v2 + nor_c12 * r2

                self.points_lines[pl] = p1
                self.points_lines[pl + 1] = p2
                pl += 2
                for d in range(self.n_d - 1):
                    self.points_lines[pl] = p1 + ((p2 - p1) * (d + 1) / self.n_d)
                    pl += 1
        return self.points_lines

    def intersect_point_of_cones(self, v1, r1, v2, r2, v3, r3, norm):
        c12 = v2 - v1
        c13 = v3 - v1
        phi_12 = math.pi / 2
        phi_13 = math.pi / 2
        p12 = v1 + F.normalize(c12, dim=0) * math.cos(phi_12) * r1
        p13 = v1 + F.normalize(c13, dim=0) * math.cos(phi_13) * r1

        mat = self.RotMat4(p12, c12)
        dir_12 = torch.matmul(mat, torch.cat((norm, torch.tensor([0]).to(norm.device))))[:3]
        intersect_p = plane_line_intersection(c13, p13, dir_12, p12)

        v1p = intersect_p - v1
        scaled_n = torch.sqrt((r1 * r1 - torch.dot(v1p, v1p))) * norm
        return intersect_p, scaled_n, p12, p13

    def get_norm_to_triangles(self, skel_xyzr):
        f = 0
        p = 0
        for face in self.faces:
            v1_i, v2_i, v3_i = face
            v1 = skel_xyzr[v1_i, :-1]
            v2 = skel_xyzr[v2_i, :-1]
            v3 = skel_xyzr[v3_i, :-1]
            r1 = skel_xyzr[v1_i, -1]
            r2 = skel_xyzr[v2_i, -1]
            r3 = skel_xyzr[v3_i, -1]
            norm = F.normalize(torch.cross(v1 - v2, v1 - v3), dim=0)
            intersect_p, scaled_n, p12, p13 = self.intersect_point_of_cones(v1, r1, v2, r2, v3, r3, norm)
            self.points_faces[f] = intersect_p + scaled_n
            self.points_faces[f + 3] = intersect_p - scaled_n
            intersect_p, scaled_n, p21, p23 = self.intersect_point_of_cones(v2, r2, v1, r1, v3, r3, norm)
            self.points_faces[f + 1] = intersect_p + scaled_n
            self.points_faces[f + 4] = intersect_p - scaled_n
            intersect_p, scaled_n, p31, p32 = self.intersect_point_of_cones(v3, r3, v1, r1, v2, r2, norm)
            self.points_faces[f + 2] = intersect_p + scaled_n
            self.points_faces[f + 5] = intersect_p - scaled_n

            self.points_baricenters[p] = compute_baricenter(self.points_faces[f:f + 3])
            self.points_baricenters[p + 1] = compute_baricenter(self.points_faces[f + 3:f + 6])
            p += 2
            f += 6
        return torch.cat((self.points_faces, self.points_baricenters), dim=0)

    def forward(self, skel_xyzr):
        return torch.cat((self.get_norm_to_lines(skel_xyzr), self.get_norm_to_triangles(skel_xyzr)), dim=0)