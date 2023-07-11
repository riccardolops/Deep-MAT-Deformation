import torch


def read_ma(filename):
    # this function reads the ma file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize lists to store vertices, faces, and edges
    vertices = []
    radii = []
    faces = []
    edges = []

    # Parse the remaining lines to extract vertices, faces, and edges
    for line in lines[1:]:
        line = line.split()
        if line[0] == 'v':
            # Parse vertex coordinates and radius of medial axis
            vertex_coords = list(map(float, line[1:4]))
            radius = float(line[4])
            radii.append(radius)
            vertices.append(vertex_coords)
        elif line[0] == 'e':
            # Parse edge connections
            edge_connections = list(map(int, line[1:]))
            edges.append(edge_connections)
        elif line[0] == 'f':
            # Parse face connections
            face_connections = list(map(int, line[1:]))
            faces.append(face_connections)

    # Convert lists to tensors
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)
    edges = torch.tensor(edges, dtype=torch.int64)
    radii = torch.tensor(radii, dtype=torch.float32)

    edges_to_remove = []
    for face in faces:
        for i in range(3):
            edge1 = torch.tensor([face[i], face[(i + 1) % 3]])
            edge2 = torch.tensor([face[(i + 1) % 3], face[i]])
            if torch.any(torch.all(edges == edge1, dim=1)):
                edges_to_remove.append(edge1)
            elif torch.any(torch.all(edges == edge2, dim=1)):
                edges_to_remove.append(edge2)

    edges_to_remove = torch.stack(edges_to_remove)
    lines = edges[~torch.any(torch.all(edges[:, None] == edges_to_remove[None], dim=-1), dim=-1)]
    return vertices, radii, edges, faces, lines
