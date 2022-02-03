import trimesh
import open3d
import numpy as np
import pyvista as pv
import json
from pyvista import examples
from easydict import EasyDict

# from build_walk_dataset import norm_model
from walks import get_seq_random_walk_random_global_jumps


def fill_dxdydz_features(vertices, mesh_extra, seq, jumps, seq_len):
    walk = np.diff(vertices[seq[:seq_len + 1]], axis=0) * 100
    return walk


def prepare_mesh_edges(mesh):
    """Does the following:
    1. for each vertex store all vertices sharig a face with it
    2. remove the vertex itself from its edge set
    3. compute the max degree of any vertex in the mesh
    4. pad each vertex vector to fit the max degree
    5. return a matrix of num_vertices * max_degree
    """
    vertices = mesh['vertices']
    faces = mesh['faces']
    mesh['edges'] = [set() for _ in range(vertices.shape[0])]
    for i in range(faces.shape[0]):
        for v in faces[i]:
            mesh['edges'][v] |= set(faces[i])
    for i in range(vertices.shape[0]):
        if i in mesh['edges'][i]:
            mesh['edges'][i].remove(i)
        mesh['edges'][i] = list(mesh['edges'][i])
    max_vertex_degree = np.max([len(e) for e in mesh['edges']])
    for i in range(vertices.shape[0]):
        if len(mesh['edges'][i]) < max_vertex_degree:
            mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
    mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)


def load_mesh(model_fn, classification=True):
    # To load and clean up mesh - "remove vertices that share position"
    if classification:
        mesh_ = trimesh.load_mesh(model_fn, process=True)
        mesh_.remove_duplicate_faces()
    else:
        mesh_ = trimesh.load_mesh(model_fn, process=False)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
    mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)

    return mesh, mesh_


def data_augmentation_rotation_test(vertices, max_rot_ang_deg):
  x = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  y = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  z = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  A = np.array(((np.cos(x), -np.sin(x), 0),
                (np.sin(x), np.cos(x), 0),
                (0, 0, 1)),
               dtype=vertices.dtype)
  B = np.array(((np.cos(y), 0, -np.sin(y)),
                (0, 1, 0),
                (np.sin(y), 0, np.cos(y))),
               dtype=vertices.dtype)
  C = np.array(((1, 0, 0),
                (0, np.cos(z), -np.sin(z)),
                (0, np.sin(z), np.cos(z))),
               dtype=vertices.dtype)
  np.dot(vertices, A, out=vertices)
  np.dot(vertices, B, out=vertices)
  np.dot(vertices, C, out=vertices)
  return vertices


def norm_mesh_test(vertices):
    # Move the mesh model so the bbox center will be at (0, 0, 0)
    mean = np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
    vertices -= mean
    # Scale mesh model to fit into the unit ball
    norm_with = np.max(vertices)
    vertices /= norm_with
    return vertices

filename = "./data/shrec_16/centaur/train/T6.obj"
filename = "./data/shrec_16/alien/test/T124.obj"
key = "data/shrec_16/alien/test/T124.obj__0"

with open("./data/walks/walks_shrec16_test_walks_128_ratio_05V.json", "rb") as f:
    data = json.load(f)
walk_edges_seq = np.hstack(data[key]["seq"])
list_arr = []
for i in range(len(walk_edges_seq)-1):
    # print(walk_edges_seq[i])
    list_arr += [[2, walk_edges_seq[i], walk_edges_seq[i+1]]]
walk_edges = np.hstack(list_arr)
# print(walk_edges)
x, x_ = load_mesh(filename)
walk_vertices = np.array(x_.vertices)
# print(walk_vertices)
# walk_edges = np.hstack([[2, 232, 14], [2, 14, 187], [2, 187, 242], [2, 242, 202], [2, 202, 139], [2, 139, 234]])
walk_mesh = pv.PolyData(walk_vertices, lines=walk_edges)
# cpos = walk_mesh.plot(show_edges=True, color='black', background='white',)

mesh = pv.read(filename)
p = pv.Plotter()
p.add_mesh(mesh, color=True)
p.add_mesh(walk_mesh, color="red", line_width=5)
p.camera.zoom(1.5)
p.show()

# mesh = pv.read(filename)
# edges = mesh.extract_all_edges()
# print(edges)
#
# vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0]])
# lines = np.hstack([[2, 0, 1], [2, 1, 2]])
# walk_mesh = pv.PolyData(vertices, lines=lines)
# cpos = walk_mesh.plot(show_edges=True, color='black', background='white',)

# edges.lines  # line connectivity stored here
# edges.plot(scalars=np.random.random(edges.n_faces), line_width=5, cmap='jet')
# edges.plot(scalars=np.random.random(edges.n_faces), line_width=5)

# cpos = mesh.plot(show_edges=True, color='yellow', background='white',)

# p = pv.Plotter()
# p.add_mesh(mesh, color=True)
# p.add_mesh(edges, color="red", line_width=5)
# p.camera.zoom(1.5)
# p.show()

# x, x_ = load_mesh(filename)
# x_.show()

# new_x_ = x_
# new_x_.vertices = data_augmentation_rotation_test(norm_mesh_test(new_x_.vertices), 90)
# new_x_.show()
#
# new_x_ = x_
# new_x_.vertices = data_augmentation_rotation_test(norm_mesh_test(new_x_.vertices), 90)
# new_x_.show()

# mesh = x_
# Todo: functions for coloring
# for facet in mesh.faces:
#     mesh.visual.face_colors[facet] = trimesh.visual.random_color()
# x_.show()
# color = trimesh.visual.random_color()
# random_walk_to_color = [132, 136, 129, 59, 58, 220, 9, 57, 161, 69, 67, 107, 149, 107, 67, 2, 184, 131, 41, 238, 143, 166, 211, 48, 22, 134, 202, 146, 226, 228, 199, 126, 47, 11, 247, 108, 37, 99, 27, 217, 145, 224, 124, 31, 218, 236, 174, 73, 53, 212, 103, 21, 44, 232, 241, 33, 60, 120, 35, 178, 135, 61, 92, 52, 221, 70, 152, 37, 99, 108, 226, 115, 89, 87, 240, 199, 105, 72, 126, 150, 177, 179, 42, 4, 88, 157, 176, 143, 238, 59, 129, 57, 9, 149, 161, 69, 107, 220, 58, 2, 207, 242, 119, 41, 238, 166, 211, 48, 0, 239, 222, 217, 248, 64, 27, 99, 224, 22, 134, 88, 177, 179, 118, 42, 4, 42, 128, 125, 176, 209, 104, 202, 81, 97, 89, 199, 218, 194, 124, 68, 243, 18, 156, 141, 236, 188, 216, 212, 103, 21, 241, 172]#, 28, 139, 181, 204, 130, 44, 147, 103, 53, 26, 117, 21, 232, 241, 33, 35, 60, 175, 240, 73, 212, 216, 174, 236, 188, 237, 31, 114, 156, 141, 194, 218, 105, 199, 126, 150, 72, 150, 74, 138, 6, 54, 169, 246, 235, 56, 168, 66, 200, 16, 163, 140, 171, 214, 5, 219, 5, 123, 45, 29, 130, 204, 181, 13, 193, 144, 86, 165, 196, 90, 79, 250, 160, 203, 43, 162, 49, 142, 14, 223, 148, 223, 14, 185, 111, 83, 229, 80, 51, 158, 154, 71, 203, 66, 200, 5, 163, 219, 214, 171, 29, 16, 140, 16, 29, 45, 123, 130, 13, 193, 160, 85, 79, 155, 55, 234, 62, 173, 165, 86, 98, 46, 78, 109, 241, 33, 60, 117, 26, 53, 103, 21, 147, 14, 185, 32, 50, 246, 6, 138, 47, 11, 228, 115, 226, 108, 99, 37, 155]
#
# for v in random_walk_to_color: #range(79):
#     mesh.visual.vertex_colors[v] = color
# x_.show()

# color = trimesh.visual.random_color()
# for facet in mesh.faces:
#     mesh.visual.face_colors[facet] = color
# x_.show()
# mesh = x
# mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
# print(len(mesh_data['vertices']))
# print(len(mesh_data['faces']))
# print(mesh_data['vertices'])
# print(norm_model(mesh_data['vertices']))
#
# m = {}
# for k, v in mesh_data.items():
#     m[k] = v
# print(m.keys())
# prepare_mesh_edges(m)
# print(m['vertices'].shape)
# print(m['faces'].shape)
# print(m['edges'].shape)
#
# print(m['edges'][190])
#
# mesh_data = m
#
# mesh_extra = {}
# mesh_extra['n_vertices'] = mesh_data["vertices"].shape[0]
# mesh_extra['edges'] = mesh_data['edges']
#
#
# n_walks_per_model = 1
# seq_len = 300
# fill_features_functions = [
#     # fill_xyz_features,
#     fill_dxdydz_features,
#     # fill_vertex_indices
# ]
#
# for walk_id in range(n_walks_per_model):
#     f0 = np.random.randint(mesh_data["vertices"].shape[0])  # Get walk starting point
#     seq, jumps = get_seq_random_walk_random_global_jumps(mesh_extra, f0,
#                                                          seq_len)  # Get walk indices (and jump indications)
#     dxdydz = fill_dxdydz_features(mesh_data["vertices"], mesh_extra, seq, jumps, seq_len)
#
#     print("seq: ", seq)
#     print("jumps: ", jumps)
#     print("dxdydz: ", dxdydz)
#     print("**********")
#     # for i in seq:
#     #   print(f"vertex: {i}")
#     #   print(f"mesh edges to: {m['edges'][i]}")
#
# print(list(seq))