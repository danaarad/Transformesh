import trimesh
import open3d
import numpy as np
from easydict import EasyDict
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


def get_mesh_id(dataset, shape, split, file):
    return f"{dataset}_{split}_{shape}_{file}"


def generate_random_walks(mesh_file, walk_len, num_walks_per_mesh):
    mesh, mesh_ = load_mesh(mesh_file)
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    m = {}
    for k, v in mesh_data.items():
        m[k] = v
    prepare_mesh_edges(m)
    mesh_extra = {'n_vertices': m["vertices"].shape[0], 'edges': m['edges']}
    walks = {}
    walks['mesh_id'] = mesh_file
    for walk_id in range(num_walks_per_mesh):
        f0 = np.random.randint(mesh_data["vertices"].shape[0])  # Get walk starting point
        seq, jumps = get_seq_random_walk_random_global_jumps(mesh_extra,
                                                             f0,
                                                             walk_len)  # Get walk indices (and jump indications)
        dxdydz = fill_dxdydz_features(mesh_data["vertices"], mesh_extra, seq, jumps, walk_len)
        # print("seq: ", seq)
        # print("jumps: ", jumps)
        # print("dxdydz: ", dxdydz)
        # print("**********")
        walks[walk_id] = {}
        walks[walk_id]["seq"] = seq
        walks[walk_id]["jumps"] = jumps
        walks[walk_id]["dxdydz"] = dxdydz
    return walks


x = generate_random_walks(mesh_file="./data/shrec_16/centaur/train/T6.obj",
                          walk_len=50,
                          num_walks_per_mesh=3)
print(x)