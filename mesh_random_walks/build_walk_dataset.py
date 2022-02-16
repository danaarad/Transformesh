import trimesh
import open3d
import numpy as np
import json
from json import JSONEncoder
from tqdm import tqdm
from pathlib import Path, PurePosixPath
from easydict import EasyDict
from walks import get_seq_random_walk_random_global_jumps
import warnings
warnings.filterwarnings("error")
warnings.simplefilter('ignore', DeprecationWarning)


CUBES_LABELS = [
    'apple', 'bat', 'bell', 'brick', 'camel',
    'car', 'carriage', 'chopper', 'elephant', 'fork',
    'guitar', 'hammer', 'heart', 'horseshoe', 'key',
    'lmfish', 'octopus', 'shoe', 'spoon', 'tree',
    'turtle', 'watch'
]
CUBES_SHAPE2LABEL = {v: k for k, v in enumerate(CUBES_LABELS)}

SHREC16_LABELS = [
    'armadillo', 'man', 'centaur', 'dinosaur', 'dog2',
    'ants', 'rabbit', 'dog1', 'snake', 'bird2',
    'shark', 'dino_ske', 'laptop', 'santa', 'flamingo',
    'horse', 'hand', 'lamp', 'two_balls', 'gorilla',
    'alien', 'octopus', 'cat', 'woman', 'spiders',
    'camel', 'pliers', 'myScissor', 'glasses', 'bird1'
]
SHREC16_SHAPE2LABEL = {v: k for k, v in enumerate(SHREC16_LABELS)}


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def fill_dxdydz_features(vertices, mesh_extra, seq, jumps, seq_len):
    walk = np.diff(vertices[seq[:seq_len + 1]], axis=0) * 100
    return walk


def norm_mesh(vertices):
    # Move the mesh model so the bbox center will be at (0, 0, 0)
    mean = np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
    vertices -= mean
    # Scale mesh model to fit into the unit ball
    norm_width = np.max(vertices)
    vertices /= norm_width
    return vertices


def rotate_mesh_vertices_by_random_degree(vertices, max_rot_ang_deg):
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


def rotate_mesh_vertices_by_axis(vertices, rot_ang_deg, axis):
    x = rot_ang_deg * np.pi / 180
    y = rot_ang_deg * np.pi / 180
    z = rot_ang_deg * np.pi / 180
    assert set(axis).issubset(['x', 'y', 'z'])
    A = np.array(((1, 0, 0),
                  (0, np.cos(x), -np.sin(x)),
                  (0, np.sin(x), np.cos(x))),
                 dtype=vertices.dtype)
    B = np.array(((np.cos(y), 0, np.sin(y)),
                  (0, 1, 0),
                  (-np.sin(y), 0, np.cos(y))),
                 dtype=vertices.dtype)
    C = np.array(((np.cos(z), -np.sin(z), 0),
                  (np.sin(z), np.cos(z), 0),
                  (0, 0, 1)),
                 dtype=vertices.dtype)
    if 'x' in axis:
        vertices = np.dot(vertices, A, out=vertices)
    if 'y' in axis:
        vertices = np.dot(vertices, B, out=vertices)
    if 'z' in axis:
        vertices = np.dot(vertices, C, out=vertices)
    return vertices


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


def generate_random_walks(mesh_file,
                          num_walks_per_mesh,
                          walk_len=None,
                          walk_len_vertices_ratio=None,
                          data_augment_rotation=None):
    mesh, mesh_ = load_mesh(mesh_file)
    # normalize mesh vertices to center unit ball for scaling issues
    mesh_data = EasyDict({'vertices': norm_mesh(np.asarray(mesh.vertices)), 'faces': np.asarray(mesh.triangles)})
    augmented_meshes = [{"mesh_data": mesh_data, "mesh_identifier": mesh_file}]
    if data_augment_rotation:
        for ax in ['x', 'y', 'z']:
            mesh_to_rotate, _ = load_mesh(mesh_file)
            rotated_normalized_vertices = rotate_mesh_vertices_by_random_degree(
                vertices=rotate_mesh_vertices_by_axis(
                    vertices=norm_mesh(np.asarray(mesh_to_rotate.vertices)),
                    rot_ang_deg=180,
                    axis=[ax]),
                max_rot_ang_deg=10)
            rotated_mesh_data = EasyDict({'vertices': rotated_normalized_vertices, 'faces': np.asarray(mesh.triangles)})
            mesh_identifier = mesh_file.replace(".obj", f"__rotated_{ax}.obj")
            augmented_meshes += [{"mesh_data": rotated_mesh_data, "mesh_identifier": mesh_identifier}]
    walks = {}  # dictionary from mesh to all of its walks
    for mesh_dict in augmented_meshes:
        mesh_data = mesh_dict["mesh_data"]
        mesh_identifier = mesh_dict["mesh_identifier"]
        m = {}
        for k, v in mesh_data.items():
            m[k] = v
        prepare_mesh_edges(m)
        mesh_extra = {'n_vertices': m["vertices"].shape[0], 'edges': m['edges']}
        # set the walk len to the ratio ov the number of vertices:
        walk_len = int(
            mesh_extra['n_vertices'] * walk_len_vertices_ratio) if walk_len_vertices_ratio is not None else walk_len
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
            mesh_walk_id = f"{mesh_identifier}__{walk_id}"
            walks[mesh_walk_id] = {}
            walks[mesh_walk_id]["shape_id"] = mesh_identifier
            walks[mesh_walk_id]["seq"] = seq
            # walks[mesh_walk_id]["jumps"] = jumps
            walks[mesh_walk_id]["dxdydz"] = dxdydz
            walk_coordinates = mesh_data["vertices"][seq]
            walks[mesh_walk_id]["edges_ratio_angle"] = random_walk_invariant_features(walk_coordinates, dxdydz=dxdydz)
    return walks


def generate_walks_from_dataset(dataset_name, dataset_path, data_split, walk_params, output_file,
                                dev_meshes_per_shape=None, data_augment_rotation=None):
    def get_dir_label(dir_path):
        for shape in shape_labels:
            delimited_shape = f"/{shape}"
            if str(dir_path).endswith(delimited_shape):
                return shape_labels[shape]
        return None

    assert dataset_name in ["shrec16", "cubes"]
    shape_labels = SHREC16_SHAPE2LABEL if dataset_name == "shrec16" else CUBES_SHAPE2LABEL
    assert data_split in ["train", "test"]
    shape_directories = get_all_subdirectories(dataset_path)
    output_dict = {}
    for i in tqdm(range(len(shape_directories))):
        shape_subdir = shape_directories[i]
        label = get_dir_label(shape_subdir)
        shapes_split_dir = PurePosixPath(shape_subdir) / data_split
        file_paths = get_object_files_in_dir(shapes_split_dir)
        dev_counter = 0 if data_split == "train" and dev_meshes_per_shape is not None else None
        for obj_file in file_paths:
            object_goes_to_dev_set = dev_counter is not None and dev_counter < dev_meshes_per_shape
            dev_counter = dev_counter + 1 if object_goes_to_dev_set else dev_counter
            obj_path_str = str(obj_file)
            random_walks = generate_random_walks(mesh_file=obj_path_str,
                                                 num_walks_per_mesh=walk_params['num_walks_per_mesh'],
                                                 walk_len=walk_params['walk_len'],
                                                 walk_len_vertices_ratio=walk_params['walk_len_vertices_ratio'],
                                                 data_augment_rotation=data_augment_rotation)
            for walk in random_walks:
                random_walks[walk]['shape_id'] = random_walks[walk]["shape_id"]
                random_walks[walk]['shape_label'] = label
                random_walks[walk]['split'] = "dev" if object_goes_to_dev_set else data_split
            for walk in random_walks:
                output_dict[walk] = random_walks[walk]
    write_dictionary_to_json(output_json=output_file, dictionary=output_dict)
    print(f"*** Done writing {len(output_dict)} random walk files to {output_file}.")
    return True


def write_dictionary_to_json(output_json, dictionary):
    with open(output_json, "w", encoding="utf-8") as outfile:
        json.dump(dictionary, outfile, indent=4, cls=NumpyArrayEncoder)
    return True


def get_all_subdirectories(dir_path):
    subdirs = []
    for path in Path(dir_path).iterdir():
        if path.is_dir():
            posix_path = PurePosixPath(path)
            subdirs += [posix_path]
    return subdirs


def get_object_files_in_dir(dir_path):
    shape_files = Path(dir_path).glob('*.obj')
    file_paths = [PurePosixPath(x) for x in shape_files if x.is_file()]
    return file_paths


def calc_3d_point_edge(point, other_point):
    return np.linalg.norm(point - other_point)


def calc_3d_point_angles(point_a, point_b, point_c):
    ba = point_a - point_b  # normalization of vectors
    bc = point_c - point_b  # normalization of vectors
    # random walk was stuck at the same point for two steps
    cosine_angle = 1 if np.dot(ba, bc) == 0 else np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return cosine_angle  # return the cosine, not the angle
    # avoid computing the angle due to numeric problems
    # angle = np.arccos(cosine_angle)
    # return np.degrees(angle)


def random_walk_edge_ratio_features(random_walk_xyz, dxdydz=None):
    """Returns array of the ratios of following edge lengths in the random walk.
    This feature is invariant to mesh scale.
    E.g., for following walk edges e1 and e2 return ratio: e1/e2"""
    walk_edge_lengths = []
    for vertex, next_vertex in list(zip(random_walk_xyz, random_walk_xyz[1:])):
        # iterate following vertex pairs
        walk_edge_lengths += [calc_3d_point_edge(vertex, next_vertex)]
        # ################
        # edge_len = calc_3d_point_edge(vertex, next_vertex)
        # if edge_len == 0:
        #     print(edge_len == 0)
        #     print("vertex: ", vertex)
        #     print("next_vertex: ", next_vertex)
        #     print(random_walk_xyz)
        #     print("******************")
        #     print(dxdydz)
        # ###################
    walk_edge_ratios = []
    for edge, next_edge in list(zip(walk_edge_lengths, walk_edge_lengths[1:])):
        if edge == 0:
            # random walk was stuck at the same point for two steps
            walk_edge_ratios += [0]
        else:
            walk_edge_ratios += [np.true_divide(next_edge, edge)]  #[float(next_edge) / float(edge)]
    return np.array(walk_edge_ratios)


def random_walk_edge_angle_features(random_walk_xyz):
    """Returns array of the degrees between following edges in the random walk.
    This feature is invariant to mesh rotation.
    E.g., for following edges e1 and e2, return the angle degree between them"""
    n = len(random_walk_xyz)
    angle_degrees = []
    for i in range(n - 2):
        angle_degrees += [calc_3d_point_angles(point_a=random_walk_xyz[i],
                                               point_b=random_walk_xyz[i + 1],
                                               point_c=random_walk_xyz[i + 2])]
    return np.array(angle_degrees)


def random_walk_invariant_features(random_walk_xyz, dxdydz=None):
    """Get the random walk edge length ratios and angle features,
    these features are invariant to mesh scale and rotation respectively.
    Returns numpy array of pairs, (following edges ratio, following edges angle)"""
    walk_edges_ratios = random_walk_edge_ratio_features(random_walk_xyz, dxdydz=dxdydz)
    walk_edges_angles = random_walk_edge_angle_features(random_walk_xyz)
    return np.stack((walk_edges_ratios, walk_edges_angles), axis=-1)


# x = generate_random_walks(mesh_file="./data/shrec_16/centaur/train/T6.obj",
#                           num_walks_per_mesh=3,
#                           walk_len=5,
#                           walk_len_vertices_ratio=None,
#                           data_augment_rotation=None)
#
# print(x)
# print(x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"])
# a = x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"][0]
# b = x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"][1]
# c = x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"][2]
# print(calc_3d_point_edge(a, b))
# print(calc_3d_point_angles(a, b, c))
# print(random_walk_edge_ratio_features(x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"]))
# print(random_walk_edge_angle_features(x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"]))
# print(random_walk_invariant_features(x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"]))
# print(random_walk_invariant_features(x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"]).shape)

# for i in range(2):
#     print(random_walk_edge_ratio_features(x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"][i]))
#     print(random_walk_edge_angle_features(x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"]))
#     # print(random_walk_invariant_features(x['./data/shrec_16/centaur/train/T6.obj__0']["dxdydz"][i]))
#     print("i: ", i)
#     print("********************")

# path = "./data/shrec_16/"
# get_all_subdirectories(path)
# shape_files = Path("./data/shrec_16/alien/test").glob('*.obj')
# files = [PurePosixPath(x) for x in shape_files if x.is_file()]
# for f in files:
#     print(f)
# print(SHREC16_SHAPE2LABEL)

# RANDOM_WALK_PARAMS = {'num_walks_per_mesh': 256, 'walk_len': None, 'walk_len_vertices_ratio': 1}

# SHREC - Split 16
# output_json = f"./data/walks/walks_shrec16_test_walks_{RANDOM_WALK_PARAMS['num_walks_per_mesh']}_ratio_1V_scaled_rotated.json"
# generate_walks_from_dataset(dataset_name="shrec16",
#                             dataset_path="./data/shrec_16/",
#                             data_split="test",
#                             walk_params=RANDOM_WALK_PARAMS,
#                             output_file=output_json,
#                             data_augment_rotation=True)
#
# output_json = f"./data/walks/walks_shrec16_train_dev_walks_{RANDOM_WALK_PARAMS['num_walks_per_mesh']}_ratio_1V_scaled_rotated.json"
# generate_walks_from_dataset(dataset_name="shrec16",
#                             dataset_path="./data/shrec_16/",
#                             data_split="train",
#                             walk_params=RANDOM_WALK_PARAMS,
#                             output_file=output_json,
#                             dev_meshes_per_shape=2,
#                             data_augment_rotation=True)

# Cube Engraving

# RANDOM_WALK_PARAMS = {'num_walks_per_mesh': 128, 'walk_len': None, 'walk_len_vertices_ratio': 0.5}
#
#
# output_json = f"./data/walks/walks_cubes_test_walks_{RANDOM_WALK_PARAMS['num_walks_per_mesh']}_ratio_05V_scaled_rotated.json"
# generate_walks_from_dataset(dataset_name="cubes",
#                             dataset_path="./data/cubes/",
#                             data_split="test",
#                             walk_params=RANDOM_WALK_PARAMS,
#                             output_file=output_json,
#                             data_augment_rotation=True)
#
# output_json = f"./data/walks/walks_cubes_train_dev_walks_{RANDOM_WALK_PARAMS['num_walks_per_mesh']}_ratio_05V_scaled_rotated.json"
# generate_walks_from_dataset(dataset_name="cubes",
#                             dataset_path="./data/cubes/",
#                             data_split="train",
#                             walk_params=RANDOM_WALK_PARAMS,
#                             output_file=output_json,
#                             dev_meshes_per_shape=15,
#                             data_augment_rotation=True)


# RANDOM_WALK_PARAMS = {'num_walks_per_mesh': 256, 'walk_len': None, 'walk_len_vertices_ratio': 0.5}
#
#
# output_json = f"./data/walks/walks_cubes_test_walks_{RANDOM_WALK_PARAMS['num_walks_per_mesh']}_ratio_05V_scaled.json"
# generate_walks_from_dataset(dataset_name="cubes",
#                             dataset_path="./data/cubes/",
#                             data_split="test",
#                             walk_params=RANDOM_WALK_PARAMS,
#                             output_file=output_json,
#                             data_augment_rotation=None)
#
# output_json = f"./data/walks/walks_cubes_train_dev_walks_{RANDOM_WALK_PARAMS['num_walks_per_mesh']}_ratio_05V_scaled.json"
# generate_walks_from_dataset(dataset_name="cubes",
#                             dataset_path="./data/cubes/",
#                             data_split="train",
#                             walk_params=RANDOM_WALK_PARAMS,
#                             output_file=output_json,
#                             dev_meshes_per_shape=15,
#                             data_augment_rotation=None)
