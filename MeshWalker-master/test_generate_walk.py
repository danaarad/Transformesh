import numpy as np
from walks import get_seq_random_walk_random_global_jumps



def fill_dxdydz_features(vertices, mesh_extra, seq, jumps, seq_len):
  walk = np.diff(vertices[seq[:seq_len + 1]], axis=0) * 100
  return walk


def get_walk(fn):
    n_walks_per_model = 1
    seq_len = 10
    fill_features_functions = [
        # fill_xyz_features,
        fill_dxdydz_features,
        # fill_vertex_indices
    ]

    print("processing ", fn)
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    print(mesh_data.files)
    # mesh_data = {'vertices': mesh_data["vertices"],
    #              'faces': mesh_data["faces"],
    #              'edges': mesh_data["edges"],
    #              }
    #
    # print(mesh_data["vertices"].size)
    # print(mesh_data["faces"].size)
    # print(mesh_data["edges"].size)

    print('mesh_data["label"]: ', mesh_data["label"])
    print(' mesh_data["vertices"].shape[0]: ', mesh_data["vertices"].shape[0])
    print("vertices[0]: ", mesh_data["vertices"][0])

    mesh_extra = {}
    mesh_extra['n_vertices'] = mesh_data["vertices"].shape[0]
    mesh_extra['edges'] = mesh_data['edges']

    seq_len = 5
    for walk_id in range(n_walks_per_model):
        f0 = np.random.randint(mesh_data["vertices"].shape[0])  # Get walk starting point
        seq, jumps = get_seq_random_walk_random_global_jumps(mesh_extra, f0,
                                                             seq_len)  # Get walk indices (and jump indications)
        dxdydz = fill_dxdydz_features(mesh_data["vertices"], mesh_extra, seq, jumps, seq_len)

        print("seq: ", seq)
        print("jumps: ", jumps)
        print("dxdydz: ", dxdydz)

get_walk("./datasets_processed/cubes/train_turtle_3_not_changed_500.npz")
# get_walk("./datasets_processed/cubes/train_key_141_not_changed_500.npz")
# get_walk("./datasets_processed/cubes/train_octopus_21_not_changed_500.npz")
# get_walk("./datasets_processed/cubes/train_octopus_9_not_changed_500.npz")
