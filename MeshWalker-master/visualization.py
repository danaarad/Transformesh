import json

import numpy as np
import trimesh
import pyvista as pv
from build_walk_dataset import load_mesh


def color_random_walk(mesh_file, random_walk_vertices):
    color = trimesh.visual.random_color()
    x, x_ = load_mesh(mesh_file)
    mesh = x_
    for v in random_walk_vertices:
        mesh.visual.vertex_colors[v] = color
    x_.show()


def visualize_mesh(mesh_file, color=None, show_edges=None, background=None):
    mesh = pv.read(mesh_file)
    mesh.plot(color=color, show_edges=show_edges, background=background)


def plot_mesh_edges(mesh_file):
    mesh = pv.read(mesh_file)
    edges = mesh.extract_all_edges()
    edges.lines  # line connectivity stored here
    # edges.plot(scalars=np.random.random(edges.n_faces), line_width=2, cmap='jet')
    edges.plot(scalars=np.random.random(edges.n_faces), line_width=2)


def plot_random_walk(mesh_file, walk_file, walk_id, walk_color=None, zoom=None):
    with open(walk_file, "rb") as f:
        data = json.load(f)
    walk_edges_seq = np.hstack(data[walk_id]["seq"])
    list_arr = []
    for i in range(len(walk_edges_seq) - 1):
        list_arr += [[2, walk_edges_seq[i], walk_edges_seq[i + 1]]]
    walk_edges = np.hstack(list_arr)
    x, x_ = load_mesh(mesh_file)
    walk_vertices = np.array(x_.vertices)
    walk_mesh = pv.PolyData(walk_vertices, lines=walk_edges)
    p = pv.Plotter()
    mesh = pv.read(mesh_file)
    p.add_mesh(mesh, color=True)
    walk_color = "#0AB9DC" if walk_color is None else walk_color
    p.add_mesh(walk_mesh, color=walk_color, line_width=5)
    if zoom is not None:
        p.camera.zoom(zoom)
    p.show()

