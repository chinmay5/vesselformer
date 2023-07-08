import json

import numpy as np
import open3d as o3d
import yaml
from skimage.measure import marching_cubes


def plot_test_sample(image, points, edges):
    meshes = []
    graphs = []
    # image = image[:, 54:-54, 54:-54]
    porder_points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    border_edges = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    ref_color = [[1, 0, 0] for i in range(len(edges))]
    ref_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points + np.array([0.6, 0, 0])),
        lines=o3d.utility.Vector2iVector(edges)
    )
    ref_line_set.colors = o3d.utility.Vector3dVector(ref_color)
    graphs.append(ref_line_set)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points + np.array([0.6, 0, 0]))
    point_cloud.paint_uniform_color([0, 1, 0])
    graphs.append(point_cloud)

    ref_color = [[0.2, 0.2, 0.2] for i in range(len(border_edges))]
    ref_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(porder_points + np.array([0.6, 0, 0])),
        lines=o3d.utility.Vector2iVector(border_edges),
    )
    ref_line_set.colors = o3d.utility.Vector3dVector(ref_color)
    graphs.append(ref_line_set)

    verts, faces, norms, vals = marching_cubes(image > 0.0, level=0, method='lewiner')
    verts = verts / np.array(image.shape)

    mesh = np.concatenate((faces[:, :2], faces[:, 1:]), axis=0)
    adjucency = np.zeros((verts.shape[0], verts.shape[0]))

    for e in mesh:
        adjucency[e[0], e[1]] = 1.0
        adjucency[e[1], e[0]] = 1.0

    adjucency = np.triu(adjucency)
    mesh = np.array(np.where(np.triu(adjucency) > 0)).T
    # mesh = np.concatenate((np.int32(2*np.ones((mesh.shape[0],1))), mesh), 1)
    # gt_mesh = pyvista.PolyData(verts)
    # gt_mesh.lines = mesh.flatten()
    pred_color = [[0, 0, 1] for i in range(len(mesh))]
    pred_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts - np.array([0.6, 0, 0])),
        lines=o3d.utility.Vector2iVector(mesh),
    )

    pred_line_set.colors = o3d.utility.Vector3dVector(pred_color)
    meshes.append(pred_line_set)

    pred_color = [[0.2, 0.2, 0.2] for i in range(len(border_edges))]
    pred_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector((porder_points - np.array([0.6, 0, 0]))),
        lines=o3d.utility.Vector2iVector(border_edges),
    )

    pred_line_set.colors = o3d.utility.Vector3dVector(pred_color)
    meshes.append(pred_line_set)
    o3d.visualization.draw_geometries(meshes + graphs)


def plot_val_rel_sample(image, points1, edges1, points2, edges2, attn_map=None, relative_coords=True):
    ref_line_sets = []
    pred_line_sets = []
    porder_points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    border_edges = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # edges = np.concatenate((np.int32(2*np.ones((edges.shape[0],1))), edges), 1)
    # gt_graph = pyvista.PolyData(points)
    # gt_graph.lines = edges.flatten()
    ref_color = [[1, 0, 0] for i in range(len(edges1))]
    ref_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points1),
        lines=o3d.utility.Vector2iVector(edges1),
    )
    ref_line_set.colors = o3d.utility.Vector3dVector(ref_color)
    ref_line_sets.append(ref_line_set)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points1)
    point_cloud.paint_uniform_color([0, 1, 0])
    ref_line_sets.append(point_cloud)

    ref_color = [[0.2, 0.2, 0.2] for i in range(len(border_edges))]
    ref_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(porder_points),
        lines=o3d.utility.Vector2iVector(border_edges),
    )
    ref_line_set.colors = o3d.utility.Vector3dVector(ref_color)
    ref_line_sets.append(ref_line_set)

    ref_color = [[1, 0, 0] for i in range(len(edges2))]
    ref_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points2 + np.array([1.1, 0, 0])),
        lines=o3d.utility.Vector2iVector(edges2),
    )
    ref_line_set.colors = o3d.utility.Vector3dVector(ref_color)
    ref_line_sets.append(ref_line_set)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points2 + np.array([1.1, 0, 0]))
    point_cloud.paint_uniform_color([0, 1, 0])
    ref_line_sets.append(point_cloud)

    ref_color = [[0.2, 0.2, 0.2] for i in range(len(border_edges))]
    ref_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(porder_points + np.array([1.1, 0, 0])),
        lines=o3d.utility.Vector2iVector(border_edges),
    )
    ref_line_set.colors = o3d.utility.Vector3dVector(ref_color)
    ref_line_sets.append(ref_line_set)

    verts, faces, norms, vals = marching_cubes(image > 0.0, level=0)
    verts = verts / np.array(image.shape)

    mesh = np.concatenate((faces[:, :2], faces[:, 1:]), axis=0)
    adjucency = np.zeros((verts.shape[0], verts.shape[0]))

    for e in mesh:
        adjucency[e[0], e[1]] = 1.0
        adjucency[e[1], e[0]] = 1.0

    adjucency = np.triu(adjucency)
    mesh = np.array(np.where(np.triu(adjucency) > 0)).T

    pred_color = [[0, 0, 1] for i in range(len(mesh))]
    pred_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts - np.array([1.1, 0, 0])),
        lines=o3d.utility.Vector2iVector(mesh),
    )

    pred_line_set.colors = o3d.utility.Vector3dVector(pred_color)
    pred_line_sets.append(pred_line_set)

    pred_color = [[0.2, 0.2, 0.2] for i in range(len(border_edges))]
    pred_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector((porder_points - np.array([1.1, 0, 0]))),
        lines=o3d.utility.Vector2iVector(border_edges),
    )

    pred_line_set.colors = o3d.utility.Vector3dVector(pred_color)
    pred_line_sets.append(pred_line_set)

    o3d.visualization.draw_geometries(ref_line_sets + pred_line_sets)


def main():
    from dataset_vessel3d import build_vessel_data
    import torch

    with open('configs/synth_3D.yaml') as f:
        print('\n*** Config file')
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['exp_name'])

    class obj:
        def __init__(self, dict1):
            self.__dict__.update(dict1)

    config = json.loads(json.dumps(config), object_hook=obj)

    train_ds, _ = build_vessel_data(config,
                                    mode='split',
                                    )
    from augmentations.transform_volume_3d import Rotate90
    from augmentations.transform_volume_3d import Flip
    # t1, t2, t3, t4, t5, t6, t7 = Rotate90(spatial_axes=(0, 1)), Rotate90(spatial_axes=(0, 2)), Rotate90(spatial_axes=(1, 2)), Flip(spatial_axis=None), Flip(spatial_axis=0), Flip(spatial_axis=1), Flip(spatial_axis=2)
    # t4, t5, t6, t7 = Flip(spatial_axis=None), Flip(spatial_axis=0), Flip(spatial_axis=1), Flip(spatial_axis=2)
    for idx, (images, seg, points, edges, radius) in enumerate(train_ds):
        # if idx != samp_choice:
        #     continue
        images = torch.stack(images)
        seg = torch.stack(seg)
        print(len(points), images.shape, points[0].shape, len(edges))
        plot_test_sample(seg[0].squeeze().cpu().numpy(), points[0].cpu().numpy(), edges[0].cpu().numpy())
        # We apply some augmentation now
        # transform_and_plot(edges, points, seg, [t4, t5, t6, t7])
        # break


def transform_and_plot(edges, points, seg, transforms):
    for t in transforms:
        seg_aug, pc_aug = t(seg[0].squeeze(0).cpu().numpy(), points[0].cpu().numpy())
        plot_test_sample(seg_aug.squeeze(), pc_aug, edges[0].cpu().numpy())


if __name__ == '__main__':
    main()
