import os
from scipy.optimize import linear_sum_assignment

from boxes.box_ops import generalized_box_iou_3d, box_cxcyczwhd_to_xyxyzz

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from utils import betti, save_input, save_output
import json
import numpy as np
import os
import networkx as nx
import pyvista
import time
import torch
import yaml
from argparse import ArgumentParser
from functools import partial
from medpy.io import load
from tqdm import tqdm

from inference import relation_infer
from metrics.box_ops_np import box_iou_np
from metrics.boxap import box_ap, iou_filter, get_unique_iou_thresholds, get_indices_of_iou_for_each_metric
from metrics.coco import COCOMetric
from metrics.smd import compute_meanSMD, SinkhornDistance
from models import build_model
import open3d as o3d
from skimage.measure import marching_cubes

parser = ArgumentParser()
# TODO the same confg is used for all the models at the moment
parser.add_argument('--config',
                    default='configs/synth_3D.yaml',
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config.')
parser.add_argument('--model',
                    default='./checkpoints/checkpoint_epoch=150.pt',
                    help='Paths to the checkpoints to use for inference separated by a space.')
parser.add_argument('--device', default='cuda',
                    help='device to use for training')
# parser.add_argument('--eval', action='store_true', help='Apply evaluation of metrics')
parser.add_argument('--eval', type=bool, default=True, help='Apply evaluation of metrics')


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def main(args):
    """
    Run inference for all the testing data
    """
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['exp_name'])
    config = dict2obj(config)
    device = torch.device("cuda") if args.device == 'cuda' else torch.device("cpu")

    nifti_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
    seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
    vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
    nifti_files = []
    vtk_files = []
    seg_files = []

    for file_ in os.listdir(nifti_folder):
        file_ = file_[:-7]
        nifti_files.append(os.path.join(nifti_folder, file_ + '.nii.gz'))
        seg_files.append(os.path.join(seg_folder, file_[:-4] + 'seg.nii.gz'))
        if args.eval:
            vtk_files.append(os.path.join(vtk_folder, file_[:-4] + 'graph.vtp'))

    net = build_model(config).to(device)

    # print('Loading model from:', args.model)
    checkpoint = torch.load(args.model, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.eval()  # Put the CNN in evaluation mode

    t_start = time.time()
    sinkhorn_distance = SinkhornDistance(eps=1e-7, max_iter=100)

    metrics = tuple([COCOMetric(classes=['Node'], per_class=False, verbose=False, max_detection=(200,))])
    iou_thresholds = get_unique_iou_thresholds(metrics)
    iou_mapping = get_indices_of_iou_for_each_metric(iou_thresholds, metrics)
    box_evaluator = box_ap(box_iou_np, iou_thresholds, max_detections=200)

    mean_smd = []
    node_ap_result = []
    edge_ap_result = []
    radius_loss_dataset = []
    beta_error = []
    for idx, _ in enumerate(tqdm(nifti_files)):

        image_data, _ = load(nifti_files[idx])
        segmentation_data, _ = load(seg_files[idx])

        image_data = torch.tensor(image_data, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(0)
        # We can use only float type input for training the network.
        segmentation_data = torch.tensor(segmentation_data, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(0)
        vmax = image_data.max() * 0.001
        image_data = image_data / vmax - 0.5
        segmentation_data = segmentation_data - 0.5
        # image_data = F.pad(image_data, (49,49, 49, 49, 0, 0)) -0.5

        if config.MODEL.USE_SEGMENTATION:
            h, out = net(segmentation_data)
            save_data = segmentation_data
        else:
            h, out = net(image_data)
            save_data = image_data
        out = relation_infer(h.detach(), out, net.relation_embed, net.radius_embed, config.MODEL.DECODER.OBJ_TOKEN,
                             config.MODEL.DECODER.RLN_TOKEN, config.MODEL.DECODER.RAD_TOKEN)

        pred_nodes = torch.tensor(out['pred_nodes'][0], dtype=torch.float)
        pred_edges = torch.tensor(out['pred_rels'][0], dtype=torch.int64)
        pred_radius = torch.tensor(out['pred_radius'][0], dtype=torch.float)
        vtk_data = pyvista.read(vtk_files[idx])
        nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
        edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]

        # Having a quick visualization
        plot_val_rel_sample(segmentation_data[0].squeeze().cpu().numpy(), nodes.cpu().numpy(), edges.cpu().numpy(),
                            out['pred_nodes'][0], out['pred_rels'][0])

        if args.eval:

            # get the radius
            gt_radius = torch.tensor(np.asarray(vtk_data.cell_data['radius']), dtype=torch.float)
            boxes = [torch.cat([nodes, config.MODEL.BOX_WIDTH * torch.ones(nodes.shape, device=nodes.device)],
                               dim=-1).numpy()]
            pred_boxes = [
                torch.cat([pred_nodes, config.MODEL.BOX_WIDTH * torch.ones(pred_nodes.shape, device=nodes.device)],
                          dim=-1).numpy()]
            boxes_class = [np.zeros(boxes[0].shape[0])]
            edge_boxes = torch.stack([nodes[edges[:, 0]], nodes[edges[:, 1]]], dim=2)
            edge_boxes = torch.cat([torch.min(edge_boxes, dim=2)[0] - 0.1, torch.max(edge_boxes, dim=2)[0] + 0.1],
                                   dim=-1).numpy()
            edge_boxes = [edge_boxes[:, [0, 1, 3, 4, 2, 5]]]
            if pred_edges.shape[0] > 0:
                pred_edge_boxes = torch.stack([pred_nodes[pred_edges[:, 0]], pred_nodes[pred_edges[:, 1]]], dim=2)
                pred_edge_boxes = torch.cat(
                    [torch.min(pred_edge_boxes, dim=2)[0] - 0.1, torch.max(pred_edge_boxes, dim=2)[0] + 0.1],
                    dim=-1).numpy()
                pred_edge_boxes = [pred_edge_boxes[:, [0, 1, 3, 4, 2, 5]]]
                edge_boxes_class = [np.zeros(edges.shape[0])]
                # Final cost matrix
                # Since we are loading a single sample during inference
                # Hungarian BEGIN:
                # Hungarian Algorithm matching
                C = -generalized_box_iou_3d(box_cxcyczwhd_to_xyxyzz(torch.as_tensor(pred_edge_boxes[0])),
                                            box_cxcyczwhd_to_xyxyzz(torch.as_tensor(edge_boxes[0])),
                                            eps=1e-10)  # Adding eps to avoid divide by zero error
                C = C.view(1, pred_edge_boxes[0].shape[0], -1).cpu()

                sizes = [len(edge_boxes[0])]
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                mapping = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                           indices]
                # The `mapping` would contain entries like:-
                # (0, 15) => pred[0]~gt[15]
                gt_radius_mapped = gt_radius[mapping[0][1]]
                pred_radius_mapped = pred_radius[mapping[0][0]]
            else:
                pred_edge_boxes = []
                edge_boxes_class = []
                gt_radius_mapped = gt_radius
                pred_radius_mapped = torch.zeros_like(gt_radius).unsqueeze(1)
            # boxes_scores = [np.ones(boxes[0].shape[0])]

            # mean AP
            node_ap_result.extend(
                box_evaluator(pred_boxes, out["pred_boxes_class"], out["pred_boxes_score"], boxes, boxes_class))

            # mean AP
            edge_ap_result.extend(
                box_evaluator(pred_edge_boxes, out["pred_rels_class"], out["pred_rels_score"], edge_boxes,
                              edge_boxes_class, convert_box=False))

            radius_loss = torch.nn.functional.l1_loss(gt_radius_mapped, pred_radius_mapped.squeeze(1))
            radius_loss_dataset.append(radius_loss)
            # mean SMD
            A = torch.zeros((nodes.shape[0], nodes.shape[0]))
            pred_A = torch.zeros((pred_nodes.shape[0], pred_nodes.shape[0]))

            G = nx.Graph()
            G.add_edges_from([tuple(e) for e in edges])
            beta = np.array(betti(G, verbose=False))

            G = nx.Graph()
            if len(pred_edges) == 0:
                beta_pred = np.array([0, 0])
            else:
                G.add_edges_from([tuple(e) for e in pred_edges])
                beta_pred = np.array(betti(G, verbose=False))

            beta_error.append(np.abs(beta_pred - beta))

            A[edges[:, 0], edges[:, 1]] = 1
            A[edges[:, 1], edges[:, 0]] = 1
            A = torch.tril(A)

            if nodes.shape[0] > 1 and pred_nodes.shape[0] > 1 and pred_edges.size != 0:
                # print(pred_edges)
                pred_A[pred_edges[:, 0], pred_edges[:, 1]] = 1.0
                pred_A[pred_edges[:, 1], pred_edges[:, 0]] = 1.0
                pred_A = torch.tril(pred_A)

                mean_smd.append(compute_meanSMD(A, nodes, pred_A, pred_nodes, sinkhorn_distance, n_points=100).numpy())

                # Saving data points for visualization
                if idx <= 50:
                    root_path = os.path.join(config.TRAIN.SAVE_PATH, "runs",
                                             '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'sample_images')
                    os.makedirs(root_path, exist_ok=True)
                    path = os.path.join(root_path, f"ref_epoch_{idx}")
                    os.makedirs(path, exist_ok=True)
                    image_name = f"{config.log.exp_name}_{idx}"
                    save_input(path, image_name, save_data[0, 0, ...].cpu().numpy(), nodes.cpu().numpy(),
                               edges.cpu().numpy(), gt_radius.cpu().numpy())
                    path = os.path.join(root_path, f"pred_epoch_{idx}")
                    os.makedirs(path, exist_ok=True)
                    save_output(path, image_name, pred_nodes.cpu().numpy(), pred_edges.cpu().numpy(),
                                pred_radius.cpu().numpy())

    # topological error
    print("Betti-error:", np.mean(beta_error, axis=0))

    # Accumulate SMD score
    print("Mean SMD:", torch.tensor(mean_smd).mean())
    print("Mean L1 radius regression:", torch.tensor(radius_loss_dataset).mean())

    # accumulate AP score
    node_metric_scores = {}
    edge_metric_scores = {}
    for metric_idx, metric in enumerate(metrics):
        _filter = partial(iou_filter, iou_idx=iou_mapping[metric_idx])
        iou_filtered_results = list(map(_filter, node_ap_result))
        score, curve = metric(iou_filtered_results)
        if score is not None:
            node_metric_scores.update(score)

        iou_filtered_results = list(map(_filter, edge_ap_result))
        score, curve = metric(iou_filtered_results)
        if score is not None:
            edge_metric_scores.update(score)

    for key in node_metric_scores.keys():
        print(key, node_metric_scores[key])

    for key in edge_metric_scores.keys():
        print(key, edge_metric_scores[key])


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


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
