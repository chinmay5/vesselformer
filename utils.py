import json

import torch
import torch.nn.functional as F
import numpy as np
import logging
from mmcv.utils import get_logger
import pyvista
# from skimage.measure import marching_cubes_lewiner
from scipy.ndimage.morphology import grey_dilation
from generate_data import prune_patch
from scipy import ndimage
from itertools import product
import pdb

def image_graph_collate(batch):
    images = torch.cat([item_ for item in batch for item_ in item[0]], 0).contiguous()
    segs = torch.cat([item_ for item in batch for item_ in item[1]], 0).contiguous()
    points = [item_ for item in batch for item_ in item[2]]
    edges = [item_ for item in batch for item_ in item[3]]
    radii = [item_ for item in batch for item_ in item[4]]
    return [images, segs, points, edges, radii]


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "mmaction".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)

def save_input(path, idx, patch, patch_coord, patch_edge, radius=None):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    
    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)
    
    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'_sample_'+str(idx).zfill(3)+'_segmentation.stl')
    
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    # 4 is the cell type for a line.
    mesh = pyvista.UnstructuredGrid(patch_edge.flatten(), np.array([4] * len(patch_edge)), patch_coord)
    if radius is not None:
        mesh.cell_data['radius'] = radius
    mesh_structured = mesh.extract_surface()
    mesh_structured.save(path + 'sample_' + str(idx).zfill(3) + '_graph.vtp')
    # mesh = pyvista.PolyData(patch_coord)
    # # print(patch_edge.shape)
    # mesh.lines = patch_edge.flatten()
    # if radius is not None:
    #     mesh.cell_data['radius'] = radius
    # mesh.save(path+'_sample_'+str(idx).zfill(3)+'_graph.vtp')
    
    
def save_output(path, idx, patch_coord, patch_edge, radius=None):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    # print('Num nodes:', patch_coord.shape[0], 'Num edges:', patch_edge.shape[0])
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    # mesh = pyvista.PolyData(patch_coord)
    # if patch_edge.shape[0] > 0:
    #     mesh.lines = patch_edge.flatten()
    # if radius is not None:
    #     mesh.cell_data['radius'] = radius
    mesh = pyvista.UnstructuredGrid(patch_edge.flatten(), np.array([4] * len(patch_edge)), patch_coord)
    if radius is not None:
        mesh.cell_data['radius'] = radius
    mesh_structured = mesh.extract_surface()
    mesh_structured.save(path + 'sample_' + str(idx).zfill(6) + '_graph.vtp')
    # mesh.extract_surface().save(path + 'cent/sample_' + str(idx).zfill(6) + '_cent.vtp')
    # mesh.save(path+'_sample_'+str(idx).zfill(3)+'_graph.vtp')


def patchify_voxel(volume, patch_size, pad):
    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    p_d = p_d -2*pad_d
    
    
    v_h, v_w, v_d = volume.shape

    # Calculate the number of patch in ach axis
    n_w = np.ceil(1.0*(v_w-p_w)/p_w+1)
    n_h = np.ceil(1.0*(v_h-p_h)/p_h+1)
    n_d = np.ceil(1.0*(v_d-p_d)/p_d+1)

    n_w = int(n_w)
    n_h = int(n_h)
    n_d = int(n_d)

    pad_1 = (n_w - 1) * p_w + p_w - v_w
    pad_2 = (n_h - 1) * p_h + p_h - v_h
    pad_3 = (n_d - 1) * p_d + p_d - v_d

    volume = np.pad(volume, ((0, pad_1), (0, pad_2), (0, pad_3)), mode='reflect')
    
    h, w, d= volume.shape
    x_ = np.int32(np.linspace(0, h-p_h, n_h))
    y_ = np.int32(np.linspace(0, w-p_w, n_w))
    z_ = np.int32(np.linspace(0, d-p_d, n_d))
    
    ind = np.meshgrid(x_, y_, z_, indexing='ij')
    
    patch_list = []
    start_ind = []
    seq_ind = []
    for i, start in enumerate(list(np.array(ind).reshape(3,-1).T)):
        patch = np.pad(volume[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d], ((pad_h,pad_h),(pad_w,pad_w),(pad_d,pad_d)))
        patch_list.append(patch)
        start_ind.append(start)
        seq_ind.append([i//(y_.shape[0]*z_.shape[0]), (i%(y_.shape[0]*z_.shape[0]))//z_.shape[0], (i%(y_.shape[0]*z_.shape[0]))%z_.shape[0]])
        
    return patch_list, start_ind, seq_ind, tuple(np.array(volume.shape)+2*np.array(pad))

def patchify_graph(volume, graph, patch_size, pad, noise_sigma=None):
    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    p_d = p_d -2*pad_d
    
    
    v_h, v_w, v_d = volume.shape

    # Calculate the number of patch in ach axis
    n_w = np.ceil(1.0*(v_w-p_w)/p_w+1)
    n_h = np.ceil(1.0*(v_h-p_h)/p_h+1)
    n_d = np.ceil(1.0*(v_d-p_d)/p_d+1)

    n_w = int(n_w)
    n_h = int(n_h)
    n_d = int(n_d)

    pad_1 = (n_w - 1) * p_w + p_w - v_w
    pad_2 = (n_h - 1) * p_h + p_h - v_h
    pad_3 = (n_d - 1) * p_d + p_d - v_d

    volume = np.pad(volume, ((0, pad_1), (0, pad_2), (0, pad_3)), mode='reflect')
    
    h, w, d= volume.shape
    x_ = np.int32(np.linspace(0, h-p_h, n_h))
    y_ = np.int32(np.linspace(0, w-p_w, n_w))
    z_ = np.int32(np.linspace(0, d-p_d, n_d))
    
    ind = np.meshgrid(x_, y_, z_, indexing='ij')
    
    patch_list = []
    start_ind = []
    seq_ind = []
    out = {'pred_nodes':[],'pred_rels':[],'pred_radius':[]}

    block = []

    for i, start in enumerate(list(np.array(ind).reshape(3,-1).T)):
        # patch = np.pad(volume[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d], ((pad_h,pad_h),(pad_w,pad_w),(pad_d,pad_d)))
        if noise_sigma==None:
            end = start + np.array(patch_size) - 2 * np.array(pad)
        else:
            end = start + np.array(patch_size) - 1 - 2 * np.array(pad)
        bounds = [start[0], end[0], start[1], end[1], start[2], end[2]]
        clipped_mesh = graph.clip_box(bounds, invert=False)
        
        
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes == 1) * 2:].reshape(-1, 3)
        patch_attribute = clipped_mesh.cell_data['radius'][np.sum(clipped_mesh.celltypes == 1) * 2:]

        patch_coord_ind = np.where(
            (np.prod(patch_coordinates >= start, 1) * np.prod(patch_coordinates <= end, 1)) > 0.0)
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]  # all coordinates inside the patch
        # Patch attribute is {(v1, v2) :attr, ...}
        patch_edge_list, patch_attribute_dict = [], {}
        for idx, l in enumerate(patch_edge[:, 1:]):
            if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]:
                patch_edge_list.append(tuple(l))
                patch_attribute_dict[tuple(l)] = patch_attribute[idx]
        # patch_edge = [tuple(l) for l in patch_edge[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]

        temp = np.array(patch_edge_list).flatten()  # flatten all the indices of the edges which completely lie inside patch
        old_2_new_idx_dict = {ind: np.where(patch_coord_ind[0] == ind)[0].item() for ind in
                              temp}
        temp = [np.where(patch_coord_ind[0] == ind) for ind in
                temp]  # remap the edge indices according to the new order
        patch_edge = list(np.array(temp).reshape(-1, 2))  # reshape the edge list into previous format
        patch_edge2 = [np.array((old_2_new_idx_dict[edge[0]], old_2_new_idx_dict[edge[1]])) for edge in patch_edge_list]
        assert np.allclose(patch_edge, patch_edge2), "Something wrong with the labeling in our approach"
        # Relabeling the patch attribute dict as well.
        patch_attribute_dict = {(old_2_new_idx_dict[edge[0]], old_2_new_idx_dict[edge[1]]): value for edge, value in patch_attribute_dict.items()}
        
        # Added noise to it
        noise = ((patch_coordinates==start).any()+(patch_coordinates==end).any())>0.0
        patch_coordinates = ((patch_coordinates - start + np.array(pad)) / np.array(patch_size)) 
        if noise_sigma!=None:
            # print(np.sum(noise), len(patch_coordinates))
            patch_coordinates += np.random.normal(loc=0, scale=noise_sigma, size=(patch_coordinates.shape[0],3))*noise
        patch_coordinates = np.clip(patch_coordinates, 0.0, 1.0-1.0/64.0)
        
        # concatenate final variables
        patch_coord_list = [patch_coordinates]  # .to(device))
        patch_edge_list = [np.array(patch_edge)]  # .to(device))
        patch_attribute_dict_list = [patch_attribute_dict]  # A list of dictionary

        mod_patch_coord_list, mod_patch_edge_list, mod_patch_attr_list = prune_patch(patch_coord_list, patch_edge_list, patch_attribute_dict_list)
        
        patch_coordinates = mod_patch_coord_list[0]*np.array(patch_size)+start-np.array(pad)
        clipped_mesh = pyvista.PolyData(patch_coordinates)
        patch_edge = np.concatenate((np.int32(2*np.ones((mod_patch_edge_list[0].shape[0],1))), mod_patch_edge_list[0]), 1)
        clipped_mesh.lines = patch_edge.flatten()
        
        block.append(clipped_mesh.extract_surface().clean())
        
        out['pred_nodes'].append(np.array(mod_patch_coord_list[0]))
        out['pred_rels'].append(np.array(mod_patch_edge_list[0]))
        out['pred_radius'].append(np.array(mod_patch_attr_list[0]))
        start_ind.append(start)
        seq_ind.append([i//(y_.shape[0]*z_.shape[0]), (i%(y_.shape[0]*z_.shape[0]))//z_.shape[0], (i%(y_.shape[0]*z_.shape[0]))%z_.shape[0]])
    
    merged = pyvista.MultiBlock(block).combine(merge_points=True).extract_surface()
    return out, start_ind, seq_ind, tuple(np.array(volume.shape)+2*np.array(pad)), merged


def unpatchify_graph(patch_graphs, start_ind, seq_ind, pad, imsize=[128,128,128]):
    """

    :param patches:
    :param step:
    :param imsize:
    :param scale_factor:
    :return:
    """
    patch_coords, patch_edges, patch_rads = patch_graphs['pred_nodes'], patch_graphs['pred_rels'], patch_graphs['pred_radius']
    occu_matrix = np.empty((8,)+imsize)  # 8 channel occu matrix
    pred_coords = []
    pred_rels = []
    pred_rads = []
    num_nodes = 0
    struct = ndimage.generate_binary_structure(3, 2)
    for i, (patch_coord, patch_edge, patch_rad) in enumerate(zip(patch_coords, patch_edges, patch_rads)):
        patch_node_label = np.zeros(imsize)
        abs_patch_coord = np.array(start_ind[i]) + patch_coord*64
        # print(abs_patch_coord.min(), abs_patch_coord.max(), start_ind[i])
        pred_coords.extend(abs_patch_coord - pad)
        # abs_patch_coord = np.int64(abs_patch_coord)
        abs_patch_coord = np.int64(np.floor((abs_patch_coord)))
        ch_idx = np.sum(2**(np.array(range(3))[::-1])*(np.array(seq_ind[i])%2))
        # print(start_ind[i], seq_ind[i], np.array(seq_ind[i])%2, ch_idx)
        
        # local patch occupancy
        patch_node_label[abs_patch_coord[:,0],abs_patch_coord[:,1],abs_patch_coord[:,2]] = np.array(list(range(num_nodes,num_nodes+patch_coord.shape[0])))+1

        # occu_matrix[ch_idx, start_ind[i][0]-pad[0]:start_ind[i][0]-pad[0]+64, start_ind[i][1]-pad[1]:start_ind[i][1]-pad[1]+64, start_ind[i][2]-pad[2]:start_ind[i][2]-pad[2]+64] = 1
        # for _ in range(8):
        #     inst_label = grey_dilation(patch_node_label, footprint=struct) #size=(3,3,3)) # structure=struct)
        #     inst_label[patch_node_label>0] = patch_node_label[patch_node_label>0]
        #     patch_node_label = inst_label
        occu_matrix[ch_idx, patch_node_label>0] = patch_node_label[patch_node_label>0]

        # occu_matrix[patch_node_label>0.0] = patch_node_label[patch_node_label>0.0]
    
        pred_rels.extend(patch_edge+num_nodes)
        pred_rads.extend(patch_rad)
        num_nodes = num_nodes+patch_coord.shape[0]
    
    for j in range(8):
        patch_node_label = occu_matrix[j,...]
        for _ in range(8):
            inst_label = grey_dilation(patch_node_label, footprint=struct) #size=(3,3,3)) # structure=struct)
            inst_label[patch_node_label>0] = patch_node_label[patch_node_label>0]
            patch_node_label = inst_label
        occu_matrix[j, patch_node_label>0] = patch_node_label[patch_node_label>0]
    
    pred_graph = {'pred_nodes':pred_coords,'pred_rels':pred_rels, 'pred_rads': pred_rads}
    return occu_matrix, pred_graph