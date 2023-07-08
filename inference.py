import itertools
import torch

from boxes import nms, box_ops


def relation_infer(h, out, relation_embed, radius_embed, obj_token, rln_token, rad_token, apply_nms=True, thresh=0.5):
    # all token except the last one is object token
    object_token = h[..., :obj_token, :]

    # last token is relation token
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]
    if rad_token == 0:
        radius_token = relation_token
    else:
        radius_token = h[..., obj_token +rln_token: obj_token + rln_token + rad_token, :]

    # valid tokens
    valid_token = torch.argmax(out['pred_logits'], -1).detach()
    # valid_token = torch.sigmoid(nodes_prob[...,3])>0.5

    pred_nodes = []
    pred_boxes = []
    pred_boxes_score = []
    pred_rels = []
    pred_rel_score = []
    pred_boxes_class = []
    pred_rel_class = []
    pred_radius_raw = []
    for batch_id in range(h.shape[0]):

        # ID of the valid tokens
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        if apply_nms:
            init_boxes = out['pred_nodes'][batch_id, node_id, :3].detach()
            init_boxes = box_ops.box_cxcyczwhd_to_xyxyzz(
                torch.cat([init_boxes, 0.1 * torch.ones(init_boxes.shape, device=init_boxes.device)], dim=-1))
            init_score = out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach()
            keep = nms.nms(init_boxes.cpu(), init_score.cpu(), thresh)
            node_id = node_id[keep]

        # coordinates of the valid tokens
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :3].detach().cpu().numpy())
        pred_boxes.append(out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy())
        pred_boxes_score.append(out['pred_logits'].softmax(-1)[
                                    batch_id, node_id, 1].detach().cpu().numpy())  # TODO: generalize over multi-class
        pred_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy() - 1.0)  # TODO: class starts from 0 not 1
        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:

            # all possible node pairs in all token ordering
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            node_pairs = list(map(list, zip(*node_pairs)))
            # node pairs in valid token order
            node_pairs_valid = torch.tensor(
                [list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))])

            # concatenate valid object pairs relation feature
            if rln_token > 0:
                relation_feature1 = torch.cat((object_token[batch_id, node_pairs[0], :],
                                               object_token[batch_id, node_pairs[1], :],
                                               relation_token[batch_id, ...].repeat(len(node_pairs_valid), 1)), 1)
                radius_feature1 = torch.cat((object_token[batch_id, node_pairs[0], :],
                                             object_token[batch_id, node_pairs[1], :],
                                             radius_token[batch_id, ...].reshape(1, -1).repeat(len(node_pairs_valid), 1)), 1)
                relation_feature2 = torch.cat((object_token[batch_id, node_pairs[1], :],
                                               object_token[batch_id, node_pairs[0], :],
                                               relation_token[batch_id, ...].repeat(len(node_pairs_valid), 1)), 1)
                radius_feature2 = torch.cat((object_token[batch_id, node_pairs[1], :],
                                             object_token[batch_id, node_pairs[0], :],
                                             radius_token[batch_id, ...].reshape(1, -1).repeat(len(node_pairs_valid), 1)), 1)
            else:
                relation_feature1 = torch.cat(
                    (object_token[batch_id, node_pairs[0], :], object_token[batch_id, node_pairs[1], :]), 1)
                relation_feature2 = torch.cat(
                    (object_token[batch_id, node_pairs[1], :], object_token[batch_id, node_pairs[0], :]), 1)

            relation_pred1 = relation_embed(relation_feature1).detach()
            relation_pred2 = relation_embed(relation_feature2).detach()
            relation_pred = (relation_pred1 + relation_pred2) / 2.0
            # Including the radius prediction
            radius_pred1 = radius_embed(radius_feature1).detach()
            radius_pred2 = radius_embed(radius_feature2).detach()
            radius_pred = (radius_pred1 + radius_pred2) / 2.0

            pred_rel = torch.nonzero(torch.argmax(relation_pred, -1)).squeeze(1).cpu().numpy()
            pred_rels.append(node_pairs_valid[pred_rel].cpu().numpy())
            pred_rel_score.append(relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy())
            pred_rel_class.append(torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy() - 1.0)
            # pred-rel are the valid edges
            pred_radius_raw.append(radius_pred[pred_rel].cpu().numpy())
        else:
            pred_rels.append(torch.empty(0, 2))
            pred_rel_score.append(torch.empty(0, 1))
            pred_rel_class.append(torch.empty(0, 1))
            pred_radius_raw.append(torch.empty(0, 1))

    out = {}
    out['pred_nodes'] = pred_nodes
    out['pred_boxes'] = pred_boxes
    out['pred_boxes_score'] = pred_boxes_score
    out['pred_boxes_class'] = pred_boxes_class
    out['pred_rels'] = pred_rels
    out['pred_rels_score'] = pred_rel_score
    out['pred_rels_class'] = pred_rel_class
    out['pred_radius'] = pred_radius_raw
    return out
