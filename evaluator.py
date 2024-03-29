import os
import gc
import torch
import numpy as np
from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointSaver, TensorBoardStatsHandler
from metrics.smd import MeanSMD
from metrics.boxap import MeanBoxAP
from monai.inferers import SimpleInferer
from monai.transforms import (
    Compose,
    AsDiscreted,
)
from multiprocessing import Pool
import pdb
from inference import relation_infer
from utils import save_input, save_output

from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from monai.utils import ForwardMode, min_version, optional_import
from monai.config import IgniteInfo
from monai.engines.utils import default_metric_cmp_fn, default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import ForwardMode, min_version, optional_import
if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

# Define customized evaluator
class RelationformerEvaluator(SupervisedEvaluator):
    def __init__(
        self,
        device: torch.device,
        val_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        mode: Union[ForwardMode, str] = ForwardMode.EVAL,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        decollate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            network = network,
            inferer = SimpleInferer() if inferer is None else inferer
        )

        self.config = kwargs.pop('config')
        self.distributed = kwargs.pop('distributed')
        
    def _iteration(self, engine, batchdata):
        images, segs, nodes, edges = batchdata[0], batchdata[1], batchdata[2], batchdata[3]
        
        # # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # # inputs = torch.cat(inputs, 1)
        images = images.to(engine.state.device,  non_blocking=False)
        segs = segs.to(engine.state.device,  non_blocking=False)
        nodes = [node.cpu() for node in nodes]
        boxes = [torch.cat([node, self.config.MODEL.BOX_WIDTH*torch.ones(node.shape, device=node.device)], dim=-1) for node in nodes]
        edges = [edge.to(engine.state.device,  non_blocking=False) for edge in edges]
        boxes_class = [np.zeros(n.shape[0]) for n in boxes]
        self.network.eval()
        
        with torch.no_grad():
            if self.config.MODEL.USE_SEGMENTATION:
                h, out = self.network(segs.float())
            else:
                h, out = self.network(images)
        
        if self.distributed:
            relation_embed = self.network.module.relation_embed
            radius_embed = self.network.module.radius_embed
        else:
            relation_embed = self.network.relation_embed
            radius_embed = self.network.radius_embed

        out = relation_infer(h.detach(), out, relation_embed, radius_embed, self.config.MODEL.DECODER.OBJ_TOKEN, self.config.MODEL.DECODER.RLN_TOKEN, self.config.MODEL.DECODER.RAD_TOKEN, apply_nms=False)
        
        if self.config.TRAIN.SAVE_VAL:
            root_path = os.path.join(self.config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (self.config.log.exp_name, self.config.DATA.SEED), 'val_samples')
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            for i, (node, edge, pred_node, pred_edge) in enumerate(zip(nodes, edges, out['pred_nodes'], out['pred_rels'])):
                path = os.path.join(root_path, "ref_epoch_"+str(engine.state.epoch).zfill(3)+"_iteration_"+str(engine.state.iteration).zfill(5))
                save_input(path, i, images[i,0,...].cpu().numpy(), node.cpu().numpy(), edge.cpu().numpy())
                path = os.path.join(root_path, "pred_epoch_"+str(engine.state.epoch).zfill(3)+"_iteration_"+str(engine.state.iteration).zfill(5))
                save_output(path, i, pred_node, pred_edge)

        gc.collect()
        torch.cuda.empty_cache()
        
        return {**{"images": images, "boxes": boxes, "boxes_class": boxes_class, "edges": edges}, **out}


def build_evaluator(val_loader, net, optimizer, scheduler, writer, config, device, local_rank=0, distributed=False):
    """[summary]

    Args:
        val_loader ([type]): [description]
        net ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    val_handlers = []
    
    if local_rank==0:
        val_handlers.extend([
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(
                writer,
                tag_name="val_smd",
                output_transform=lambda x: None,
                global_epoch_transform=lambda x: scheduler.last_epoch
            ),
            CheckpointSaver(
                save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models'),
                save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
                save_key_metric=True,
                key_metric_n_saved=5,
                save_interval=1,
                # key_metric_negative_sign=True,
            ),
        ])

    # val_post_transform = Compose(
    #     [AsDiscreted(keys=("pred", "label"),
    #     argmax=(True, False),
    #     to_onehot=True,
    #     n_classes=N_CLASS)]
    # )

    evaluator = RelationformerEvaluator(
        config= config,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        # post_transform=val_post_transform,
        key_val_metric={
            # "val_smd": MeanSMD(
            #     output_transform=lambda x: (x["boxes"], x["edges"], x["pred_boxes"], x["pred_rels"]),
            # )
            "val_AP": MeanBoxAP(
                output_transform=lambda x: (x["pred_boxes"], x["pred_boxes_class"], x["pred_boxes_score"], x["boxes"], x["boxes_class"]),
                max_detections=40
            )
        },
        val_handlers=val_handlers,
        amp=False,
        distributed=distributed,
    )

    return evaluator