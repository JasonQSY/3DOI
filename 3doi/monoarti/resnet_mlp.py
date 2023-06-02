from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb
from torchvision.ops import roi_pool

from . import axis_ops, ilnr_loss #, pwnp_loss
from .detr.detr import MLP
from .detr.transformer import Transformer
from .detr.backbone import Backbone, Joiner
from .detr.position_encoding import PositionEmbeddingSine
from .detr.misc import nested_tensor_from_tensor_list, interpolate
from .detr import box_ops
from .detr.segmentation import (
    MHAttentionMap, MaskHeadSmallConv, dice_loss, sigmoid_focal_loss
)


class ResNetMLP(torch.nn.Module):
    """
    ResNet + MLP to make predictions.
    """

    def __init__(
        self,
        backbone_name = 'resnet50',
        image_size = [192, 256],
        ignore_index = -100,
        num_classes = 1,
        num_queries = 15,
        freeze_backbone = False,
        layers_movable = 3,
        layers_rigid = 3,
        layers_kinematic = 3,
        layers_action = 3,
        layers_axis = 2,
        layers_affordance = 3,
        axis_bins = 30,
        depth_on = True,
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self._ignore_index = ignore_index
        self._image_size = image_size
        self._axis_bins = axis_bins

        # backbone
        backbone_base = Backbone(backbone_name, not freeze_backbone, True, False)
        N_steps = 256 // 2
        #N_steps = transformer_hidden_dim // 2
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        backbone = Joiner(backbone_base, position_embedding)
        backbone.num_channels = backbone_base.num_channels
        self.backbone = backbone

        #hidden_dim = self.transformer.d_model
        hidden_dim = 2048
        self.hidden_dim = hidden_dim

        self.num_queries = num_queries

        # before transformer, input_proj maps 2048 channel resnet50 output to 512-channel
        # transformer input
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        # query mlp maps 2d keypoint coordinates to 256-dim positional encoding
        self.query_mlp = MLP(2, hidden_dim, hidden_dim, 2)
        
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        #self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        if layers_movable > 1:
            self.movable_embed = MLP(hidden_dim, hidden_dim, 3, layers_movable)
        elif layers_movable == 1:
            self.movable_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        if layers_rigid > 1:
            self.rigid_embed = MLP(hidden_dim, hidden_dim, 2, layers_rigid)
        elif layers_rigid == 1:
            #self.rigid_embed = nn.Linear(hidden_dim, 2)
            self.rigid_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        if layers_kinematic > 1:
            self.kinematic_embed = MLP(hidden_dim, hidden_dim, 3, layers_kinematic)
        elif layers_kinematic == 1:
            self.kinematic_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        if layers_action > 1:
            self.action_embed = MLP(hidden_dim, hidden_dim, 3, layers_action)
        elif layers_action == 1:
            self.action_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        if layers_axis > 1:
            #self.axis_embed = MLP(hidden_dim, hidden_dim, 4, layers_axis)
            self.axis_embed = MLP(hidden_dim, hidden_dim, 3, layers_axis)

            # classification
            # self.axis_embed = MLP(hidden_dim, hidden_dim, self._axis_bins * 2, layers_axis)
        elif layers_axis == 1:
            self.axis_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        # self.axis_head = AxisHead()

        # affordance
        # if layers_affordance > 1:
        #     self.aff_embed = MLP(hidden_dim, hidden_dim, 2, layers_affordance)
        # elif layers_affordance == 1:
        #     self.aff_embed = nn.Linear(hidden_dim, 2)
        # else:
        #     raise ValueError("not supported")

        # mask head
        # nheads = self.transformer.nhead
        # self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        # self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim, nheads)

        # # depth head
        # self._depth_on = depth_on
        # if self._depth_on:
        #     self.depth_query = nn.Embedding(1, hidden_dim)
        #     self.depth_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        #     self.depth_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim, nheads)
        #     self.depth_loss = ilnr_loss.MEADSTD_TANH_NORM_Loss()

    def freeze_layers(self, names):
        """
        Freeze layers in 'names'.
        """
        for name, param in self.named_parameters():
            for freeze_name in names:
                if freeze_name in name:
                    param.requires_grad = False
    
    def forward(
        self,
        image: torch.Tensor,
        valid: torch.Tensor,
        keypoints: torch.Tensor,
        bbox: torch.Tensor,
        masks: torch.Tensor,
        movable: torch.Tensor,
        rigid: torch.Tensor,
        kinematic: torch.Tensor,
        action: torch.Tensor,
        affordance: torch.Tensor,
        affordance_map: torch.FloatTensor,
        depth: torch.Tensor,
        axis: torch.Tensor,
        backward: bool = True,
        **kwargs,
    ):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        device = image.device

        # number of queries can be different in runtime
        num_queries = keypoints.shape[1]

        # DETR forward
        samples = image
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None

        # sample keypoint queries from the positional embedding
        use_sine = True
        if use_sine:
            anchors = keypoints.float()
            anchors_float = anchors.clone()
            anchors_float = anchors_float.reshape(-1, 2)
            anchors_float[:, 0] = ((anchors_float[:, 0] / self._image_size[1]) - 0.5) * 2
            anchors_float[:, 1] = ((anchors_float[:, 1] / self._image_size[0]) - 0.5) * 2
            anchors_float = anchors_float.unsqueeze(1).unsqueeze(1)
            # 4x256x1x1
            feature_map = features[-1].tensors # can also be -2
            keypoint_queries = F.grid_sample(
                #pos[0].repeat(self.num_queries, 1, 1, 1), 
                #pos[-1].repeat(self.num_queries, 1, 1, 1), 
                feature_map.repeat(self.num_queries, 1, 1, 1),
                anchors_float, 
                mode='nearest', 
                align_corners=True
            )
            # 4 x 10 (number of object queires) x 256
            hidden_dim = feature_map.shape[1]
            keypoint_queries = keypoint_queries.squeeze().reshape(-1, self.num_queries, hidden_dim)
        else:
            # use learned MLP to map postional encoding
            anchors = keypoints.float()
            anchors_float = anchors.clone()
            anchors_float[:, :, 0] = ((anchors_float[:, :, 0] / self._image_size[1]) - 0.5) * 2
            anchors_float[:, :, 1] = ((anchors_float[:, :, 1] / self._image_size[0]) - 0.5) * 2
            keypoint_queries = self.query_mlp(anchors_float)

        # append depth_query if the model is learning depth.
        # if self._depth_on:
        #     bs = keypoint_queries.shape[0]
        #     depth_query = self.depth_query.weight.unsqueeze(0).repeat(bs, 1, 1)
        #     keypoint_queries = torch.cat((keypoint_queries, depth_query), dim=1)

        # transformer forward
        #src_proj = self.input_proj(src)
        #hs, memory = self.transformer(src_proj, mask, keypoint_queries, pos[-1])

        # In ResNetMLP, we use ResNet features as transformer hs
        ord_hs = keypoint_queries

        #pdb.set_trace()

        # if self._depth_on:
        #     depth_hs = hs[-1][:, -1:]
        #     ord_hs = hs[-1][:, :-1]
        # else:
        #     ord_hs = hs[-1]

        outputs_coord = self.bbox_embed(ord_hs).sigmoid()
        outputs_movable = self.movable_embed(ord_hs)
        outputs_rigid = self.rigid_embed(ord_hs)
        outputs_kinematic = self.kinematic_embed(ord_hs)
        outputs_action = self.action_embed(ord_hs)
        outputs_axis = self.axis_embed(ord_hs)

        # mask forward
        # bbox_mask = self.bbox_attention(ord_hs, memory, mask=mask)
        # seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        # outputs_seg_masks = seg_masks.view(bs, num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        outputs_seg_masks = torch.zeros((bs, num_queries, 192, 256)).to(device)
        outputs_aff_masks = torch.zeros((bs, num_queries, 192, 256)).to(device)
        outputs_depth = torch.zeros((bs, 1, 192, 256)).to(device)
        
        out = {
            'pred_boxes': box_ops.box_cxcywh_to_xyxy(outputs_coord),
            'pred_movable': outputs_movable,
            'pred_rigid': outputs_rigid,
            'pred_kinematic': outputs_kinematic,
            'pred_action': outputs_action,
            'pred_masks': outputs_seg_masks,
            'pred_axis': outputs_axis,
            'pred_depth': outputs_depth,
            'pred_affordance': outputs_aff_masks,
        }

        if not backward:
            return out

        # backward
        src_boxes = outputs_coord # [-1] #.reshape(-1, 4)
        target_boxes = bbox
        target_boxes = box_ops.box_xyxy_to_cxcywh(target_boxes)
        bbox_valid = bbox[:, :, 0] > -0.5
        num_boxes = bbox_valid.sum()
        if num_boxes == 0:
            out['loss_bbox'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_giou'] = torch.tensor(0.0, requires_grad=True).to(device)
        else:
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            loss_bbox = loss_bbox * bbox_valid.unsqueeze(2) # remove invalid
            out['loss_bbox'] = loss_bbox.sum() / num_boxes
        
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes).reshape(-1, 4),
                box_ops.box_cxcywh_to_xyxy(target_boxes).reshape(-1, 4),
            )).reshape(-1, self.num_queries)
            loss_giou = loss_giou * bbox_valid # remove invalid
            out['loss_giou'] = loss_giou.sum() / num_boxes

        # affordance
        affordance_valid = affordance[:, :, 0] > -0.5
        out['loss_affordance'] = torch.tensor(0.0, requires_grad=True)

        # axis
        axis_valid = axis[:, :, 0] > 0.0
        num_axis = axis_valid.sum()
        if num_axis == 0:
            out['loss_axis_angle'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_axis_offset'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_eascore'] = torch.tensor(0.0, requires_grad=True)
        else:
            src_axis_angle = outputs_axis[axis_valid]
            src_axis_angle_norm = F.normalize(src_axis_angle[:, :2])
            src_axis_angle = torch.cat((src_axis_angle_norm, src_axis_angle[:, 2:]), dim=-1)
            target_axis_xyxy = axis[axis_valid]
            
            axis_center = target_boxes[axis_valid].clone()
            axis_center[:, 2:] = axis_center[:, :2] 
            target_axis_angle = axis_ops.line_xyxy_to_angle(target_axis_xyxy, center=axis_center)

            #pdb.set_trace()
            loss_axis_angle = F.l1_loss(src_axis_angle[:, :2], target_axis_angle[:, :2], reduction='sum') / num_axis
            loss_axis_offset = F.l1_loss(src_axis_angle[:, 2:], target_axis_angle[:, 2:], reduction='sum') / num_axis
            #loss_axis = F.l1_loss(src_axis_angle, target_axis_angle, reduction='sum') / num_axis
            out['loss_axis_angle'] = loss_axis_angle
            out['loss_axis_offset'] = loss_axis_offset
            
            src_axis_xyxy = axis_ops.line_angle_to_xyxy(src_axis_angle, center=axis_center)
            target_axis_xyxy = axis_ops.line_angle_to_xyxy(target_axis_angle, center=axis_center)

            axis_eascore, _, _ = axis_ops.ea_score(src_axis_xyxy, target_axis_xyxy)
            loss_eascore = 1 - axis_eascore
            out['loss_eascore'] = loss_eascore.mean()


        loss_movable = F.cross_entropy(outputs_movable.permute(0, 2, 1), movable, ignore_index=self._ignore_index)
        if torch.isnan(loss_movable):
            loss_movable = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_movable'] = loss_movable

        loss_rigid = F.cross_entropy(outputs_rigid.permute(0, 2, 1), rigid, ignore_index=self._ignore_index)
        if torch.isnan(loss_rigid):
            loss_rigid = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_rigid'] = loss_rigid

        loss_kinematic = F.cross_entropy(outputs_kinematic.permute(0, 2, 1), kinematic, ignore_index=self._ignore_index)
        if torch.isnan(loss_kinematic):
            loss_kinematic = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_kinematic'] = loss_kinematic

        loss_action = F.cross_entropy(outputs_action.permute(0, 2, 1), action, ignore_index=self._ignore_index)
        if torch.isnan(loss_action):
            loss_action = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_action'] = loss_action

        # depth backward
        out['loss_depth'] = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_vnl'] = torch.tensor(0.0, requires_grad=True).to(device)
 
        # mask backward
        out['loss_mask'] = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_dice'] = torch.tensor(0.0, requires_grad=True).to(device)


        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]






