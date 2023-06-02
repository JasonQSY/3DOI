from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pdb

from .detr import box_ops
from .detr.segmentation import dice_loss, sigmoid_focal_loss
from .detr.misc import nested_tensor_from_tensor_list, interpolate
from . import axis_ops


class CohesiveModel(torch.nn.Module):
    """
    Implement Cohesive model.
    
    Dandan Shan, Richard Higgins, David Fouhey.
    COHESIV: Contrastive Object and Hand Embedding Segmentation In Video
    NeurIPS 2021.

    """
    
    def __init__(
        self,
        encoder,
        image_size=[192, 256],
        output_size=[192, 256],
        fusion='concat',
        loss_mask_type='ce',
        ignore_index=-100,
        affordance_focal_alpha=0.95,
    ):
        """
        fusion can be 'concat', 'multiply'
        """
        super().__init__()
        self.encoder = encoder
        self.n_embedding = encoder.n_embedding
        self._image_size = image_size
        self._output_size = output_size
        self._fusion = fusion
        self._ignore_index = ignore_index
        self._affordance_focal_alpha = affordance_focal_alpha

        channel = self.n_embedding

        # predict point-level properties of the keypoints
        self.properties = {
            'movable': 3,
            'rigid': 2,
            'kinematic': 3,
            'action': 3,
            'axis': 3,
        }
        self.property_branch = nn.ModuleDict()
        for property in self.properties:
            out_channels = self.properties[property]
            self.property_branch[property] = nn.Sequential(
                nn.Conv2d(channel, channel, (3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_channels, (3, 3), padding=1),
            )

        # movable branch
        # self.category_branch = nn.Sequential(
        #     nn.Conv2d(channel, channel, (3, 3), padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel, 3, (3, 3), padding=1),
        # )

        # depth branch
        self.depth_branch = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, (3, 3), padding=1),
        )

        # # bbox branch
        # self.bbox_branch = nn.Sequential(
        #     nn.Conv2d(channel, channel, (3, 3), padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel, channel, (3, 3), padding=1),
        # )
        # self.bbox_fc = nn.Sequential(
        #     nn.Linear(channel * 2, channel),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel, channel),
        # )

        # anchor branch
        self.anchor_branch = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 3), padding=1),
        )
        
        # embedding branch
        self.embedding_branch = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 3), padding=1),
        )

        # affordance embedding branch
        self.affordance_branch = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 3), padding=1),
        )
        

        # generate final predictions
        if self._fusion == 'concat':
            self.refine_branch = nn.Sequential(
                nn.Conv2d(channel * 2, channel, (3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, (3, 3), padding=1),
                nn.Conv2d(channel, 1, (1, 1), padding=0),
            )

            self.affordance_refine_branch = nn.Sequential(
                nn.Conv2d(channel * 2, channel, (3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, (3, 3), padding=1),
                nn.Conv2d(channel, 1, (1, 1), padding=0),
            )
        elif self._fusion == 'multiply':
            self.refine_branch = nn.Sequential(
                nn.Conv2d(channel, channel, (3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, (3, 3), padding=1),
                nn.Conv2d(channel, 1, (1, 1), padding=0),
            )

            self.affordance_refine_branch = nn.Sequential(
                nn.Conv2d(channel, channel, (3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, (3, 3), padding=1),
                nn.Conv2d(channel, 1, (1, 1), padding=0),
            )
        else:
            raise NotImplementedError

        #self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        #self.re_loss = nn.BCEWithLogitsLoss()
        self.re_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        #self.category_loss = nn.CrossEntropyLoss()
        self._loss_mask_type = loss_mask_type
        
        pass 


    def cohesive_forward(
        self,
        anchors_float: torch.Tensor,
        embedding: torch.Tensor,
        anchor_layers,
        embedding_layers,
        refinement_layers,
    ):
        """
        1. Extract anchor embeddings from anchor layers
        2. Extract mask embeddings from embeddings
        3. Fuse them together and generate the final output.
        """
        # output = headPredict(cat(tile(Q[i,j,:],H,W),K))
        anchor_map = anchor_layers(embedding)
        embedding_map = embedding_layers(embedding)

        anchor_map = F.normalize(anchor_map)
        embedding_map = F.normalize(embedding_map)

        q = F.grid_sample(anchor_map.repeat(self.num_queries, 1, 1, 1), anchors_float, mode='nearest', align_corners=True)
        #
        q = q.tile((1, 1, *self._image_size)) # (self.num_queries, channel, h, w)
        #pdb.set_trace()
        k = embedding_map
        k = k.repeat(self.num_queries, 1, 1, 1)
        if self._fusion == 'concat':
            output = torch.cat((q, k), dim=1)
        elif self._fusion == 'multiply':
            output = q * k

        output = refinement_layers(output)
        
        return output

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
        keypoints: torch.Tensor,
        bbox: torch.Tensor,
        bbox_to_mask: torch.Tensor,
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
        """
        Right now it only supports batch_size = 1.
        """
        device = image.device
        
        # UNet backbone
        _, embedding, _, _, _ = self.encoder(image)

        # return dict
        out = {}

        # keypoint
        self.num_queries = keypoints.shape[1]
        anchors = keypoints[0].float()
        anchors_float = anchors.clone()
        anchors_float[:, 0] = ((anchors[:, 0] / self._image_size[1]) - 0.5) * 2
        anchors_float[:, 1] = ((anchors[:, 1] / self._image_size[0]) - 0.5) * 2
        anchors_float = anchors_float.unsqueeze(1).unsqueeze(1)

        # 1. Point-level Properties
        for property in self.properties:
            property_map = self.property_branch[property](embedding)
            property_pred = F.grid_sample(property_map.repeat(self.num_queries, 1, 1, 1), anchors_float, mode='nearest', align_corners=True)
            property_pred = property_pred[:, :, 0, 0]

            # add to return dict
            if property == 'axis':
                #pred_axis = property_pred.sigmoid().unsqueeze(0)
                pred_axis = property_pred.unsqueeze(0)
                out['pred_' + property] = pred_axis
            else:
                out['pred_' + property] = property_pred.unsqueeze(0)

            # loss
            if backward:
                if property == 'axis':
                    axis_valid = axis[:, :, 0] > 0.0
                    num_axis = axis_valid.sum()
                    if num_axis == 0:
                        out['loss_axis_angle'] = torch.tensor(0.0, requires_grad=True).to(device)
                        out['loss_axis_offset'] = torch.tensor(0.0, requires_grad=True).to(device)
                        out['loss_eascore'] = torch.tensor(0.0, requires_grad=True).to(device)
                        continue

                    src_axis_angle = pred_axis[axis_valid]
                    src_axis_angle_norm = F.normalize(src_axis_angle[:, :2])
                    src_axis_angle = torch.cat((src_axis_angle_norm, src_axis_angle[:, 2:]), dim=-1)
                    target_axis_xyxy = axis[axis_valid]
                    
                    axis_center = bbox[axis_valid].clone()
                    axis_center[:, 2:] = axis_center[:, :2] 
                    target_axis_angle = axis_ops.line_xyxy_to_angle(target_axis_xyxy, center=axis_center)

                    loss_axis_angle = F.l1_loss(src_axis_angle[:, :2], target_axis_angle[:, :2], reduction='sum') / num_axis
                    loss_axis_offset = F.l1_loss(src_axis_angle[:, 2:], target_axis_angle[:, 2:], reduction='sum') / num_axis
                    out['loss_axis_angle'] = loss_axis_angle
                    out['loss_axis_offset'] = loss_axis_offset * 20
                    
                    src_axis_xyxy = axis_ops.line_angle_to_xyxy(src_axis_angle, center=axis_center)
                    target_axis_xyxy = axis_ops.line_angle_to_xyxy(target_axis_angle, center=axis_center)

                    axis_eascore, _, _ = axis_ops.ea_score(src_axis_xyxy, target_axis_xyxy)
                    loss_eascore = 1 - axis_eascore
                    out['loss_eascore'] = loss_eascore.mean()

                    # out['loss_axis_angle'] = torch.tensor(0.0, requires_grad=True).to(device)
                    # out['loss_axis_offset'] = torch.tensor(0.0, requires_grad=True).to(device)
                    # out['loss_eascore'] = torch.tensor(0.0, requires_grad=True).to(device)
                    continue

                property_loss = F.cross_entropy(property_pred, eval(property)[0], ignore_index=self._ignore_index)
                out['loss_' + property] = property_loss

        # 2. Object and Affordance map
        pred_mask = self.cohesive_forward(
            anchors_float, 
            embedding, 
            self.anchor_branch, 
            self.embedding_branch, 
            self.refine_branch
        )
        pred_mask = pred_mask[:, 0].unsqueeze(0)
        pred_mask = interpolate(pred_mask, size=tuple(self._output_size), mode='nearest')
        out['pred_masks'] = pred_mask
        
        if backward:
            tgt_masks = masks
            src_masks = pred_mask
            valid_mask = tgt_masks.sum(dim=-1).sum(dim=-1) > 10
            if valid_mask.sum() == 0:
                out['loss_mask'] = torch.tensor(0.0, requires_grad=True).to(device)
                out['loss_dice'] = torch.tensor(0.0, requires_grad=True).to(device)
            else:
                num_masks = valid_mask.sum()
                src_masks = src_masks[valid_mask]
                tgt_masks = tgt_masks[valid_mask]
                src_masks = src_masks.flatten(1)
                tgt_masks = tgt_masks.flatten(1)
                tgt_masks = tgt_masks.view(src_masks.shape)

                if self._loss_mask_type == 'focal':
                    out['loss_mask'] = sigmoid_focal_loss(src_masks, tgt_masks.float(), num_masks)
                    out['loss_dice'] = dice_loss(src_masks, tgt_masks, num_masks)
                elif self._loss_mask_type == 'ce':
                    out['loss_dice'] = torch.tensor(0.0, requires_grad=True).to(device)
                    loss_mask = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction='none')
                    loss_mask = loss_mask.mean(dim=1).sum() / num_masks
                    out['loss_mask'] = loss_mask
                else:
                    raise ValueError(self._loss_mask_type)


        pred_mask_bbox = (pred_mask.clone() > 0.0).long() 
        empty_mask = pred_mask_bbox.sum(dim=-1).sum(dim=-1)
        pred_mask_bbox[empty_mask == 0] += 1
        pred_boxes = torchvision.ops.masks_to_boxes(pred_mask_bbox[0])
        #pred_boxes = box_ops.rescale_bboxes(pred_boxes, [1 / self._image_size[1], 1 / self._image_size[0]])
        pred_boxes = box_ops.rescale_bboxes(pred_boxes, [1 / 768, 1 / 1024])
        out['pred_boxes'] = pred_boxes.unsqueeze(0)
        if backward:
            #out['loss_bbox'] = F.cross_entropy(out['pred_mask'], bbox_to_mask[0], ignore_index=self._ignore_index)
            out['loss_bbox'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_giou'] = torch.tensor(0.0, requires_grad=True).to(device)

        

        pred_affordance = self.cohesive_forward(
            anchors_float, 
            embedding, 
            self.anchor_branch, 
            self.affordance_branch, 
            self.affordance_refine_branch
        )
        pred_affordance = pred_affordance[:, 0].unsqueeze(0)
        affordance_size = (self._output_size[0] // 4, self._output_size[1] // 4)
        pred_affordance = interpolate(pred_affordance, size=affordance_size, mode='nearest')
        out['pred_affordance'] = pred_affordance

        
        if backward:
            affordance_valid = affordance[:, :, 0] > -0.5
            if affordance_valid.sum() == 0:
                out['loss_affordance'] = torch.tensor(0.0, requires_grad=True).to(device)
            else:
                src_aff_masks = pred_affordance[affordance_valid]
                tgt_aff_masks = affordance_map[affordance_valid]
                
                src_aff_masks = src_aff_masks.flatten(1)
                tgt_aff_masks = tgt_aff_masks.flatten(1)
                loss_aff = sigmoid_focal_loss(
                    src_aff_masks,
                    tgt_aff_masks,
                    affordance_valid.sum(),
                    alpha=self._affordance_focal_alpha,
                )
                out['loss_affordance'] = loss_aff

        #out['loss_affordance'] = loss_aff

        #pdb.set_trace()


        # anchor_map = self.anchor_branch(embedding)
        # embedding_map = self.embedding_branch(embedding)
        # anchor_map = F.normalize(anchor_map)
        # embedding_map = F.normalize(embedding_map)
        # q = F.grid_sample(anchor_map, anchors_float, mode='nearest', align_corners=True)
        # q = q.tile((1, 1, *self._image_size)) # (1, channel, h, w)
        # k = embedding_map
        # if self._fusion == 'concat':
        #     output = torch.cat((q, k), dim=1)
        # elif self._fusion == 'multiply':
        #     output = q * k
        # output = self.refine_branch(output)
        # out['mask_pred'] = output

        # TODO: 3. Image-level Properties (depth)
        # depth_map = self.depth_branch(embedding)
        out['pred_depth'] = torch.zeros((1, 1, 192, 256)).to(device)
        if backward:
            out['loss_depth'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_vnl'] = torch.tensor(0.0, requires_grad=True).to(device)

        #pdb.set_trace()

        #category_loss = self.category_loss(category_pred, category[:, 0])
        
        return out