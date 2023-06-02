from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class AssociativeEmbeddingModel(torch.nn.Module):
    """
    Implement the Associative Embedding Model.
    
    Alejandro Newell, Zhiao Huang, Jia Deng. 
    Associative embedding: End-to-end learning for joint detection and grouping.
    NeurIPS 2017.

    """
    
    def __init__(
        self,
        encoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        pass 

    def forward(
        self,
        image: torch.Tensor,
        ref_pt: List,
        pos_pt: List,
        neg_pt: List,
    ):
        
        # UNet first
        _, embedding, _, _, _ = self.encoder(image)

        batch_size = embedding.shape[0]
        ref_pt = ref_pt.unsqueeze(1)
        pos_pt = pos_pt.unsqueeze(1)
        neg_pt = neg_pt.unsqueeze(1)
        pts = torch.cat((ref_pt, pos_pt, neg_pt), dim=1)#[None, :]
        xy_sample = pts.view(batch_size, -1, 1, 2).clone()

        embedding_sampled = F.grid_sample(
            embedding,
            xy_sample,
            align_corners=True,
            mode='bilinear'
        )

        anchor = embedding_sampled[:, :, 0, 0]
        positive = embedding_sampled[:, :, 1, 0]
        negative = embedding_sampled[:, :, 2, 0]

        loss = self.triplet_loss(anchor, positive, negative)

        out = {
            'triplet_loss': loss,
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'embedding': embedding,
        }

        return out