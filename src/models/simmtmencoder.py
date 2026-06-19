from torch import nn
import torch
from src.losses.simmtm_losses import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss
from src.models.inceptiontime_pool import *
from src.models.resnet1D import *



class SimMTMEncoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SimMTMEncoder, self).__init__()

        self.feature_extractor = InceptionTime(input_dims, output_dims)
        #self.feature_extractor = ResNet1D(input_dims, output_dims)

        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight()
        self.aggregation = AggregationRebuild()
        self.mse = torch.nn.MSELoss()


    def forward(self, x_in_t, eval=False):

        x_in_t = x_in_t.float()
        z = self.feature_extractor(x_in_t)
        
        if eval:
            return z
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)

        rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x_in_t)

        pred_x = agg_x.reshape(agg_x.size(0), -1)

        loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
        loss = self.awl(loss_cl, loss_rb)

        return loss, z
