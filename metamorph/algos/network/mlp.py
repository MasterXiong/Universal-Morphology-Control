import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from metamorph.config import cfg
from metamorph.utils import model as tu


class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)


class VanillaMLP(nn.Module):
    def __init__(self, obs_space, out_dim):
        super(VanillaMLP, self).__init__()
        self.model_args = copy.deepcopy(cfg.MODEL.MLP)
        self.seq_len = cfg.MODEL.MAX_LIMBS

        # input layer
        self.input_layer = nn.Linear(obs_space["proprioceptive"].shape[0], self.model_args.HIDDEN_DIM)
        # hidden layers
        hidden_dims = [self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM)]
        # output layer
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
            hidden_dims[0] += cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS[-1] 
        self.hidden_layers = tu.make_mlp_default(hidden_dims, dropout=self.model_args.DROPOUT)
        self.output_layer = nn.Linear(self.model_args.HIDDEN_DIM, out_dim)
        # dropout
        if self.model_args.DROPOUT is not None:
            self.input_dropout = nn.Dropout(p=self.model_args.DROPOUT)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, unimal_ids=None):

        batch_size = obs.shape[0]

        # zero-padding limbs won't have zero value due to vector normalization. 
        # Need to explicitly set them as 0 to avoid their influence on the hidden layer computation
        obs = obs.reshape(batch_size, self.seq_len, -1) * (1. - obs_mask.float())[:, :, None]
        obs = obs.reshape(batch_size, -1)
        embedding = self.input_layer(obs)
        embedding /= (1. - obs_mask.float()).sum(dim=1, keepdim=True)
        embedding = F.relu(embedding)
        if self.model_args.DROPOUT is not None:
            embedding = self.input_dropout(embedding)
        # hfield
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            hfield_embedding = self.hfield_encoder(obs_env["hfield"])
            embedding = torch.cat([embedding, hfield_embedding], 1)
        # hidden layers
        embedding = self.hidden_layers(embedding)
        # output layer
        output = self.output_layer(embedding)
        return output, None