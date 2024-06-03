import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metamorph.config import cfg
from metamorph.utils import model as tu
from .transformer import TransformerEncoder, TransformerEncoderLayerResidual
from .gnn import GraphNeuralNetwork


class ContextEncoder(nn.Module):
    def __init__(self, obs_space):
        super(ContextEncoder, self).__init__()
        self.model_args = cfg.MODEL.HYPERNET
        self.seq_len = cfg.MODEL.MAX_LIMBS
        context_obs_size = obs_space["context"].shape[0] // self.seq_len

        if self.model_args.CONTEXT_ENCODER_TYPE == 'linear':
            context_encoder_dim = [context_obs_size] + [self.model_args.CONTEXT_EMBED_SIZE for _ in range(self.model_args.ENCODER_LAYER_NUM)]
            self.context_encoder = tu.make_mlp_default(context_encoder_dim)
        elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
            context_embed = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            context_encoder_layers = TransformerEncoderLayerResidual(
                self.model_args.CONTEXT_EMBED_SIZE,
                self.model_args.CONTEXT_TF_ENCODER_NHEAD,
                self.model_args.CONTEXT_TF_ENCODER_FF_DIM,
                0., 
                batch_first=True, 
            )
            context_encoder_TF = TransformerEncoder(
                context_encoder_layers, 1, norm=None,
            )
            if self.model_args.CONTEXT_MASK:
                self.context_embed = context_embed
                self.context_encoder = context_encoder_TF
            else:
                self.context_encoder = nn.Sequential(
                    context_embed, 
                    context_encoder_TF, 
                )
        elif self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
            self.context_embed_input = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            self.context_encoder_for_input = GraphNeuralNetwork(
                dim=self.model_args.CONTEXT_EMBED_SIZE, 
                num_layer=self.model_args.ENCODER_LAYER_NUM, 
                final_nonlinearity=False, 
            )

        if self.model_args.EMBEDDING_DROPOUT is not None:
            self.embedding_dropout = nn.Dropout(p=self.model_args.EMBEDDING_DROPOUT)

    def forward(self, obs_context, obs_mask, morphology_info=None):
        context_embedding = obs_context
        if self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
            if self.model_args.CONTEXT_MASK:
                context_embedding = self.context_embed(context_embedding)
                context_embedding = self.context_encoder(context_embedding, src_key_padding_mask=obs_mask)
            else:
                context_embedding = self.context_encoder(context_embedding)
        elif self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
            context_embedding = self.context_embed(context_embedding)
            context_embedding = self.context_encoder(context_embedding, morphology_info["adjacency_matrix"])
        else:
            context_embedding = self.context_encoder(context_embedding)
        if self.model_args.EMBEDDING_DROPOUT is not None:
            context_embedding = self.embedding_dropout(context_embedding)
        return context_embedding


class HypernetLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, init_dim=None):
        super(HypernetLayer, self).__init__()
        self.base_input_dim = base_input_dim
        self.base_output_dim = base_output_dim
        self.init_dim = init_dim
        self.model_args = cfg.MODEL.HYPERNET
        HN_input_dim = self.model_args.CONTEXT_EMBED_SIZE

        self.HN_weight = nn.Linear(HN_input_dim, base_input_dim * base_output_dim)
        self.HN_bias = nn.Linear(HN_input_dim, base_output_dim)

        self.init_hypernet()

    def init_hypernet(self):
        if self.model_args.HN_INIT_STRATEGY == 'bias_init':
            initrange = np.sqrt(1 / self.init_dim)
            self.HN_weight.weight.data.zero_()
            self.HN_weight.bias.data.normal_(std=initrange)
        elif self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
            initrange = np.sqrt(1 / self.init_dim)
            self.HN_weight.weight.data.zero_()
            self.HN_weight.bias.data.uniform_(-initrange, initrange)
        else:
            # use a heuristic value as the init range
            initrange = 0.001
            self.HN_weight.weight.data.uniform_(-initrange, initrange)
            self.HN_weight.bias.data.zero_()

        if self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
            initrange = np.sqrt(1 / self.init_dim)
            self.HN_bias.weight.data.zero_()
            self.HN_bias.bias.data.uniform_(-initrange, initrange)
        else:
            self.HN_bias.weight.data.zero_()
            self.HN_bias.bias.data.zero_()
    
    def forward(self, context_embedding):
        weight = self.HN_weight(context_embedding)
        bias = self.HN_bias(context_embedding)
        return weight, bias


class HNMLP(nn.Module):
    def __init__(self, obs_space, out_dim):
        super(HNMLP, self).__init__()
        self.seq_len = cfg.MODEL.MAX_LIMBS
        self.limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.limb_out_dim = out_dim // self.seq_len
        self.HN_args = cfg.MODEL.HYPERNET
        self.base_args = cfg.MODEL.MLP

        self.input_context_encoder = ContextEncoder(obs_space)
        self.input_HN_layer = HypernetLayer(self.limb_obs_size, self.base_args.HIDDEN_DIM, init_dim=obs_space["proprioceptive"].shape[0])

        if not self.HN_args.SHARE_CONTEXT_ENCODER:
            self.output_context_encoder = ContextEncoder(obs_space)
        self.output_HN_layer = HypernetLayer(self.base_args.HIDDEN_DIM, self.limb_out_dim, init_dim=self.base_args.HIDDEN_DIM)

        if self.base_args.LAYER_NUM > 1:
            if not self.HN_args.SHARE_CONTEXT_ENCODER:
                self.hidden_context_encoder = ContextEncoder(obs_space)
            self.hidden_HN_layers = []
            self.hidden_dims = [self.base_args.HIDDEN_DIM for _ in range(self.base_args.LAYER_NUM)]
            if "hfield" in cfg.ENV.KEYS_TO_KEEP:
                self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
                self.hidden_dims[0] += cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS[-1]
            for i in range(self.base_args.LAYER_NUM - 1):
                layer = HypernetLayer(self.hidden_dims[i], self.hidden_dims[i + 1], init_dim=self.hidden_dims[i])
                self.hidden_HN_layers.append(layer)
            self.hidden_HN_layers = nn.ModuleList(self.hidden_HN_layers)

    def generate_params(self, obs_context, obs_mask, morphology_info=None):

        batch_size = obs_context.shape[0]
        obs_context = obs_context.view(batch_size, self.seq_len, -1)

        # input layer
        input_context_embedding = self.input_context_encoder(obs_context, obs_mask)
        self.input_weight, self.input_bias = self.input_HN_layer(input_context_embedding)
        self.input_weight = self.input_weight.view(batch_size, self.seq_len, self.limb_obs_size, self.base_args.HIDDEN_DIM)

        # hidden layers
        if self.base_args.LAYER_NUM > 1:
            if self.HN_args.SHARE_CONTEXT_ENCODER:
                hidden_context_embedding = input_context_embedding
            else:
                hidden_context_embedding = self.hidden_context_encoder(obs_context, obs_mask)
            hidden_context_embedding = (hidden_context_embedding * (1. - obs_mask.float())[:, :, None]).sum(dim=1) / (1. - obs_mask.float()).sum(dim=1, keepdim=True)
            self.hidden_weights, self.hidden_bias = [], []
            for i, HN_layer in enumerate(self.hidden_HN_layers):
                weight, bias = HN_layer(hidden_context_embedding)
                weight = weight.view(batch_size, self.hidden_dims[i], self.hidden_dims[i + 1])
                self.hidden_weights.append(weight)
                self.hidden_bias.append(bias)

        # output layer            
        if self.HN_args.SHARE_CONTEXT_ENCODER:
            output_context_embedding = input_context_embedding
        else:
            output_context_embedding = self.output_context_encoder(obs_context, obs_mask)
        self.output_weight, self.output_bias = self.output_HN_layer(output_context_embedding)
        self.output_weight = self.output_weight.view(batch_size, self.seq_len, self.base_args.HIDDEN_DIM, self.limb_out_dim)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, unimal_ids=None):

        # we only need to generate weights once for evaluation
        if self.training:
            self.generate_params(obs_context, obs_mask)

        batch_size = obs.shape[0]
        obs = obs.view(batch_size, self.seq_len, -1)

        # input layer
        embedding = (obs[:, :, :, None] * self.input_weight).sum(dim=-2) + self.input_bias
        embedding = embedding * (1. - obs_mask.float())[:, :, None]
        # aggregate all limbs' embedding
        if self.HN_args.INPUT_AGGREGATION == 'limb_num':
            embedding = embedding.sum(dim=1) / (1. - obs_mask.float()).sum(dim=1, keepdim=True)
        elif self.HN_args.INPUT_AGGREGATION == 'sqrt_limb_num':
            embedding = embedding.sum(dim=1) / torch.sqrt((1. - obs_mask.float()).sum(dim=1, keepdim=True))
        elif self.HN_args.INPUT_AGGREGATION == 'max_limb_num':
            embedding = embedding.mean(dim=1)
        else:
            embedding = embedding.sum(dim=1)
        embedding = F.relu(embedding)

        # hidden layers
        for weight, bias in zip(self.hidden_weights, self.hidden_bias):
            embedding = (embedding[:, :, None] * weight).sum(dim=1) + bias
            embedding = F.relu(embedding)

        # output layer
        output = (embedding[:, None, :, None] * self.output_weight).sum(dim=-2) + self.output_bias
        output = output.reshape(batch_size, -1)
        return output, None
