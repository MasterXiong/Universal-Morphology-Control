import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from metamorph.config import cfg
from metamorph.utils import model as tu

from metamorph.algos.network.transformer import TransformerEncoder
from metamorph.algos.network.transformer import TransformerEncoderLayerResidual
from metamorph.algos.network.gnn import GraphNeuralNetwork
from metamorph.algos.network.mlp import VanillaMLP
from metamorph.algos.network.hypernet import HNMLP

import time
import matplotlib.pyplot as plt


# J: Max num joints between two limbs. 1 for 2D envs, 2 for unimal
class TransformerModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super(TransformerModel, self).__init__()

        self.decoder_out_dim = decoder_out_dim

        self.model_args = cfg.MODEL.TRANSFORMER
        self.seq_len = cfg.MODEL.MAX_LIMBS
        # Embedding layer for per limb obs
        limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE
        if self.model_args.PER_NODE_EMBED:
            print ('independent weights for each node')
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.limb_embed_weights = nn.Parameter(torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), limb_obs_size, self.d_model).uniform_(-initrange, initrange))
            self.limb_embed_bias = nn.Parameter(torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), self.d_model))
        else:
            self.limb_embed = nn.Linear(limb_obs_size, self.d_model)
        self.ext_feat_fusion = self.model_args.EXT_MIX

        if self.model_args.POS_EMBEDDING == "learnt":
            print ('use PE learnt')
            seq_len = self.seq_len
            self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        elif self.model_args.POS_EMBEDDING == "abs":
            print ('use PE abs')
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            cfg.MODEL.LIMB_EMBED_SIZE,
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.model_args.NLAYERS, norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model

        # Task based observation encoder
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])

        if self.ext_feat_fusion == "late":
            decoder_input_dim += self.hfield_encoder.obs_feat_dim
        self.decoder_input_dim = decoder_input_dim

        if self.model_args.PER_NODE_DECODER:
            # only support a single output layer
            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            self.decoder_weights = torch.zeros(decoder_input_dim, decoder_out_dim).uniform_(-initrange, initrange)
            self.decoder_weights = self.decoder_weights.repeat(self.seq_len, len(cfg.ENV.WALKERS), 1, 1)
            self.decoder_weights = nn.Parameter(self.decoder_weights)
            self.decoder_bias = torch.zeros(decoder_out_dim).uniform_(-initrange, initrange)
            self.decoder_bias = self.decoder_bias.repeat(self.seq_len, len(cfg.ENV.WALKERS), 1)
            self.decoder_bias = nn.Parameter(self.decoder_bias)
        else:
            self.decoder = tu.make_mlp_default(
                [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim],
                final_nonlinearity=False,
            )

        if self.model_args.FIX_ATTENTION:
            print ('use fix attention')
            # the network to generate context embedding from the morphology context
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_attention = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)

            if self.model_args.CONTEXT_ENCODER == 'transformer':
                print ('use transformer context encoder')
                context_encoder_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.context_encoder_attention = TransformerEncoder(
                    context_encoder_layers, self.model_args.CONTEXT_LAYER, norm=None,
                )
            else: # MLP context encoder: the default choice
                print ('use MLP context encoder')
                modules = [nn.ReLU()]
                for _ in range(self.model_args.LINEAR_CONTEXT_LAYER):
                    modules.append(nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE))
                    modules.append(nn.ReLU())
                self.context_encoder_attention = nn.Sequential(*modules)

            # whether to include hfield in attention map computation: default to False
            if self.model_args.HFIELD_IN_FIX_ATTENTION:
                self.context_hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
                self.context_compress = nn.Sequential(
                    nn.Linear(self.model_args.EXT_HIDDEN_DIMS[-1] + self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE), 
                    nn.ReLU(), 
                )

        if self.model_args.HYPERNET:
            print ('use HN')
            # the network to generate context embedding from the morphology context
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_HN = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            
            if self.model_args.HN_CONTEXT_ENCODER == 'linear': # the default architecture choice
                modules = [nn.ReLU()]
                for _ in range(self.model_args.HN_CONTEXT_LAYER_NUM):
                    modules.append(nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE))
                    modules.append(nn.ReLU())
                self.context_encoder_HN = nn.Sequential(*modules)
            elif self.model_args.HN_CONTEXT_ENCODER == 'transformer':
                context_encoder_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.context_encoder_HN = TransformerEncoder(
                    context_encoder_layers, self.model_args.HN_CONTEXT_LAYER_NUM, norm=None,
                )

            HN_input_dim = self.model_args.CONTEXT_EMBED_SIZE

            self.hnet_embed_weight = nn.Linear(HN_input_dim, limb_obs_size * self.d_model)
            self.hnet_embed_bias = nn.Linear(HN_input_dim, self.d_model)

            self.decoder_dims = [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim]

            self.hnet_decoder_weight = []
            self.hnet_decoder_bias = []
            for i in range(len(self.decoder_dims) - 1):
                layer_w = nn.Linear(HN_input_dim, self.decoder_dims[i] * self.decoder_dims[i + 1])
                self.hnet_decoder_weight.append(layer_w)
                layer_b = nn.Linear(HN_input_dim, self.decoder_dims[i + 1])
                self.hnet_decoder_bias.append(layer_b)
            self.hnet_decoder_weight = nn.ModuleList(self.hnet_decoder_weight)
            self.hnet_decoder_bias = nn.ModuleList(self.hnet_decoder_bias)

        # whether to use SWAT PE and RE: default to False
        if self.model_args.USE_SWAT_PE:
            self.swat_PE_encoder = SWATPEEncoder(self.d_model, self.seq_len)

        self.dropout = nn.Dropout(p=0.1)

        self.init_weights()

    def init_weights(self):
        # init obs embedding
        if not self.model_args.PER_NODE_EMBED:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.limb_embed.weight.data.uniform_(-initrange, initrange)
        # init decoder
        if not self.model_args.PER_NODE_DECODER:
            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            self.decoder[-1].bias.data.zero_()
            self.decoder[-1].weight.data.uniform_(-initrange, initrange)

        if self.model_args.FIX_ATTENTION:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.context_embed_attention.weight.data.uniform_(-initrange, initrange)

        if self.model_args.HYPERNET:
            initrange = cfg.MODEL.TRANSFORMER.HN_EMBED_INIT
            self.context_embed_HN.weight.data.uniform_(-initrange, initrange)

            # initialize the hypernet following https://arxiv.org/abs/2210.11348
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.hnet_embed_weight.weight.data.zero_()
            self.hnet_embed_weight.bias.data.uniform_(-initrange, initrange)
            self.hnet_embed_bias.weight.data.zero_()
            self.hnet_embed_bias.bias.data.zero_()

            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            for i in range(len(self.hnet_decoder_weight)):
                self.hnet_decoder_weight[i].weight.data.zero_()
                self.hnet_decoder_weight[i].bias.data.uniform_(-initrange, initrange)
                self.hnet_decoder_bias[i].weight.data.zero_()
                self.hnet_decoder_bias[i].bias.data.zero_()

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, dropout_mask=None, unimal_ids=None):
        # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, d_model)
        _, batch_size, limb_obs_size = obs.shape

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            # (batch_size, embed_size)
            hfield_obs = self.hfield_encoder(obs_env["hfield"])

        if self.ext_feat_fusion in ["late"]:
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)

        if self.model_args.FIX_ATTENTION:
            context_embedding_attention = self.context_embed_attention(obs_context)

            if self.model_args.CONTEXT_ENCODER == 'transformer':
                context_embedding_attention = self.context_encoder_attention(
                    context_embedding_attention, 
                    src_key_padding_mask=obs_mask, 
                    morphology_info=morphology_info)
            else:
                context_embedding_attention = self.context_encoder_attention(context_embedding_attention)

            if self.model_args.HFIELD_IN_FIX_ATTENTION:
                hfield_embedding = self.context_hfield_encoder(obs_env["hfield"])
                hfield_embedding = hfield_embedding.repeat(self.seq_len, 1).reshape(self.seq_len, batch_size, -1)
                context_embedding_attention = torch.cat([context_embedding_attention, hfield_embedding], dim=-1)
                context_embedding_attention = self.context_compress(context_embedding_attention)

        if self.model_args.HYPERNET:
            context_embedding_HN = self.context_embed_HN(obs_context)
            context_embedding_HN = self.context_encoder_HN(context_embedding_HN)

        if self.model_args.HYPERNET and self.model_args.HN_EMBED:
            embed_weight = self.hnet_embed_weight(context_embedding_HN).reshape(self.seq_len, batch_size, limb_obs_size, self.d_model)
            embed_bias = self.hnet_embed_bias(context_embedding_HN)
            obs_embed = (obs[:, :, :, None] * embed_weight).sum(dim=-2, keepdim=False) + embed_bias
        else:
            if self.model_args.PER_NODE_EMBED:
                obs_embed = (obs[:, :, :, None] * self.limb_embed_weights[:, unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.limb_embed_bias[:, unimal_ids, :]
            else:
                obs_embed = self.limb_embed(obs)
        
        if self.model_args.EMBEDDING_SCALE: # default to true
            obs_embed *= math.sqrt(self.d_model)

        attention_maps = None

        # add PE
        if self.model_args.POS_EMBEDDING in ["learnt", "abs"]:
            obs_embed = self.pos_embedding(obs_embed)
        if self.model_args.USE_SWAT_PE:
            obs_embed = self.swat_PE_encoder(obs_embed, morphology_info['traversals'])

        # dropout
        if self.model_args.EMBEDDING_DROPOUT:
            if self.model_args.CONSISTENT_DROPOUT:
                # do dropout in a consistent way. Refer to Appendix in the paper
                if dropout_mask is None:
                    obs_embed_after_dropout = self.dropout(obs_embed)
                    dropout_mask = torch.where(obs_embed_after_dropout == 0., 0., 1.).permute(1, 0, 2)
                    obs_embed = obs_embed_after_dropout
                else:
                    obs_embed = obs_embed * dropout_mask.permute(1, 0, 2) / 0.9
            else:
                # do dropout in an inconsistent way, as in MetaMorph
                obs_embed = self.dropout(obs_embed)
                dropout_mask = 0.
        else:
            # do not do dropout
            dropout_mask = 0.

        if self.model_args.FIX_ATTENTION:
            context_to_base = context_embedding_attention
        else:
            context_to_base = None
        
        if self.model_args.USE_SWAT_RE:
            attn_mask = morphology_info['SWAT_RE']
        else:
            attn_mask = None
        src_key_padding_mask = obs_mask

        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, 
                mask=attn_mask, 
                src_key_padding_mask=src_key_padding_mask, 
                context=context_to_base, 
                morphology_info=morphology_info
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                obs_embed, 
                mask=attn_mask, 
                src_key_padding_mask=src_key_padding_mask, 
                context=context_to_base, 
                morphology_info=morphology_info
            )
        
        decoder_input = obs_embed_t
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

        # (num_limbs, batch_size, J)
        if self.model_args.HYPERNET and self.model_args.HN_DECODER:
            output = decoder_input
            layer_num = len(self.hnet_decoder_weight)
            for i in range(layer_num):
                layer_w = self.hnet_decoder_weight[i](context_embedding_HN).reshape(self.seq_len, batch_size, self.decoder_dims[i], self.decoder_dims[i + 1])
                layer_b = self.hnet_decoder_bias[i](context_embedding_HN)
                output = (output[:, :, :, None] * layer_w).sum(dim=-2, keepdim=False) + layer_b
                if i != (layer_num - 1):
                    output = F.relu(output)
        else:
            if self.model_args.PER_NODE_DECODER:
                output = (decoder_input[:, :, :, None] * self.decoder_weights[:, unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.decoder_bias[:, unimal_ids, :]
            else:
                output = self.decoder(decoder_input)

        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)

        return output, attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0., batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if batch_first:
            self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
        else:
            self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return x


class SWATPEEncoder(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe_dim = [d_model // len(cfg.MODEL.TRANSFORMER.TRAVERSALS) for _ in cfg.MODEL.TRANSFORMER.TRAVERSALS]
        self.pe_dim[-1] = d_model - self.pe_dim[0] * (len(cfg.MODEL.TRANSFORMER.TRAVERSALS) - 1)
        print (self.pe_dim)
        self.swat_pe = nn.ModuleList([nn.Embedding(seq_len, dim) for dim in self.pe_dim])

    def forward(self, x, indexes):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        embeddings = []
        batch_size = x.size(1)
        for i in range(len(cfg.MODEL.TRANSFORMER.TRAVERSALS)):
            idx = indexes[:, :, i]
            pe = self.swat_pe[i](idx)
            embeddings.append(pe)
        embeddings = torch.cat(embeddings, dim=-1)
        x = x + embeddings
        return x


class GNNModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super(GNNModel, self).__init__()

        self.model_args = cfg.MODEL.GNN
        self.seq_len = cfg.MODEL.MAX_LIMBS
        # Embedding layer for per limb obs
        limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE
        self.limb_embed = nn.Linear(limb_obs_size, self.d_model)

        # GNN encoder layers
        self.GNN_encoder = GraphNeuralNetwork(self.d_model, self.model_args.LAYER_NUM, final_nonlinearity=True)

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model

        # Task based observation encoder
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
            decoder_input_dim += self.hfield_encoder.obs_feat_dim

        self.decoder = tu.make_mlp_default(
            [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim],
            final_nonlinearity=False,
        )

        self.init_weights()

    def init_weights(self):
        # init obs embedding
        initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
        self.limb_embed.weight.data.uniform_(-initrange, initrange)
        # init decoder
        initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
        self.decoder[-1].bias.data.zero_()
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, dropout_mask=None, unimal_ids=None):
        batch_size, _, limb_obs_size = obs.shape

        obs_embed = self.limb_embed(obs)

        obs_embed_t = self.GNN_encoder(obs_embed, morphology_info['adjacency_matrix'])
        
        decoder_input = obs_embed_t
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

        output = self.decoder(decoder_input)

        output = output.reshape(batch_size, -1)

        return output, None


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        print (obs_space)
        self.seq_len = cfg.MODEL.MAX_LIMBS
        if cfg.DISTILL.VALUE_NET:
            if cfg.MODEL.TYPE == 'transformer':
                self.v_net = TransformerModel(obs_space, 1)
            elif cfg.MODEL.TYPE == 'hnmlp':
                self.v_net = HNMLP(obs_space, cfg.MODEL.MAX_LIMBS)
            elif cfg.MODEL.TYPE == 'gnn':
                self.v_net = GNNModel(obs_space, 1)
            else:
                self.v_net = VanillaMLP(obs_space, cfg.MODEL.MAX_LIMBS)

        if cfg.ENV_NAME == "Unimal-v0":
            if cfg.MODEL.TYPE == 'transformer':
                self.mu_net = TransformerModel(obs_space, 2)
            elif cfg.MODEL.TYPE == 'hnmlp':
                self.mu_net = HNMLP(obs_space, cfg.MODEL.MAX_LIMBS * 2)
            elif cfg.MODEL.TYPE == 'gnn':
                self.mu_net = GNNModel(obs_space, 2)
            else:
                self.mu_net = VanillaMLP(obs_space, cfg.MODEL.MAX_LIMBS * 2)
            self.num_actions = cfg.MODEL.MAX_LIMBS * 2
        # elif cfg.ENV_NAME == 'Modular-v0':
        #     if cfg.MODEL.TYPE == 'transformer':
        #         self.mu_net = TransformerModel(obs_space, 1)
        #     else:
        #         self.mu_net = MLPModel(obs_space, cfg.MODEL.MAX_LIMBS)
        #     self.num_actions = cfg.MODEL.MAX_LIMBS
        else:
            raise ValueError("Unsupported ENV_NAME")

        if cfg.MODEL.ACTION_STD_FIXED:
            log_std = np.log(cfg.MODEL.ACTION_STD)
            self.log_std = nn.Parameter(
                log_std * torch.ones(1, self.num_actions), requires_grad=False,
            )
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))
        
        # hard code the index of context features if they are included in proprioceptive features
        limb_context_index = np.arange(13, 13 + 17)
        # two joints features for each node
        joint_context_index = np.concatenate([np.arange(2, 2 + 9), np.arange(11 + 2, 11 + 2 + 9)]) + 30
        self.context_index = np.concatenate([limb_context_index, joint_context_index])
        print ('context index', self.context_index)

        # hard code fix-range normalization for state inputs
        if cfg.ENV_NAME == "Unimal-v0":
            self.state_norm_index = np.concatenate([np.arange(9), np.array([30, 31, 41, 42])])
            self.state_min = np.array([-1.6, -25, -1, -15, -15, -15, -50, -50, -50, -0.3, -35, -0.3, -35])
            self.state_min = torch.Tensor(self.state_min.reshape(1, 1, -1)).cuda()
            self.state_max = np.array([1.6, 25, 4, 15, 15, 15, 50, 50, 50, 1.3, 35, 1.3, 35])
            self.state_max = torch.Tensor(self.state_max.reshape(1, 1, -1)).cuda()
        elif cfg.ENV_NAME == 'Modular-v0':
            self.state_norm_index = np.arange(12)
            if 'humanoid' in cfg.ENV.WALKER_DIR:
                self.state_min = np.array([-1, -0.4, 0, -10, -10, -10, -1, -40, -1, -np.pi, -np.pi, -np.pi])
                self.state_max = np.array([55, 0.4, 1.5, 10, 10, 10, 1, 40, 1, np.pi, np.pi, np.pi])
            elif 'walker' in cfg.ENV.WALKER_DIR:
                self.state_min = np.array([-1, -1, 0, -10, -10, -10, -1, -60, -1, -np.pi, -np.pi, -np.pi])
                self.state_max = np.array([50, 1, 1.6, 10, 10, 10, 1, 60, 1, np.pi, np.pi, np.pi])
            self.state_min = torch.Tensor(self.state_min.reshape(1, 1, -1)).cuda()
            self.state_max = torch.Tensor(self.state_max.reshape(1, 1, -1)).cuda()

        if cfg.MODEL.NORMALIZE_CONTEXT:
            with open('context_norm.pkl', 'rb') as f:
                self.context_min, self.context_max = pickle.load(f)
            self.context_min = torch.Tensor(self.context_min).float().unsqueeze(0).cuda()
            self.context_max = torch.Tensor(self.context_max).float().unsqueeze(0).cuda()
            self.context_range = self.context_max - self.context_min
            self.context_range[self.context_range == 0] = 1e-8

    def forward(self, obs, act=None, return_attention=False, unimal_ids=None, compute_val=True):
        
        # all_start = time.time()
        
        if act is not None:
            # batch_size = cfg.PPO.BATCH_SIZE
            batch_size = act.shape[0]
        else:
            # batch_size = cfg.PPO.NUM_ENVS
            batch_size = obs['proprioceptive'].shape[0]

        obs_env = {k: obs[k] for k in cfg.ENV.KEYS_TO_KEEP}
        if "obs_padding_cm_mask" in obs:
            obs_cm_mask = obs["obs_padding_cm_mask"]
        else:
            obs_cm_mask = None

        if 'context' in obs:
            obs_context = obs['context']
        else:
            obs_context = None

        if cfg.MODEL.NORMALIZE_CONTEXT:
            obs_context = obs_context.view(batch_size * self.seq_len, -1)
            obs_context = (obs_context - self.context_min) / self.context_range
            obs_context = obs_context.view(batch_size, -1)

        obs_dict = obs
        obs, obs_mask, act_mask = (
            obs["proprioceptive"],
            obs["obs_padding_mask"],
            obs["act_padding_mask"],
        )
        obs_mask = obs_mask.bool()
        act_mask = act_mask.bool()

        # start = time.time()
        morphology_info = {}
        if 'adjacency_matrix' in obs_dict:
            morphology_info["adjacency_matrix"] = obs_dict["adjacency_matrix"]
        if cfg.MODEL.TRANSFORMER.USE_CONNECTIVITY_IN_ATTENTION:
            morphology_info['connectivity'] = obs_dict['connectivity'].bool()
        if cfg.MODEL.TRANSFORMER.USE_MORPHOLOGY_INFO_IN_ATTENTION:
            morphology_info['connectivity'] = obs_dict['connectivity']
        if cfg.MODEL.TRANSFORMER.USE_SWAT_PE:
            # (batch_size, seq_len, traversal_num) ->(seq_len, batch_size, traversal_num)
            morphology_info['traversals'] = obs_dict['traversals'].permute(1, 0, 2).long()
        if cfg.MODEL.TRANSFORMER.USE_SWAT_RE:
            # (batch_size, seq_len, traversal_num) ->(seq_len, batch_size, traversal_num)
            morphology_info['SWAT_RE'] = obs_dict['SWAT_RE']
        if cfg.MODEL.TRANSFORMER.RNN_CONTEXT:
            morphology_info['node_path_length'] = obs_dict['node_path_length']
            morphology_info['node_path_mask'] = obs_dict['node_path_mask'].bool()
        # semantic for modular robots
        if cfg.MODEL.TRANSFORMER.USE_SEMANTIC_PE:
            morphology_info['position_id'] = obs_dict['position_id'].long()
        
        if len(morphology_info.keys()) == 0:
            morphology_info = None
        # end = time.time()
        # print ('time on connectivity', end - start)

        # reshape the obs for transformer input
        if type(self.mu_net) == TransformerModel:
            obs = obs.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
            if obs_context is not None:
                obs_context = obs_context.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
            if cfg.MODEL.BASE_CONTEXT_NORM == 'fixed':
                obs[:, :, self.context_index] = obs_context.clone()
            if cfg.MODEL.OBS_FIX_NORM:
                normed_obs = obs.clone()
                normed_obs[:, :, self.state_norm_index] = -1. + 2. * (obs[:, :, self.state_norm_index] - self.state_min) / (self.state_max - self.state_min)
                obs = normed_obs

        if cfg.MODEL.TYPE == 'gnn':
            obs = obs.reshape(batch_size, self.seq_len, -1)

        if compute_val and cfg.DISTILL.VALUE_NET:
            # Per limb critic values
            limb_vals, v_attention_maps = self.v_net(
                obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
                return_attention=return_attention,  
                unimal_ids=unimal_ids, 
            )
            # Zero out mask values
            limb_vals = limb_vals * (1 - obs_mask.int())
            # Use avg/max to keep the magnitidue same instead of sum
            num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
            val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)
        else:
            val, v_attention_maps = 0., None

        mu, mu_attention_maps = self.mu_net(
            obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
            return_attention=return_attention, 
            unimal_ids=unimal_ids, 
        )
        self.action_mu = mu
        if cfg.PPO.TANH == 'mean':
            mu = torch.tanh(mu)
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        # all_end = time.time()
        # print ('full forward time', all_end - all_start)

        if act is not None:
            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            self.limb_logp = logp
            logp = logp.sum(-1, keepdim=True)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.mean()
            return val, pi, logp, entropy
        else:
            if return_attention:
                return val, pi, v_attention_maps, mu_attention_maps
            else:
                return val, pi, None, None


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs, return_attention=False, unimal_ids=None, compute_val=True):
        val, pi, v_attention_maps, mu_attention_maps = self.ac(obs, return_attention=return_attention, unimal_ids=unimal_ids, compute_val=compute_val)
        self.pi = pi
        if not cfg.DETERMINISTIC:
            act = pi.sample()
        else:
            act = pi.loc
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        self.limb_logp = logp
        logp = logp.sum(-1, keepdim=True)
        self.v_attention_maps = v_attention_maps
        self.mu_attention_maps = mu_attention_maps
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs, unimal_ids=None):
        val, _, _, _ = self.ac(obs, unimal_ids=unimal_ids)
        return val
