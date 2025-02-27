"""Configuration file (powered by YACS)."""

import copy
import os

from metamorph.yacs import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# ----------------------------------------------------------------------------#
# XML template params
# ----------------------------------------------------------------------------#
# Refer mujoco docs for what each param does

_C.XML = CN()

_C.XML.NJMAX = 1000

_C.XML.NCONMAX = 200

_C.XML.GEOM_CONDIM = 3

_C.XML.GEOM_FRICTION = [0.7, 0.1, 0.1]

_C.XML.FILTER_PARENT = "enable"

_C.XML.SHADOWCLIP = 0.5

# ----------------------------------------------------------------------------#
# Unimal Env Options
# ----------------------------------------------------------------------------#
_C.ENV = CN()

_C.ENV.FORWARD_REWARD_WEIGHT = 1.0

_C.ENV.AVOID_REWARD_WEIGHT = 100.0

_C.ENV.CTRL_COST_WEIGHT = 0.0

_C.ENV.HEALTHY_REWARD = 0.0

_C.ENV.STAND_REWARD_WEIGHT = 0.0

_C.ENV.STATE_RANGE = (-100.0, 100.0)

_C.ENV.Z_RANGE = (-0.1, float("inf"))

_C.ENV.ANGLE_RANGE = (-0.2, 0.2)

_C.ENV.RESET_NOISE_SCALE = 5e-3

# Healthy reward is 1 if head_pos >= STAND_HEIGHT_RATIO * head_pos in
# the xml i.e the original height of the unimal.
_C.ENV.STAND_HEIGHT_RATIO = 0.5

# List of modules to add to the env. Modules will be added in the same order
_C.ENV.MODULES = ["Floor", "Agent"]

# Agent name if you are not using unimal but want to still use the unimal env
_C.ENV.WALKER_DIR = "./unimals_100/train"

# Agent name if you are not using unimal but want to still use the unimal env
_C.ENV.WALKERS = []

# Keys to keep in SelectKeysWrapper
_C.ENV.KEYS_TO_KEEP = []

# Skip position of free joint (or x root joint) in position observation for
# translation invariance
_C.ENV.SKIP_SELF_POS = False

# Specify task. Can be locomotion, manipulation
_C.ENV.TASK = "locomotion"

# Optional wrappers to add to task. Most wrappers for a task will eventually be
# hardcoded in make_env_task func. Put wrappers which you want to experiment
# with.
_C.ENV.WRAPPERS = []

# Task sampling strategy for multi-task envs. See multi_env_wrapper.py
# support: 
#   - uniform_random_strategy (uniform sampling)
#   - balanced_replay_buffer (the method proposed in MetaMorph)
#   - UED (minimax regret sampling)
_C.ENV.TASK_SAMPLING = "balanced_replay_buffer"

# For envs which change on each reset e.g. vt and mvt this should be true.
# For floor env this should be false, leads to 3x speed up.
_C.ENV.NEW_SIM_ON_RESET = True

# whether to use a fixed env for each process
_C.ENV.FIX_ENV  = False

# randomly change hardware parameters
_C.ENV.CHANGE_JOINT_ANGLE = False

# filter walkers
_C.ENV.FILTER_WALKERS = False
_C.ENV.FILTER_SCORE_FOLDER = ''
_C.ENV.FILTER_THRESHOLD = 0.

# fix obs norm
_C.ENV.FIX_OBS_NORM = None

# ----------------------------------------------------------------------------#
# Terrain Options
# ----------------------------------------------------------------------------#
# Attributes for x will be called length, y width and z height
_C.TERRAIN = CN()

# Size of the "floor/0" x, y, z
_C.TERRAIN.SIZE = [25, 20, 1]

_C.TERRAIN.START_FLAT = 2

_C.TERRAIN.CENTER_FLAT = 2

# Supported types of terrain obstacles
_C.TERRAIN.TYPES = ["gap", "jump"]

# Length of flat terrain
_C.TERRAIN.FLAT_LENGTH_RANGE = [9, 15, 2]

# Shared across avoid and jump
_C.TERRAIN.WALL_LENGTH = 0.1

# Length of terrain on which there will be hfield
_C.TERRAIN.HFIELD_LENGTH_RANGE = [4, 8, 4]

# Max height in case of slope profile
_C.TERRAIN.CURVE_HEIGHT_RANGE = [0.6, 1.2, 0.1]

_C.TERRAIN.BOUNDARY_WALLS = True

# Height of individual step
_C.TERRAIN.STEP_HEIGHT = 0.2

# Length of terrain on which there will be steps
_C.TERRAIN.STEP_LENGTH_RANGE = [12, 16, 4]

_C.TERRAIN.NUM_STEPS = 8

_C.TERRAIN.RUGGED_SQUARE_CLIP_RANGE = [0.2, 0.3, 0.1]

# Max height of bumps in bowl
_C.TERRAIN.BOWL_MAX_Z = 1.3

# Angle of incline for incline task
_C.TERRAIN.INCLINE_ANGLE = 0

# Vertical distance between the bottom most point of unimal and floor
_C.TERRAIN.FLOOR_OFFSET = 0.2
# ----------------------------------------------------------------------------#
# Objects Options
# ----------------------------------------------------------------------------#
# Attributes for x will be called length, y width and z height
_C.OBJECT = CN()

# Goal position, if empty each episode will have a different goal position. Or
# you can specify the position here. Only specify the x, y position.
_C.OBJECT.GOAL_POS = []

# Same as GOAL_POS
_C.OBJECT.BOX_POS = []

# Min distance from the walls to place the object
_C.OBJECT.PLACEMENT_BUFFER_LEN = 3

_C.OBJECT.PLACEMENT_BUFFER_WIDTH = 0

# Half len of square for close placement
_C.OBJECT.CLOSE_PLACEMENT_DIST = 10

# Min distance between agent and goal for success
_C.OBJECT.SUCCESS_MARGIN = 0.5

# Side len of the box
_C.OBJECT.BOX_SIDE = 0.5

# Number of obstacles for obstacle env
_C.OBJECT.NUM_OBSTACLES = 50

# Length of the obstacle box
_C.OBJECT.OBSTACLE_LEN_RANGE = [0.5, 1, 0.1]

# Width of the obstacle box
_C.OBJECT.OBSTACLE_WIDTH_RANGE = [0.5, 1, 0.1]

# Range of distance between successive object placements for forward_placement
_C.OBJECT.FORWARD_PLACEMENT_DIST = [10, 15]

# Typpe of object to manipulate can be box or ball
_C.OBJECT.TYPE = "box"

_C.OBJECT.BALL_RADIUS = 0.15

_C.OBJECT.BOX_MASS = 1.0

# ----------------------------------------------------------------------------#
# Hfield Options
# ----------------------------------------------------------------------------#
_C.HFIELD = CN()

# For planer walker type unimals 1 otherwise 2
_C.HFIELD.DIM = 2

# Slice of hfield given to agent as obs. [behind, front, right, left] or
# [-x, +x, -y, +y]
_C.HFIELD.OBS_SIZE = [1, 4, 4, 4]

_C.HFIELD.ADAPTIVE_OBS = False

# See _cal_hfield_bounds in hfield.py
_C.HFIELD.ADAPTIVE_OBS_SIZE = [0.10, 0.50, 1.5, 5.0]

# Pad hfiled for handling agents on edges of the terrain. Padding value should
# be greater than sqrt(2) * max(HFIELD.OBS_SIZE). As when you rotate the
# the hfield obs, square diagonal should fit inside padding.
_C.HFIELD.PADDING = 10

# Number representing that the terrain has gap in hfield obs
_C.HFIELD.GAP_DEPTH = -10

# Number of divisions in 1 unit for hfield, should be a multiple of 10
_C.HFIELD.NUM_DIVS = 10

# Viz the extreme points of hfield
_C.HFIELD.VIZ = False

# ----------------------------------------------------------------------------#
# Video Options
# ----------------------------------------------------------------------------#
_C.VIDEO = CN()

# Save video
_C.VIDEO.SAVE = False

# Frame width
_C.VIDEO.WIDTH = 640

# Frame height
_C.VIDEO.HEIGHT = 360

# FPS for saving
_C.VIDEO.FPS = 30

# --------------------------------------------------------------------------- #
# PPO Options
# --------------------------------------------------------------------------- #
_C.PPO = CN()

# Discount factor for rewards
_C.PPO.GAMMA = 0.99

# GAE lambda parameter
_C.PPO.GAE_LAMBDA = 0.95

# Hyperparameter which roughly says how far away the new policy is allowed to
# go from the old
_C.PPO.CLIP_EPS = 0.2

# Number of epochs (K in PPO paper) of sgd on rollouts in buffer
_C.PPO.EPOCHS = 8

# Batch size for sgd (M in PPO paper)
_C.PPO.BATCH_SIZE = 5120

# Value (critic) loss term coefficient
_C.PPO.VALUE_COEF = 0.5

# If KL divergence between old and new policy exceeds KL_TARGET_COEF * 0.01
# stop updates. Default value is high so that it's not used by default.
_C.PPO.KL_TARGET_COEF = 20.0

# Clip value function
_C.PPO.USE_CLIP_VALUE_FUNC = True

# Entropy term coefficient
_C.PPO.ENTROPY_COEF = 0.0

# Max timesteps per rollout
_C.PPO.TIMESTEPS = 2560

# Number of parallel envs for collecting rollouts
_C.PPO.NUM_ENVS = 32

# Learning rate ranges from BASE_LR to MIN_LR*BASE_LR according to the LR_POLICY
_C.PPO.BASE_LR = 3e-4
_C.PPO.MIN_LR = 0.0

# Learning rate policy select from {'cos', 'lin', 'constant', 'adaptive'}
_C.PPO.LR_POLICY = "cos"
# when to anneal lr if using adaptive lr policy
_C.PPO.LR_ANNEAL_THRESHOLD = 32

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.PPO.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of iters
_C.PPO.WARMUP_ITERS = 5

# EPS for Adam/RMSProp
_C.PPO.EPS = 1e-5

_C.PPO.WEIGHT_DECAY = 0.

# Value to clip the gradient via clip_grad_norm_
_C.PPO.MAX_GRAD_NORM = 0.5

# Total number of env.step() across all processes and all rollouts over the
# course of training
_C.PPO.MAX_STATE_ACTION_PAIRS = 1e8

# Iter here refers to 1 cycle of experience collection and policy update.
# Refer PPO paper. This is field is inferred see: calculate_max_iters()
_C.PPO.MAX_ITERS = -1

# Length of video to save while evaluating policy in num env steps. Env steps
# may not be equal to actual simulator steps. Actual simulator steps would be
# env_steps * frame_skip.
_C.PPO.VIDEO_LENGTH = 1000

# Path to load model from
_C.PPO.CHECKPOINT_PATH = ""
# whether we load a checkpoint model for continuing training or fine-tuning
_C.PPO.CONTINUE_TRAINING = False

_C.PPO.EARLY_EXIT = False

_C.PPO.EARLY_EXIT_STATE_ACTION_PAIRS = 1e8

_C.PPO.EARLY_EXIT_MAX_ITERS = -1

# my abs ratio idea
_C.PPO.ABS_CLIP = False
_C.PPO.ABS_CLIP_THRESHOLD = 0.2

_C.PPO.TANH = None

# optimizer
_C.PPO.OPTIMIZER = 'adam'

# --------------------------------------------------------------------------- #
# Task sampling options
# --------------------------------------------------------------------------- #

_C.TASK_SAMPLING = CN()

_C.TASK_SAMPLING.EMA_ALPHA = 0.1

_C.TASK_SAMPLING.PROB_ALPHA = 1.0

_C.TASK_SAMPLING.AVG_TYPE = "ema"

# --------------------------------------------------------------------------- #
# Model Options
# --------------------------------------------------------------------------- #
_C.MODEL = CN()

# Type of actor critic model: ActorCritic
_C.MODEL.ACTOR_CRITIC = "ActorCritic"

_C.MODEL.LIMB_EMBED_SIZE = 128

_C.MODEL.JOINT_EMBED_SIZE = 128

# Max number of joints across all the envs
_C.MODEL.MAX_JOINTS = 7

# Max number of limbs across all the envs
_C.MODEL.MAX_LIMBS = 8

# Fixed std value
_C.MODEL.ACTION_STD = 0.9

# Use fixed or learnable std
_C.MODEL.ACTION_STD_FIXED = True

# Types of proprioceptive obs to include
_C.MODEL.PROPRIOCEPTIVE_OBS_TYPES = [
    "body_xpos", "body_xvelp", "body_xvelr", "body_xquat", 
    "body_pos", "body_ipos", "body_iquat", "geom_quat", # limb model
    "body_mass", "body_shape", # limb hardware
    "qpos", "qvel", 
    "jnt_pos", # joint model
    "joint_range", "joint_axis", "gear" # joint hardware
]

# _C.MODEL.CONTEXT_OBS_TYPES = [
#     "body_pos", "body_ipos", "body_iquat", "geom_quat", # limb model
#     "body_mass", "body_shape", # limb hardware
#     "jnt_pos", # joint model
#     "joint_range", "joint_axis", "gear" # joint hardware
# ]
_C.MODEL.CONTEXT_OBS_TYPES = [
    "absolute_body_pos", "absolute_body_ipos", "body_iquat", # limb model
    "body_mass", "body_shape", # limb hardware
    "joint_range_onehot", "joint_axis", "gear_onehot",  # joint hardware
    # "jnt_pos", # joint model
    # "joint_range", "gear", # joint hardware
    # "torso_limb_indicator", 
    # "body_pos", "body_ipos", 
]

# Model specific observation types to keep
_C.MODEL.OBS_TYPES = [
    "proprioceptive", "edges", "obs_padding_mask", "act_padding_mask", 
    "context", 
    "adjacency_matrix", 
    # "connectivity", 
    # "node_depth", 
    # "traversals", 
    # 'node_path_length', 
    # 'node_path_mask', 
    # 'SWAT_RE', 
]

# Observations to normalize via VecNormalize
_C.MODEL.OBS_TO_NORM = ["proprioceptive"]

# Context normalization with RunningMeanStd or fixed range
_C.MODEL.BASE_CONTEXT_NORM = 'running'
_C.MODEL.NORMALIZE_CONTEXT = False

# normalize state inputs with fixed range
_C.MODEL.OBS_FIX_NORM = False

# normalize over all limbs
_C.MODEL.NORM_OVER_LIMB = False
_C.MODEL.INCLUDE_PADDING_LIMB_IN_NORM = True

# Wrappers to add specific to model
_C.MODEL.WRAPPERS = ["MultiUnimalNodeCentricObservation", "MultiUnimalNodeCentricAction"]

# --------------------------------------------------------------------------- #
# Transformer Options
# --------------------------------------------------------------------------- #
# model type
_C.MODEL.TYPE = 'transformer'
# hyperparameters for MLP model
_C.MODEL.MLP = CN()
_C.MODEL.MLP.HIDDEN_DIM = 256
_C.MODEL.MLP.LAYER_NUM = 3
# dropout in the base MLP
_C.MODEL.MLP.DROPOUT = None

# configs for GNN model
_C.MODEL.GNN = CN()
_C.MODEL.GNN.LAYER_NUM = 3
_C.MODEL.GNN.DECODER_DIMS = []

# hyperparameters for hypernet
# configs for context encoder
_C.MODEL.HYPERNET = CN()
_C.MODEL.HYPERNET.CONTEXT_ENCODER_TYPE = 'transformer'
_C.MODEL.HYPERNET.CONTEXT_EMBED_SIZE = 128
_C.MODEL.HYPERNET.CONTEXT_MASK = True
_C.MODEL.HYPERNET.CONTEXT_TF_ENCODER_NHEAD = 2
_C.MODEL.HYPERNET.CONTEXT_TF_ENCODER_FF_DIM = 256
_C.MODEL.HYPERNET.ENCODER_LAYER_NUM = 3
_C.MODEL.HYPERNET.EMBEDDING_DROPOUT = None
# configs for the output layer of HN
_C.MODEL.HYPERNET.HN_INIT_STRATEGY = 'bias_init'
# configs for general HN setup
_C.MODEL.HYPERNET.SHARE_CONTEXT_ENCODER = False
# how to aggregate limb embedding into the first hidden layer
# choices: sum (default), limb_num, sqrt_limb_num, max_limb_num
_C.MODEL.HYPERNET.INPUT_AGGREGATION = 'sum'

# hyperparameters for transformers
_C.MODEL.TRANSFORMER = CN()

# Number of attention heads in TransformerEncoderLayer (nhead)
_C.MODEL.TRANSFORMER.NHEAD = 2

# TransformerEncoderLayer (dim_feedforward)
_C.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 1024

# TransformerEncoderLayer (dropout)
_C.MODEL.TRANSFORMER.DROPOUT = 0.0

# Number of TransformerEncoderLayer in TransformerEncoder
_C.MODEL.TRANSFORMER.NLAYERS = 5

# Init for input embedding
_C.MODEL.TRANSFORMER.EMBED_INIT = 0.1

# Init for output decoder embodedding
_C.MODEL.TRANSFORMER.DECODER_INIT = 0.01

# init range for HN embedding
_C.MODEL.TRANSFORMER.HN_EMBED_INIT = 0.1

_C.MODEL.TRANSFORMER.DECODER_DIMS = []

_C.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS = []

# Early vs late fusion of exterioceptive observation
_C.MODEL.TRANSFORMER.EXT_MIX = "none"

# Type of position embedding to use: None, learnt
_C.MODEL.TRANSFORMER.POS_EMBEDDING = "learnt"

# dropout setting
_C.MODEL.TRANSFORMER.EMBEDDING_DROPOUT = True
_C.MODEL.TRANSFORMER.CONSISTENT_DROPOUT = False

# whether to scale the embedding by sqrt(embedding_dim) before feeding into transformer
_C.MODEL.TRANSFORMER.EMBEDDING_SCALE = True

# Whether to use hypernet to generate the weights of the decoder
_C.MODEL.TRANSFORMER.HYPERNET = False
_C.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE = 128
_C.MODEL.TRANSFORMER.HN_EMBED = True
_C.MODEL.TRANSFORMER.HN_DECODER = True
_C.MODEL.TRANSFORMER.HN_CONTEXT_ENCODER = 'linear'
_C.MODEL.TRANSFORMER.HN_CONTEXT_LAYER_NUM = 1

# whether to use context to generate fixed attention weights
_C.MODEL.TRANSFORMER.FIX_ATTENTION = False
_C.MODEL.TRANSFORMER.CONTEXT_LAYER = 3
_C.MODEL.TRANSFORMER.LINEAR_CONTEXT_LAYER = 2
_C.MODEL.TRANSFORMER.CONTEXT_ENCODER = 'linear'
# use hfield in fix attention
_C.MODEL.TRANSFORMER.HFIELD_IN_FIX_ATTENTION = False
# use node context / state features as FA input
_C.MODEL.TRANSFORMER.CONTEXT_AS_FA_INPUT = True

# whether adjust attention weights based on morphology information
_C.MODEL.TRANSFORMER.USE_MORPHOLOGY_INFO_IN_ATTENTION = False
# context PE
_C.MODEL.TRANSFORMER.CONTEXT_PE = False
# use SWAT PE
_C.MODEL.TRANSFORMER.USE_SWAT_PE = False
_C.MODEL.TRANSFORMER.TRAVERSALS = ['pre', 'inlcrs', 'postlcrs']
_C.MODEL.TRANSFORMER.USE_SWAT_RE = False
# use separate PE
_C.MODEL.TRANSFORMER.USE_SEPARATE_PE = False
_C.MODEL.TRANSFORMER.SEPARATE_PE_UPDATE_ITER = 0
# tree PE
_C.MODEL.TRANSFORMER.TREE_PE_IN_CONTEXT = False
_C.MODEL.TRANSFORMER.MAX_CHILD_NUM = 4
# graph PE
_C.MODEL.TRANSFORMER.GRAPH_PE_IN_CONTEXT = False
_C.MODEL.TRANSFORMER.GRAPH_PE_DIM = 3
# node depth PE
_C.MODEL.TRANSFORMER.NODE_DEPTH_IN_CONTEXT = False
_C.MODEL.TRANSFORMER.MAX_NODE_DEPTH = 20
# children number PE
_C.MODEL.TRANSFORMER.CHILD_NUM_IN_CONTEXT = False
# RNN PE
_C.MODEL.TRANSFORMER.RNN_CONTEXT = False
# add connectivity to TF attention mask
_C.MODEL.TRANSFORMER.USE_CONNECTIVITY_IN_ATTENTION = False
# test per-node embed and decode
_C.MODEL.TRANSFORMER.PER_NODE_EMBED = False
_C.MODEL.TRANSFORMER.PER_NODE_DECODER = False
# semantic PE
_C.MODEL.TRANSFORMER.USE_SEMANTIC_PE = False

# --------------------------------------------------------------------------- #
# Finetuning Options
# --------------------------------------------------------------------------- #
_C.MODEL.FINETUNE = CN()

# If true fine tune all the model params, if false fine tune only specific layer
_C.MODEL.FINETUNE.FULL_MODEL = True

# Name of layers to fine tune
_C.MODEL.FINETUNE.LAYER_SUBSTRING = []

# --------------------------------------------------------------------------- #
# Sampler (VecEnv) Options
# --------------------------------------------------------------------------- #
_C.VECENV = CN()

# Type of vecenv. DummyVecEnv is generally the fastest option for light weight
# envs. The fatest configuration is most likely DummyVecEnv coupled with DDP.
# Note: It is faster to have N dummyVecEnvs in DDP than having the same config
# via SubprocVecEnv.
_C.VECENV.TYPE = "SubprocVecEnv"

# Number of envs to run in series for SubprocVecEnv
_C.VECENV.IN_SERIES = 2

# --------------------------------------------------------------------------- #
# CUDNN options
# --------------------------------------------------------------------------- #
_C.CUDNN = CN()

_C.CUDNN.BENCHMARK = False
_C.CUDNN.DETERMINISTIC = True

# UED options
_C.UED = CN()

# parameters of how to generate new agents
_C.UED.GENERATION = False
_C.UED.GENERATION_FREQ = 10
_C.UED.GENERATION_NUM = 10
# how to select mutation parent: 'learning_progress', 'uniform'
_C.UED.PARENT_SELECT_STRATEGY = None
# whether to balance the generation process
_C.UED.BALANCE_GENERATION = False
# only grow limbs
_C.UED.GROW_LIMB_ONLY = False
# only mutate agents with high enough scores
_C.UED.MUTATE_THRESHOLD = None

# UED method: 'regret', 'uniform', 'positive_value_loss', 'L1_value_loss', 'GAE', 'learning_progress'
_C.UED.CURATION = 'uniform'
_C.UED.STALENESS_WEIGHT = 0.1 # also used for validation
_C.UED.SCORE_EMA_COEF = 0.5

_C.UED.UPPER_BOUND_PATH = None
_C.UED.REGRET_TYPE = 'absolute' # absolute or relative

_C.UED.PROB_CHANGE_RATE = None

# expand by validation set
_C.UED.USE_VALIDATION = False
_C.UED.CHECK_TRAIN_FREQ = 10
_C.UED.TRAIN_MAX_WAIT = 10
_C.UED.CHECK_VALID_FREQ = 10
_C.UED.VALID_MAX_WAIT = 5
_C.UED.VALID_TIMESTEPS = 10000
_C.UED.RANDOM_VALIDATION_ROBOT_NUM = 10
_C.UED.VALIDATION_START_ITER = 30

_C.UED.EPISODE_NUM_DISCOUNT = 0.9

# parameters for learning dynamics model
_C.DYNAMICS = CN()

_C.DYNAMICS.BATCH_SIZE = 25600
_C.DYNAMICS.EPOCH_NUM = 100
_C.DYNAMICS.BASE_LR = 1e-4
_C.DYNAMICS.EPS = 1e-5
_C.DYNAMICS.WEIGHT_DECAY = 0.

_C.DYNAMICS.MODEL_STEP = False
_C.DYNAMICS.MODEL_PATH = ''

# parameters for policy distillation
_C.DISTILL = CN()

_C.DISTILL.PER_AGENT_SAMPLE_NUM = 8000
_C.DISTILL.BATCH_SIZE = 5120
_C.DISTILL.EPOCH_NUM = 100
_C.DISTILL.BASE_LR = 3e-4
_C.DISTILL.EPS = 1e-5
_C.DISTILL.WEIGHT_DECAY = 0.
_C.DISTILL.SOURCE = ''
_C.DISTILL.TARGET = ''
_C.DISTILL.SAVE_FREQ = 10
# whether use sampled action 'act' or action mean 'act_mean' as distillation target
_C.DISTILL.IMITATION_TARGET = 'act'
_C.DISTILL.VALUE_NET = True
_C.DISTILL.BALANCED_LOSS = False
_C.DISTILL.GRAD_NORM = None
_C.DISTILL.SAMPLE_STRATEGY = 'env_first'
_C.DISTILL.LOSS_TYPE = 'logp'
_C.DISTILL.KL_TARGET = 'act_mean'
_C.DISTILL.OPTIMIZER = 'adam'
_C.DISTILL.SAMPLE_WEIGHT = False
_C.DISTILL.LARGE_ACT_DECAY = 1.
_C.DISTILL.CONCAT_CONTEXT_TO_OBS = False

# configs for DAgger
_C.DAGGER = CN()
_C.DAGGER.STUDENT_PATH = ''
_C.DAGGER.TEACHER_PATH = ''
_C.DAGGER.STUDENT_CHECKPOINT = 5
_C.DAGGER.ITERS = 10
_C.DAGGER.ITER_STEPS = 100
_C.DAGGER.BASE_LR = 3e-4
_C.DAGGER.EPS = 1e-5
_C.DAGGER.WEIGHT_DECAY = 0.
_C.DAGGER.SAVE_FREQ = 5
_C.DAGGER.BATCH_SIZE = 5120
_C.DAGGER.EPOCH_PER_ITER = 10
_C.DAGGER.MINIBATCH_UPDATE_PER_ITER = 100
_C.DAGGER.ANNEAL_THRESHOLD = 1
_C.DAGGER.SUFFIX = ''
_C.DAGGER.TEST_DATA_PATH = ''

# ----------------------------------------------------------------------------#
# Misc Options
# ----------------------------------------------------------------------------#
# Output directory
_C.OUT_DIR = "./output"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries. This is the only seed
# which will effect env variations.
_C.RNG_SEED = 1

# Name of the environment used for experience collection
_C.ENV_NAME = "Unimal-v0"

# Use GPU
_C.DEVICE = "cuda:0"

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters.
_C.LOG_PERIOD = 10

# Checkpoint period in iters. Refer LOG_PERIOD for meaning of iter
_C.CHECKPOINT_PERIOD = 100

# Evaluate the policy after every EVAL_PERIOD iters
_C.EVAL_PERIOD = -1
# whether to include terminating on falling mask during evaluation
_C.TERMINATE_ON_FALL = True
# whether to use deterministic action (only used during evaluation)
_C.DETERMINISTIC = False

# Node ID for distributed runs
_C.NODE_ID = -1

# Number of nodes
_C.NUM_NODES = 1

# Unimal template path relative to the basedir
_C.UNIMAL_TEMPLATE = "./metamorph/envs/assets/unimal.xml"

# Save histogram weights
_C.SAVE_HIST_WEIGHTS = False
_C.SAVE_HIST_RATIO = False
_C.PER_LIMB_GRAD = False
_C.SAVE_LIMB_RATIO = False

# Optional description for exp
_C.DESC = ""

# How to handle mjstep exception
_C.EXIT_ON_MJ_STEP_EXCEPTION = False

_C.MIRROR_DATA_AUG = False

def dump_cfg(cfg_name=None):
    """Dumps the config to the output directory."""
    if not cfg_name:
        cfg_name = _C.CFG_DEST
    cfg_file = os.path.join(_C.OUT_DIR, cfg_name)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def get_default_cfg():
    return copy.deepcopy(cfg)
