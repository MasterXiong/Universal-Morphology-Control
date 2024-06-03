# python tools/train_single_task_transformer.py --start 0 --end 10
import os
import argparse
import json
import matplotlib.pyplot as plt
import pickle


def create_data(source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    agent_names = [x.split('.')[0] for x in os.listdir(f'{source_folder}/xml')]
    for agent in agent_names:
        os.makedirs(f'{target_folder}/{agent}', exist_ok=True)
        os.makedirs(f'{target_folder}/{agent}/xml', exist_ok=True)
        os.makedirs(f'{target_folder}/{agent}/metadata', exist_ok=True)
        os.system(f'cp {source_folder}/xml/{agent}.xml {target_folder}/{agent}/xml/')
        os.system(f'cp {source_folder}/metadata/{agent}.json {target_folder}/{agent}/metadata/')
        # os.mkdir('log_single_task/%s' %(agent))


def train(agent, output_folder, seed, task, extra_args=''):
    os.system(f'python tools/train_ppo.py --cfg ./configs/{task}.yaml --no_context_in_state OUT_DIR output/{output_folder}/{agent}/{seed} \
        ENV.WALKER_DIR data_single_robot/{agent} PPO.MAX_STATE_ACTION_PAIRS 10000000.0 RNG_SEED {seed} \
        MODEL.TRANSFORMER.EMBEDDING_DROPOUT False \
        {extra_args}')



if __name__ == '__main__':
    
    # vanilla MLP
    # python tools/train_single_robot.py --model_type linear --task ft --output_folder MLP_ST_ft_256*2_KL_5_wo_context --kl 5. --seed 1409 --no_sim_reset --start 0 --end 1
    # python tools/train_single_task_transformer.py --model_type linear --task csr --output_folder MLP_ST_csr_256*2_KL_3 --kl 3. --seed 1410 --start 0 --end 50
    # Transformer
    # python tools/train_single_task_transformer.py --model_type transformer --task ft --output_folder TF_ST_ft_KL_5_wo_dropout --seed 1409 --start 0 --end 10
    # python tools/train_single_task_transformer.py --model_type transformer --task csr --output_folder TF_ST_csr_KL_3_wo_PE+dropout --no_PE --seed 1409 --start 0 --end 10
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=100, type=int)
    parser.add_argument("--seed", default=1409, type=int)
    parser.add_argument("--output_folder", default='log_test', type=str)
    parser.add_argument("--task", default='ft', type=str)
    parser.add_argument("--model_type", default='transformer', type=str)
    parser.add_argument("--no_PE", action="store_true")
    parser.add_argument("--kl", default=5., type=float)
    parser.add_argument("--tf_num_layer", default=None, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--std", default='fixed', type=str)
    parser.add_argument("--save_limb_ratio", action="store_true")
    parser.add_argument("--fix_obs_rms", type=str)
    parser.add_argument("--no_sim_reset", action="store_true")
    args = parser.parse_args()
    
    folder = 'data/train_mutate_1000'
    # create_data('data/train_mutate_1000', 'data_single_robot')
    if os.path.exists(f'{folder}/agents.pkl'):
        with open(f'{folder}/agents.pkl', 'rb') as f:
            agent_names = pickle.load(f)
    else:
        agent_names = [x.split('.')[0] for x in os.listdir(f'{folder}/xml')]
        with open(f'{folder}/agents.pkl', 'wb') as f:
            pickle.dump(agent_names, f)

    agents = agent_names[args.start:args.end]
    print (agents)

    # extra_args = 'MODEL.TRANSFORMER.POS_EMBEDDING None'
    # extra_args = []
    # if 'embed' in args.output_folder:
    #     extra_args.append('MODEL.TRANSFORMER.PER_NODE_EMBED True')
    # if 'decoder' in args.output_folder:
    #     extra_args.append('MODEL.TRANSFORMER.PER_NODE_DECODER True')
    # # if 'wo_PE+dropout' in args.output_folder:
    # #     extra_args.append('MODEL.TRANSFORMER.EMBEDDING_DROPOUT False')
    # extra_args.append('MODEL.TRANSFORMER.EMBEDDING_DROPOUT False')
    # extra_args.append('MODEL.TRANSFORMER.POS_EMBEDDING None')
    # extra_args.append('PPO.KL_TARGET_COEF 5.')
    # extra_args = ' '.join(extra_args)
    # print (extra_args)

    extra_args = []
    params = args.output_folder.split('_')
    # params for MLP
    for param in params:
        if '*' in param:
            hidden_dim = eval(param.split('*')[0])
            layer_num = eval(param.split('*')[1])
            extra_args.append(f'MODEL.MLP.HIDDEN_DIM {hidden_dim}')
            extra_args.append(f'MODEL.MLP.LAYER_NUM {layer_num}')
            break
    # change learning rate scheme
    for param in params:
        if param == 'constant':
            extra_args.append('PPO.LR_POLICY constant')
            break
    extra_args.append(f'PPO.BASE_LR {args.lr}')
    if args.std == 'learn':
        extra_args.append('MODEL.ACTION_STD_FIXED False')
    if args.save_limb_ratio:
        extra_args.append('SAVE_LIMB_RATIO True')
    # if args.task in ['ft', 'incline']:
    #     extra_args.append('PPO.KL_TARGET_COEF 5.')
    # else:
    #     extra_args.append('PPO.KL_TARGET_COEF 3.')
    extra_args.append(f'PPO.KL_TARGET_COEF {args.kl}')
    if args.model_type == 'linear':
        extra_args.append('MODEL.TYPE vanilla_mlp')
    if args.no_PE:
        extra_args.append('MODEL.TRANSFORMER.POS_EMBEDDING None')
    if args.no_sim_reset:
        extra_args.append('ENV.NEW_SIM_ON_RESET False')
    if args.tf_num_layer is not None:
        extra_args.append(f'MODEL.TRANSFORMER.NLAYERS {args.tf_num_layer}')
    if args.fix_obs_rms is not None:
        extra_args.append(f'ENV.FIX_OBS_NORM {args.fix_obs_rms}')
    extra_args = ' '.join(extra_args)
    print (extra_args)

    for agent in agents:
        if os.path.exists(f'output/{args.output_folder}/{agent}/{args.seed}/checkpoint_-1.pt'):
            print (f'already finish {agent} {args.seed}')
            continue
        train(agent, args.output_folder, args.seed, args.task, extra_args)
        # os.system(f'cp unimals_100/train/xml/{agent}.xml unimals_20/xml/')
        # os.system(f'cp unimals_100/train/metadata/{agent}.json unimals_20/metadata/')
