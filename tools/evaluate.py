import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.algos.ppo.model import Agent
from metamorph.algos.network.hypernet import HNMLP
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch

from tools.train_ppo import set_cfg_options

torch.manual_seed(0)


# evaluate on a single robot
def evaluate(policy, env, agent, compute_gae=False):

    episode_return = np.zeros(cfg.PPO.NUM_ENVS)
    not_done = np.ones(cfg.PPO.NUM_ENVS)
    episode_values, episode_rewards = [], []
    episode_len = np.zeros(cfg.PPO.NUM_ENVS, dtype=int)
    timeout = np.zeros(cfg.PPO.NUM_ENVS, dtype=int)

    obs = env.reset()
    ood_ratio = (obs['proprioceptive'].abs() == 10.).float().mean().item()
    if type(policy.ac.mu_net) == HNMLP:
        morphology_info = {}
        morphology_info['adjacency_matrix'] = obs['adjacency_matrix']
        with torch.no_grad():
            policy.ac.mu_net.generate_params(obs['context'], obs['obs_padding_mask'].bool(), morphology_info=morphology_info)

    unimal_ids = env.get_unimal_idx()
    for t in range(2000):
        if compute_gae:
            val, act, _ = policy.act(obs, return_attention=False, compute_val=True, unimal_ids=unimal_ids)
            episode_values.append(val)
        else:
            _, act, _ = policy.act(obs, return_attention=False, compute_val=False, unimal_ids=unimal_ids)

        if cfg.PPO.TANH == 'action':
            obs, reward, done, infos = env.step(torch.tanh(act))
        else:
            obs, reward, done, infos = env.step(act)
        if compute_gae:
            episode_rewards.append(reward)
        ood_ratio += (obs['proprioceptive'].abs() == 10.).float().mean().item()

        idx = np.where(done)[0]
        for i in idx:
            if not_done[i] == 1:
                not_done[i] = 0
                episode_return[i] = infos[i]['episode']['r']
                episode_len[i] = t + 1
                timeout[i] = 'timeout' in infos[i]
        if not_done.sum() == 0:
            break

    ood_ratio /= (t + 1)

    episode_gae = []
    if compute_gae:
        episode_values = torch.stack(episode_values).cpu().numpy()
        episode_rewards = torch.stack(episode_rewards).cpu().numpy()
        for i in range(cfg.PPO.NUM_ENVS):
            episode_value = episode_values[:episode_len[i], i]
            episode_reward = episode_rewards[:episode_len[i], i]
            episode_gae.append(compute_GAE(episode_value, episode_reward, timeout=timeout[i]))

    return episode_return, ood_ratio, episode_gae


def evaluate_model(model_path, agent_path, policy_folder, suffix=None, terminate_on_fall=True, deterministic=False, compute_gae=False):

    test_agents = [x.split('.')[0] for x in os.listdir(f'{agent_path}/xml')]

    print (policy_folder)
    cfg.merge_from_file(f'{policy_folder}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    cfg.TERMINATE_ON_FALL = terminate_on_fall
    cfg.DETERMINISTIC = deterministic
    set_cfg_options()
    cfg.PPO.NUM_ENVS = 16
    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    # change to eval mode as we have dropout in the model
    policy.ac.eval()

    ood_list = np.zeros(len(test_agents))
    avg_score = []
    if len(policy_folder.split('/')) == 3:
        folder_name = policy_folder.split('/')[1]
    else:
        folder_name = policy_folder.split('/')[0]
    output_name = folder_name + '/' + suffix
    os.makedirs(f'eval/{folder_name}', exist_ok=True)
    print (output_name)

    if os.path.exists(f'eval/{output_name}.pkl'):
        with open(f'eval/{output_name}.pkl', 'rb') as f:
            eval_result = pickle.load(f)
    else:
        eval_result = {}

    obs_rms = get_ob_rms(ppo_trainer.envs)
    try:
        obs_rms['proprioceptive'].mean = obs_rms['proprioceptive'].mean.numpy()
        obs_rms['proprioceptive'].var = obs_rms['proprioceptive'].var.numpy()
    except:
        pass

    all_obs = dict()
    for i, agent in enumerate(test_agents):
        # with open(f'{agent_path}/metadata/{agent}.json', 'r') as f:
        #     limb_num = json.load(f)["num_limbs"]
        if agent in eval_result and len(eval_result[agent]) == 100:
            episode_return, ood_ratio, _ = eval_result[agent]
            avg_score.append(np.array(episode_return).mean())
        else:
            envs = make_vec_envs(xml_file=agent, training=False, norm_rew=False, render_policy=True, max_episode_num=1)
            set_ob_rms(envs, obs_rms)
            set_ret_rms(envs, get_ret_rms(ppo_trainer.envs))
            episode_return, ood_ratio, episode_gae = evaluate(policy, envs, agent, compute_gae=compute_gae)
            envs.close()
            # print ([np.maximum(x, 0.).mean() for x in episode_gae])
            eval_result[agent] = [episode_return, ood_ratio, episode_gae]
            ood_list[i] = ood_ratio
            avg_score.append(np.array(episode_return).mean())
            with open(f'eval/{output_name}.pkl', 'wb') as f:
                pickle.dump(eval_result, f)
        print (agent, f'{episode_return.mean():.2f} +- {episode_return.std():.2f}', f'OOD ratio: {ood_ratio}')

    print ('avg score across all test agents: ', np.array(avg_score).mean())
    return np.array(avg_score).mean()


def evaluate_checkpoint(folder, test_set, interval=600, additional_suffix=None, seeds=None, deterministic=False, version=0):

    if seeds is None:
        seeds = os.listdir(folder)
    all_seed_scores = []
    for seed in seeds:
        seed_scores = {}
        iteration = interval
        while (1):
            test_set_name = test_set.split('/')[1]
            # suffix = f'{test_set_name}_cp_{iteration}_wo_height_check'
            suffix = f'{seed}_{test_set_name}_cp_{iteration}'
            if additional_suffix is not None:
                suffix = suffix + '_' + additional_suffix
            if deterministic:
                suffix = suffix + '_deterministic'
            if iteration == -1:
                model_path = f'{folder}/{seed}/Unimal-v0.pt'
            else:
                model_path = f'{folder}/{seed}/checkpoint_{iteration}.pt'
            agent_path = test_set
            policy_folder = f'{folder}/{seed}'
            print (model_path)
            # folder_name = folder.split('/')[-1]
            # if os.path.exists(f'eval/{folder_name}/{suffix}.pkl'):
            #     with open(f'eval/{folder_name}/{suffix}.pkl', 'rb') as f:
            #         results = pickle.load(f)
            #     avg_score = np.mean([np.mean(results[agent][0]) for agent in results])
            #     seed_scores[iteration] = avg_score
            #     iteration += interval
            #     continue
            if not os.path.exists(model_path):
                break
            if version == 0:
                score = evaluate_model(model_path, agent_path, policy_folder, suffix=suffix, compute_gae=False, \
                    terminate_on_fall=True, deterministic=deterministic)
            elif version == 1:
                score = evaluate_model_v2(model_path, agent_path, policy_folder, suffix=suffix, deterministic=deterministic)
            seed_scores[iteration] = score
            if iteration == -1:
                break
            iteration += interval
        all_seed_scores.append(seed_scores)
    for iteration in all_seed_scores[0].keys():
        avg_score = np.mean([seed_scores[iteration] for seed_scores in all_seed_scores])
        print ([seed_scores[iteration] for seed_scores in all_seed_scores])
        std = np.std([seed_scores[iteration] for seed_scores in all_seed_scores])
        print (f'iteration {iteration}: {avg_score} +- {std} ({len(all_seed_scores)} seeds)')



if __name__ == '__main__':
    
    # python tools/evaluate.py --folder distilled_policy/ft_ST_MLP_to_HN-MLP_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_weighted_loss --test_set data/test --interval 10 --seed 1409 --deterministic
    # python tools/evaluate.py --folder distilled_policy/ft_ST_MLP_to_TF_KL_loss_balanced_expert_size_8k*1000_weighted_loss --test_set data/test --interval 10 --seed 1409 --deterministic

    # python tools/eval_learning_curve.py --folder output/ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout/1409 --test_set unimals_100/train --interval 100
    # python tools/eval_learning_curve.py --folder output/ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout/1409 --test_set unimals_100/train_remove_level_1 --interval 100
    # python tools/eval_learning_curve.py --folder output/ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout/1409
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--test_set", default='unimals_100/test', type=str)
    parser.add_argument("--interval", default=600, type=int)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--seed", type=str, default='')
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--version", default=0, type=int)
    args = parser.parse_args()

    if args.seed == '':
        seeds = None
    else:
        seeds = [eval(x) for x in args.seed.split('+')]

    if args.deterministic:
        deterministic = True
    else:
        deterministic = False
    evaluate_checkpoint(args.folder, args.test_set, args.interval, args.suffix, seeds=seeds, deterministic=deterministic, version=args.version)
