from metamorph.envs.jax.unimal import Unimal

import jax
from jax import numpy as jp

from brax.io.image import render_array

import time
import matplotlib.pyplot as plt

import functools
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


xml_path = 'data/train/xml/floor-1409-0-3-01-15-56-55.xml'
agent = Unimal(xml_path)
action_dim = agent.sys.num_links() * 2

jit_env_reset = jax.jit(agent.reset)
jit_env_step = jax.jit(agent.step)

# start = time.time()
# state = jit_env_reset(rng=jax.random.PRNGKey(seed=0))
# end = time.time()
# print (end - start)

# # save image
# # im = render_array(agent.sys, state.pipeline_state, 256, 256)
# # plt.imsave('figures/test.png', im)

# episode_length = 2560
# random_action = jax.random.normal(jax.random.PRNGKey(seed=1), shape=(episode_length, action_dim))

# start = time.time()
# for t in range(episode_length):
#     state = jit_env_step(state, random_action[t])
# end = time.time()
# print (end - start)

train_fn = functools.partial(
    ppo.train, num_timesteps=10_000_000, num_evals=5, reward_scaling=0.1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
    discounting=0.97, learning_rate=3e-4, entropy_cost=0., num_envs=2048,
    batch_size=5120, seed=0)


x_data = []
y_data = []
ydataerr = []
times = [time.time()]

max_y, min_y = 13000, 0
def progress(num_steps, metrics):
  times.append(time.time())
  x_data.append(num_steps)
  y_data.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])
  print (x_data, y_data)

#   plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
#   plt.ylim([min_y, max_y])

#   plt.xlabel('# environment steps')
#   plt.ylabel('reward per episode')
#   plt.title(f'y={y_data[-1]:.3f}')

#   plt.errorbar(
#       x_data, y_data, yerr=ydataerr)
#   plt.show()

make_inference_fn, params, _= train_fn(environment=agent, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
