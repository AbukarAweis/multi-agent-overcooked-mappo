from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

import os
import gym
import time

import mappo
import gen_outputs as go

import torch

# I reused a lot of code from main.py from my project 2

layout_abr = {'cramped_room': '1cram',
              'coordination_ring': '3coor',
              'counter_circuit_o_1order': '5coun'}

def train_or_test(mode, settings, save, name='', save_checkpoint=False):
    output = None
    if mode == 'train':
        training_eps = settings['training_eps']
        batch_size = settings['batch_size']
        num_epochs = settings['num_epochs']
        train_path = f'outputs/train_{training_eps}_{batch_size}_{num_epochs}_{name}.pth'
        check_path = f'outputs/check_{name}.pth'

        if os.path.exists(check_path):
            print(f'****Checkpoint Loaded ({name})****')
            checkpoint = torch.load(check_path, weights_only=False)
            train_data = mappo.train(**settings, name=name, checkpoint=checkpoint)
            torch.save(train_data, check_path)
        else:
            if not os.path.exists(train_path):
                train_data, output = mappo.train(**settings, name=name)

                if save:
                    if not os.path.exists('outputs'):
                        os.makedirs('outputs')
                    torch.save(train_data, train_path)
            else:
                print(f'****Train Data Loaded ({training_eps}|{batch_size}|{num_epochs}|{name})****')
                train_data = torch.load(train_path, weights_only=False)

        del settings['training_eps']
        del settings['batch_size']
        del settings['num_epochs']

        if save_checkpoint:
            checkpoint = f'outputs/check_{name}.pth'
            torch.save(train_data, checkpoint)

        return train_data

    elif mode == 'test':
        test_eps = settings['testing_eps']
        test_path = f'outputs/test_{test_eps}_{name}.pth'

        if not os.path.exists(test_path):
            test_data = mappo.test(**settings)

            if save:
                if not os.path.exists('outputs'):
                    os.makedirs('outputs')
                torch.save(test_data, test_path)
        else:
            print(f'****Test Data Loaded ({test_eps}|{name})****')
            test_data = torch.load(test_path, weights_only=False)

        del settings['testing_eps']
        del settings['trained_pi']

        return test_data

    else:
        raise ValueError("Mode must be either 'train' or 'test'")


def run(base_env, env, layout, params, nums, debug=False, save=False, save_checkpoint=False):
    settings = {"env": env, "params": params, "debug": debug}
    name = layout_abr[layout]

    print(f"--{layout} w/ params = {params}--")

    training_eps = nums[layout]["training_eps"]
    testing_eps = nums[layout]["testing_eps"]
    batch_size = nums[layout]["batch_size"]
    num_epochs = nums[layout]["num_epochs"]
    trained_pi = None

    for mode in ["train", "test"]:
    # for mode in ["train"]:
        start_time = time.perf_counter()
        print(f"\tStart <{mode}>")

        if mode == "train":
            settings.update({"training_eps": training_eps, "batch_size": batch_size, "num_epochs": num_epochs})
        elif mode == "test":
            if trained_pi is None:
                trained_data = torch.load(f"outputs/train_{training_eps}_{batch_size}_{num_epochs}_{name}.pth",
                                          weights_only=False)
                trained_pi = trained_data["pi"]
            settings.update({"testing_eps": testing_eps, "trained_pi": trained_pi})

        data = train_or_test(mode=mode, settings=settings, name=name, save=save,
                                     save_checkpoint=save_checkpoint)

        for metric_name, metric_data in data.items():
            if metric_name in ["pi", "pi_optimizer", "vf", "vf_optimizer", "episode_counter", "output"]:
                continue
            go.plot_metric(data=metric_data, mode=mode, name=name, num_epochs=num_epochs, metric_name=metric_name)

        go.write_output(out_list=data['output'], mode=mode, name=name, optuna=False)

        print(f"\tEnd  <{mode}> | Time Taken: {(time.perf_counter() - start_time) / 60:.2f} mins\n")

if __name__ == '__main__':

    params = {'pi_lr': 0.0003, 'vf_lr': 0.001, 'gamma': 0.99, 'epsilon': 0.2, 'entropy_coeff': 0.01}

    # 3 required layouts
    layouts = ['cramped_room', 'coordination_ring', 'counter_circuit_o_1order']

    # episodes/batch_size/num_epochs for each layout
    batch_size = 32
    num_epochs = 15

    testing_episodes = 100
    nums = {'cramped_room': {'training_eps': 7000, 'testing_eps': testing_episodes, 'batch_size': batch_size,
                             'num_epochs': num_epochs},
            'coordination_ring': {'training_eps': 38000, 'testing_eps': testing_episodes, 'batch_size': batch_size,
                                  'num_epochs': num_epochs},
            'counter_circuit_o_1order': {'training_eps': 85000, 'testing_eps': testing_episodes,
                                         'batch_size': batch_size, 'num_epochs': num_epochs}}

    start_time = time.perf_counter()
    # algorithm/reward-shaping/hyperparameters must be the same for all layouts
    for layout in layouts:
        reward_shaping = {'PLACEMENT_IN_POT_REW': 3, 'DISH_PICKUP_REWARD': 3, 'SOUP_PICKUP_REWARD': 5}

        # length of episodes. DO NOT MODIFY
        horizon = 400

        # build environment. DO NOT MODIFY
        mdp = OvercookedGridworld.from_layout_name(layout, reward_shaping=reward_shaping)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        env = gym.make(id="Overcooked-v0",
                       base_env=base_env,
                       featurize_fn=base_env.featurize_state_mdp,
                       disable_env_checker=True)

        run(base_env=base_env,
            env=env,
            layout=layout,
            params=params,
            nums=nums,
            debug=True,
            save=True,
            save_checkpoint=False)

    print(f'\n***TOTAL TIME: {((time.perf_counter() - start_time) / 60):.2f} MINS***')
