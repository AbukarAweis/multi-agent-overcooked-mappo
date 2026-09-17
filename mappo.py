import torch
import time
import numpy as np

from neural_networks import PolicyNetwork, ValueFunctionNetwork

import gen_outputs as go

DEBUG = None

def out_debug(msg, out_list):
    out_list.append(msg)
    if DEBUG:
        print(msg)

def train(env, training_eps, batch_size, num_epochs, params, name, checkpoint=None, debug=False):
    global DEBUG
    DEBUG = debug

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pi = PolicyNetwork()
    vf = ValueFunctionNetwork()

    if checkpoint is not None:
        pi.load_state_dict(checkpoint['pi'])
        vf.load_state_dict(checkpoint['vf'])

    pi.train()
    vf.train()

    pi.to(device)
    vf.to(device)

    epsilon = params['epsilon']
    gamma = params['gamma']
    pi_lr = params['pi_lr']
    vf_lr = params['vf_lr']
    entropy_coeff = params['entropy_coeff']

    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(vf.parameters(), lr=vf_lr)

    if checkpoint is not None:
        pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
        vf_optimizer.load_state_dict(checkpoint['vf_optimizer'])

    episode_data = {}
    pi_loss_curve = []
    pi_entropy_curve = []
    vf_loss_curve = []
    output = [params]

    episode_counter = 0 if checkpoint is None else len(checkpoint['reward_history'])
    reward_history = [] if checkpoint is None else checkpoint['reward_history']
    soups_delivered_history = [] if checkpoint is None else checkpoint['soups_delivered_history']
    mean_soup_delivered_history = [] if checkpoint is None else checkpoint['mean_soup_delivered_history']
    useful_dish_pickup_history = [] if checkpoint is None else checkpoint['useful_dish_pickup_history']
    dish_pickup_history = [] if checkpoint is None else checkpoint['dish_pickup_history']
    total_soups_delivered = 0 if checkpoint is None else sum(ep[2] for ep in checkpoint['soups_delivered_history'])

    out_debug('', checkpoint['output']) if checkpoint is not None else 0

    final_episode = training_eps if checkpoint is None else training_eps + episode_counter

    while episode_counter < final_episode:
        obs = env.reset()
        obs_arr = obs["both_agent_obs"]
        obs0 = torch.zeros(96, device=device)
        obs1 = torch.zeros(96, device=device)
        obs0.copy_(torch.from_numpy(obs_arr[0]).to(device))
        obs1.copy_(torch.from_numpy(obs_arr[1]).to(device))

        ep_traj = []
        ep_rewards = [0, 0]
        done = False

        num_soups_delivered = 0
        num_soups_delivered_agents = [0, 0]
        num_useful_dish_pickups = [0, 0]
        num_dish_pickups = [0, 0]

        while not done:
            with torch.inference_mode():
                #centralized
                # logits0 = pi(torch.cat((obs0, obs1), dim=-1))
                # logits1 = pi(torch.cat((obs1, obs0), dim=-1))

                logits0 = pi(obs0)
                logits1 = pi(obs1)

                dist0 = torch.distributions.Categorical(logits=logits0)
                dist1 = torch.distributions.Categorical(logits=logits1)

                a0 = dist0.sample()
                a1 = dist1.sample()

                logp0 = dist0.log_prob(a0)
                logp1 = dist1.log_prob(a1)

            next_obs, reward, done, info = env.step([a0.item(), a1.item()])

            next_obs_arr = next_obs["both_agent_obs"]
            next_obs0 = torch.from_numpy(next_obs_arr[0]).float().to(device)
            next_obs1 = torch.from_numpy(next_obs_arr[1]).float().to(device)

            r_shaped = info["shaped_r_by_agent"]
            if env.agent_idx:
                r_shaped_0 = r_shaped[1] + reward
                r_shaped_1 = r_shaped[0] + reward
            else:
                r_shaped_0 = r_shaped[0] + reward
                r_shaped_1 = r_shaped[1] + reward

            num_soups_delivered += int(reward / 20)

            # ep_traj.append((obs0, a0, r_shaped_0, logp0))
            # ep_traj.append((obs1, a1, r_shaped_1, logp1))

            #single obs for actor, double obs for critic
            ep_traj.append((obs0, torch.cat((obs0, obs1), dim=-1), a0, r_shaped_0, logp0))
            ep_traj.append((obs1, torch.cat((obs1, obs0), dim=-1), a1, r_shaped_1, logp1))

            ep_rewards[0] += r_shaped_0
            ep_rewards[1] += r_shaped_1

            obs0 = next_obs0
            obs1 = next_obs1

        num_useful_dish_pickups[0] += len(info["episode"]["ep_game_stats"]["useful_dish_pickup"][0])
        num_useful_dish_pickups[1] += len(info["episode"]["ep_game_stats"]["useful_dish_pickup"][1])

        num_dish_pickups[0] += len(info["episode"]["ep_game_stats"]["dish_pickup"][0])
        num_dish_pickups[1] += len(info["episode"]["ep_game_stats"]["dish_pickup"][1])

        num_soups_delivered_agents[0] += int(info["episode"]["ep_sparse_r_by_agent"][0] / 20)
        num_soups_delivered_agents[1] += int(info["episode"]["ep_sparse_r_by_agent"][1] / 20)

        soups_delivered_history.append(num_soups_delivered_agents + [num_soups_delivered])

        total_soups_delivered += num_soups_delivered
        mean_soups_delivered = total_soups_delivered / (episode_counter + 1)
        mean_soup_delivered_history.append(mean_soups_delivered)

        useful_dish_pickup_history.append(num_useful_dish_pickups[:] + [0])
        dish_pickup_history.append(num_dish_pickups[:] + [0])

        rewards_to_go = calc_rewards_to_go(ep_traj, gamma)
        value_estimates = [vf(obs_joint).item() for _, obs_joint, _, _, _ in ep_traj]
        advantages = calc_advantage_estimate(rewards_to_go, value_estimates)

        episode_data.update({episode_counter: (ep_traj, rewards_to_go, value_estimates, advantages)})
        reward_history.append([ep_rewards[0], ep_rewards[1]])

        rolling_avg_rewards = [
            sum([ep[0] for ep in reward_history[-100:]]) / min(len(reward_history), 100),
            sum([ep[1] for ep in reward_history[-100:]]) / min(len(reward_history), 100)
        ]

        if (episode_counter + 1) % 100 == 0:
            out_debug(
                f'Train EP #{episode_counter + 1:>4} |'
                f' Tot. Rew. = {sum(ep_rewards):>6.2f} [{ep_rewards[0]:>5.1f}, {ep_rewards[1]:>5.1f}] |'
                f' # Soups = {num_soups_delivered:>3} '
                f'(Tot: {sum(ep[2] for ep in soups_delivered_history):>2}, Avg: {mean_soups_delivered:>7.4f})'
                f' RA. (100) = [{rolling_avg_rewards[0]:>4.2f}, {rolling_avg_rewards[1]:>4.2f}]', output)
        else:
            out_debug(
                f'Train EP #{episode_counter + 1:>4} |'
                f' Tot. Rew. = {sum(ep_rewards):>6.2f} [{ep_rewards[0]:>5.1f}, {ep_rewards[1]:>5.1f}] |'
                f' # Soups = {num_soups_delivered:>3} '
                f'(Tot: {sum(ep[2] for ep in soups_delivered_history):>2}, Avg: {mean_soups_delivered:>7.4f})', output)

        episode_counter += 1

        if episode_counter % batch_size == 0 or episode_counter == training_eps:

            pi_losses = []
            pi_entropies = []
            vf_losses = []

            for _ in range(num_epochs):
                pi, pi_loss, pi_entropy = update_policy(pi, pi_optimizer, episode_data, epsilon, entropy_coeff, device)
                vf, vf_loss = update_value_function(vf, vf_optimizer, episode_data, device)

                pi_losses.append(pi_loss)
                pi_entropies.append(pi_entropy)
                vf_losses.append(vf_loss)

                out_debug(
                    f'\tBatch Update | Avg. Loss: PI0 = {pi_loss:>6.2f}'
                    f' VF = {vf_loss:>6.2f}', output)

            pi_loss_curve.append(sum(pi_losses) / len(pi_losses))
            pi_entropy_curve.append(sum(pi_entropies) / len(pi_entropies))
            vf_loss_curve.append(sum(vf_losses) / len(vf_losses))

            episode_data.clear()

        train_data = {
            'pi': pi.state_dict(),
            'pi_optimizer': pi_optimizer.state_dict(),
            'pi_loss': pi_loss_curve,
            'pi_entropy': pi_entropy_curve,
            'vf': vf.state_dict(),
            'vf_optimizer': vf_optimizer.state_dict(),
            'vf_loss': vf_loss_curve,
            'reward_history': reward_history,
            'output': output,
            'soups_delivered_history': soups_delivered_history,
            'mean_soup_delivered_history': mean_soup_delivered_history,
            'useful_dish_pickup_history': useful_dish_pickup_history,
            'dish_pickup_history': dish_pickup_history
        }

        if episode_counter % 500 == 0:
            train_path = f'outputs/train_{training_eps}_{batch_size}_{num_epochs}_{name}.pth'
            for metric_name, metric_data in train_data.items():
                if metric_name in ["pi", "pi_optimizer", "vf", "vf_optimizer", "episode_counter", "output"]:
                    continue
                go.plot_metric(data=metric_data, mode='train', name=name, num_epochs=num_epochs,
                               metric_name=metric_name)
            go.write_output(out_list=output, mode='train', name=name, optuna=False)
            torch.save(train_data, train_path)
            print(f'\t***Saved Data for Previous 500 Episodes***')

            if round(mean_soups_delivered, 4) >= 7.0000:
                print(f'\***THRESHOLD REACHED @ EP # {episode_counter}***')
                break

    return train_data, output

def update_policy(pi, optimizer, episode_data, epsilon, entropy_coeff, device):
    observations, actions, old_log_probs, advantages = [], [], [], []

    for episode in episode_data.values():
        ep_traj, _, _, ep_ae = episode
        for i in range(len(ep_traj)):
            obs, _, action, _, old_action_prob = ep_traj[i]
            observations.append(obs)
            actions.append(action)
            old_log_probs.append(old_action_prob)
            advantages.append(ep_ae[i])

    observations = torch.stack(observations)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)
    advantages = torch.from_numpy(np.array(advantages)).float().to(device)

    logits = pi(observations)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)

    r_t = torch.exp(new_log_probs - old_log_probs)
    clipped_r_t = torch.clamp(r_t, 1 - epsilon, 1 + epsilon)
    entropy_bonus = dist.entropy().mean()
    loss = -torch.min(r_t * advantages, clipped_r_t * advantages).mean() - entropy_coeff * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pi.parameters(), max_norm=0.5)
    optimizer.step()

    return pi, loss.item(), entropy_bonus.item()

def update_value_function(vf, optimizer, episode_data, device):
    observations, target_values = [], []

    for episode in episode_data.values():
        ep_traj, ep_rtg, _, _ = episode
        for i in range(len(ep_traj)):
            _, obs, _, _, _ = ep_traj[i]
            observations.append(obs)
            target_values.append(ep_rtg[i])

    observations = torch.stack(observations).to(device)
    target_values = torch.from_numpy(np.array(target_values)).float().to(device)
    # target_values = (target_values - target_values.mean()) / (target_values.std() + 1e-8)

    value_estimates = vf(observations).squeeze(-1)
    loss = torch.nn.functional.mse_loss(value_estimates, target_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm=0.5)
    optimizer.step()

    return vf, loss.item()

def calc_rewards_to_go(trajectories, gamma):
    rewards = np.array([step[3] for step in trajectories], dtype=np.float32)
    rtg_list = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        rtg_list[i] = rewards[i] + (gamma * rtg_list[i + 1] if i + 1 < len(rewards) else 0)
    rtg_list = (rtg_list - rtg_list.mean()) / (rtg_list.std() + 1e-8)
    return rtg_list.tolist()

def calc_advantage_estimate(rewards_to_go, value_estimates):
    return [rtg - v for rtg, v in zip(rewards_to_go, value_estimates)]

def test(env, trained_pi, testing_eps, params, debug=False):
    global DEBUG
    DEBUG = debug

    pi = PolicyNetwork()
    pi.load_state_dict(trained_pi)
    pi.eval()

    episode_counter = 0
    reward_history = []
    time_history = []
    soups_delivered_history = []
    mean_soup_delivered_history = []
    output = [params]

    while episode_counter < testing_eps:
        obs = env.reset()
        obs_arr = obs["both_agent_obs"]
        obs0 = torch.from_numpy(obs_arr[0]).float()
        obs1 = torch.from_numpy(obs_arr[1]).float()

        ep_rewards = [0, 0]
        done = False
        start_time = time.perf_counter()
        num_soups_delivered = 0
        num_soups_delivered_agents = [0, 0]

        while not done:
            with torch.inference_mode():
                logits0 = pi(obs0)
                logits1 = pi(obs1)

                dist0 = torch.distributions.Categorical(logits=logits0)
                dist1 = torch.distributions.Categorical(logits=logits1)

                a0 = dist0.sample()
                a1 = dist1.sample()

            next_obs, reward, done, info = env.step([a0.item(), a1.item()])

            next_obs_arr = next_obs["both_agent_obs"]
            obs0 = torch.from_numpy(next_obs_arr[0]).float()
            obs1 = torch.from_numpy(next_obs_arr[1]).float()

            r_shaped = info["shaped_r_by_agent"]
            if env.agent_idx:
                r_shaped_0 = r_shaped[1]
                r_shaped_1 = r_shaped[0]
            else:
                r_shaped_0 = r_shaped[0]
                r_shaped_1 = r_shaped[1]

            num_soups_delivered += int(reward / 20)

            ep_rewards[0] += r_shaped_0
            ep_rewards[1] += r_shaped_1

        num_soups_delivered_agents[0] += len(info["episode"]["ep_game_stats"]["soup_delivery"][0])
        num_soups_delivered_agents[1] += len(info["episode"]["ep_game_stats"]["soup_delivery"][1])

        soups_delivered_history.append(num_soups_delivered_agents + [num_soups_delivered])

        mean_soups_delivered = sum(ep[2] for ep in soups_delivered_history) / (episode_counter + 1)
        mean_soup_delivered_history.append(mean_soups_delivered)

        reward_history.append([ep_rewards[0], ep_rewards[1]])
        time_history.append((time.perf_counter() - start_time) * 1000)

        rolling_avg_rewards = [
            sum([ep[0] for ep in reward_history[-100:]]) / min(len(reward_history), 100),
            sum([ep[1] for ep in reward_history[-100:]]) / min(len(reward_history), 100)
        ]

        if (episode_counter + 1) % 100 == 0:
            out_debug(f'Test EP #{episode_counter + 1:>4} |'
                      f' Tot. Rew. = {sum(ep_rewards):>6.2f} [{ep_rewards[0]:>5.1f}, {ep_rewards[1]:>5.1f}] |'
                      f' # Soups = {num_soups_delivered:>3} '
                      f'(Tot: {sum(ep[2] for ep in soups_delivered_history):>2}, Avg: {mean_soups_delivered:>7.4f})'
                      f' RA. (100) = [{rolling_avg_rewards[0]:>4.2f}, {rolling_avg_rewards[1]:>4.2f}]', output)
        else:
            out_debug(f'Test EP #{episode_counter + 1:>4} |'
                      f' Tot. Rew. = {sum(ep_rewards):>6.2f} [{ep_rewards[0]:>5.1f}, {ep_rewards[1]:>5.1f}] |'
                      f' # Soups = {num_soups_delivered:>3} '
                      f'(Tot: {sum(ep[2] for ep in soups_delivered_history):>2}, Avg: {mean_soups_delivered:>7.4f})', output)

        episode_counter += 1

    test_data = {
        'reward_history': reward_history,
        'time_history': time_history,
        'soups_delivered_history': soups_delivered_history,
        'mean_soup_delivered_history': mean_soup_delivered_history,
        'output': output
    }

    return test_data
