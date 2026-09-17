import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 19})

abbreviations = {
                'pi_loss': 'AL',
                'pi0_loss': 'AL',
                'pi1_loss': 'AL',
                 'vf_loss': 'CL',
                 'pi_entropy': 'AE',
                 'pi0_entropy': 'AE',
                 'pi1_entropy': 'AE',
                 'reward_history': 'REW',
                 'time_history': 'TIME',
                 'soups_delivered_history': 'SOUP',
                 'useful_dish_pickup_history': 'uDISH',
                'mean_soup_delivered_history': 'AVG',
                 'dish_pickup_history': 'xDISH'}

formatted_name = {
                'pi_loss': 'Actor Loss',
                'pi0_loss': 'Actor Loss',
                'pi1_loss': 'Actor Loss',
                 'vf_loss': 'Critic Loss',
                 'pi_entropy': 'Actor Entropy',
                 'pi0_entropy': 'Actor Entropy',
                 'pi1_entropy': 'Actor Entropy',
                 'reward_history': 'Shaped Rewards (Average)',
                  'time_history': 'Runtime',
                  'soups_delivered_history': 'Total # Soups Delivered',
                 'useful_dish_pickup_history': 'Useful Dishes Picked Up',
                'mean_soup_delivered_history': 'Average # of Soups Delivered',
                 'dish_pickup_history': 'Dishes Picked Up'}

y_label_name = {
                'pi_loss': 'Total Loss (MAPPO Objective + Entropy)',
                'pi0_loss': 'Total Loss (MAPPO Objective + Entropy)',
                'pi1_loss': 'Total Loss (MAPPO Objective + Entropy)',
                 'vf_loss': 'MSE Loss',
                 'pi_entropy': 'Entropy',
                 'pi0_entropy': 'Entropy',
                 'pi1_entropy': 'Entropy',
                 'reward_history': 'Shaped Rewards',
                 'time_history': 'Runtime (ms)',
                'soups_delivered_history': 'Count',
                'useful_dish_pickup_history': 'Count',
                'mean_soup_delivered_history': 'Count',
                'dish_pickup_history': 'Count'}

legend_name = {
                'pi_loss': 'Loss',
                'pi0_loss': 'Loss',
                'pi1_loss': 'Loss',
                 'vf_loss': 'Loss',
                 'pi_entropy': 'Entropy',
                 'pi0_entropy': 'Entropy',
                 'pi1_entropy': 'Entropy',
                 'reward_history': 'Rewards',
                 'time_history': 'Runtime',
               'soups_delivered_history': 'SOUP',
               'useful_dish_pickup_history': 'uDISH',
                'mean_soup_delivered_history':'AVG',
               'dish_pickup_history': 'xDISH'}

param_name = {
        'pi_lr': 'Actor Learning Rate',
        'vf_lr': 'Critic Learning Rate',
        'gamma': 'Gamma',
        'epsilon': 'Epsilon',
        'entropy_coeff': 'Entropy Coefficient'
    }

param_symbol = {
    'pi_lr': r'$\alpha_{\pi}$',
    'pi1_lr': r'$\alpha_{\pi}1$',
    'vf_lr': r'$\alpha_{V}$',
    'gamma': r'$\gamma$',
    'epsilon': r'$\epsilon$',
    'entropy_coeff': r'$c_2$'
}

layout_name = {'1cram': 'Cramped Room',
              '3coor': 'Coordination Ring',
              '5coun': 'Counter Circuit'}

def calc_cumulative_rewards(rewards):
    return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)

def smooth_data(data, window=100):
    return pd.Series(data).rolling(window=window, min_periods=1).mean().tolist()

def plot_metric(data, mode, name, num_epochs, metric_name="Metric"):
    plt.figure(figsize=(10, 6))

    agent_colors = ['green', 'blue']

    m = 'Training' if mode == 'train' else 'Testing'
    n = layout_name[name]

    if 'loss' in metric_name or 'entropy' in metric_name:
        x_label = f"# Batches ({num_epochs} Updates/Batch)"
    else:
        x_label = f'# {m} Episodes'

    if 'soups' in metric_name or 'dish' in metric_name:
        modified_data = []
        running_totals = [0, 0, 0]

        for row in data:
            running_totals[0] += row[0]
            running_totals[1] += row[1]
            running_totals[2] = running_totals[0] + running_totals[1]
            modified_data.append(running_totals[:])

        num_agents = len(modified_data[0]) - 1
        labels = [f"Agent {i}" for i in range(num_agents)] + ["Total"]

        agent_value = [[timestep[i] for timestep in modified_data] for i in range(num_agents)]
        total_value = [timestep[-1] for timestep in modified_data]

        for i in range(num_agents):
            plt.plot(range(1, len(agent_value[i]) + 1), agent_value[i],
                     color=agent_colors[i % len(agent_colors)], label=f'{labels[i]} ({agent_value[i][-1]})')

        plt.plot(range(1, len(total_value) + 1), total_value, color='k', linestyle='dashed', label=f"Total ({total_value[-1]})")

    elif 'reward' in metric_name and isinstance(data[0], list) and isinstance(data[0][0], (int, float)):
        num_agents = len(data[0])
        labels = [f"Agent {i}" for i in range(num_agents)]
        agent_rewards = [[timestep[i] for timestep in data] for i in range(num_agents)]
        for i in range(num_agents):
            smoothed_data = smooth_data(agent_rewards[i])
            plt.plot(range(1, len(smoothed_data) + 1), smoothed_data,
                     color=agent_colors[i % len(agent_colors)], label=labels[i])

    elif 'mean' in metric_name:
        plt.plot(range(1, len(data) + 1), data, color='red')
        plt.axhline(y=7, color='green', linestyle='dashed', linewidth=2, label="Goal")
        if mode == 'train':
            plt.axvline(x=len(data), color='orange', linestyle='dashed', linewidth=2, label=f"Cumulative Avg. ({round(data[-1], 1)})")
        elif mode == 'test':
            plt.axvline(x=len(data), color='orange', linestyle='dashed', linewidth=2, label=f"Rolling Avg. ({round(data[-1], 1)})")

    elif isinstance(data[0], list):
        num_agents = len(data)
        labels = [f"Agent {i}" for i in range(num_agents)]
        for i in range(num_agents):
            agent_data = [float(val) for val in data[i]]
            smoothed_data = smooth_data(agent_data)
            plt.plot(range(1, len(smoothed_data) + 1), smoothed_data,
                     color=agent_colors[i % len(agent_colors)], label=labels[i])
    else:
        data = [float(val) for val in data]
        smoothed_data = smooth_data(data)
        plt.plot(range(1, len(smoothed_data) + 1), smoothed_data, color='red', label="Total")

    plt.xlabel(x_label)
    plt.ylabel(y_label_name[metric_name])
    plt.title(f"{formatted_name[metric_name]} vs. Episodes | {n}")

    hide_legend_metrics = {'vf_loss', 'time_history', 'pi_loss', 'pi_entropy'}

    if metric_name not in hide_legend_metrics:
        plt.legend()

    if 'entropy' in metric_name or 'loss' in metric_name:
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}k' if x >= 10000 else int(x)))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y / 1000)}k' if y >= 10000 else f'{y:.2f}'))
    else:
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}k' if x >= 10000 else int(x)))
        if 'mean' in metric_name and mode == 'test':
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y / 1000)}k' if y >= 10000 else f'{y:.1f}'))
        else:
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y / 1000)}k' if y >= 10000 else int(y)))

    plt.grid()
    save_plot(name, mode, metric_name)
    plt.close()

def save_plot(name, mode, metric_name, param_name='', multi=False):
    directory = f'outputs/images/{name}' if not multi else f'outputs/images/{name}/{metric_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not multi:
        plt.savefig(os.path.join(directory, f'{abbreviations[metric_name]}_{mode}_{name}.png'))
    else:
        plt.savefig(os.path.join(directory, f'{abbreviations[metric_name]}_{param_name}_{mode}.png'))

def write_output(out_list, mode, name, optuna=False):
    directory = f'outputs'
    if not os.path.exists(directory):
        os.makedirs(directory)

    if optuna:
        file_path = os.path.join(directory, f'P3_opt.txt')
    else:
        file_path = os.path.join(directory, f'P3_{mode}_{name}.txt')

    count = 0
    batch_output = []
    batch = []
    with open(file_path, 'w') as file:
        for entry in out_list:
            if isinstance(entry, dict):
                file.write(f'params: {str(entry)}\n\n')
                file.write('=' * 60 + '\n')
            elif 'Batch' in entry:
                entry = entry.replace('\t', '')
                if not batch:
                    batch.append(f'{"-" * 60}')
                    batch.append(f'EP # {count:>4}{" " * 2}{entry.replace("Batch Update", "").strip()}')
                else:
                    batch.append(f'{" " * 12}{entry.split("|")[-1]}')
            else:
                count += 1
                file.write(str(entry) + '\n')

                if batch:
                    batch_output.append(batch.copy())
                    batch.clear()

        if batch:
            batch_output.append(batch.copy())

        file.write('=' * 60 + '\n')

        if mode == 'train':
            file.write(f'\n\t\t{len(batch_output[0]) - 1} Updates Per Batch\n')
            for row in batch_output:
                for entry in row:
                    file.write(str(entry) + '\n')
