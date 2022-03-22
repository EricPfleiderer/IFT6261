import torch

configs = {
    # DATASET
    'dataset_name': 'MNIST',
    'dataset_root': 'data/',
    'download_data': True,
    'shuffle_data': True,

    'attacker': 'GA',  # Type of attack algorithm (genetic algorithm, GANs)
    # 'defender': ,    # Defensive strategy to adopt during training (perturbations, etc)
    # 'context': ,  # Black box vs white box
}