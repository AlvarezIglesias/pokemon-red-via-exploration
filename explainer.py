import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency
from captum.attr import IntegratedGradients 
from torchvision import transforms
from PIL import Image
import os

from train_cnn_lstm import CustomPolicy  # Update with your filename
from sb3_contrib import RecurrentPPO

import numpy as np
import os
import glob
import io
import zipfile
from reward_engine import Engine
import time
import torch as th
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from captum.attr import IntegratedGradients


import tempfile

def load_all_at_once(npz_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(npz_file, 'r') as zip_ref:
            zip_ref.extract("dabug_inputs.npz", tmpdir)
            extracted_npz_path = os.path.join(tmpdir, "dabug_inputs.npz")

            with np.load(extracted_npz_path) as npz:
                data = [npz[name] for name in npz.files]
                return data

def open_checkpoint(dir):
    os.makedirs(dir, exist_ok=True) 
    files = glob.glob(os.path.join(dir, "*"))
    file_index = int(input(f"Select file from {len(files)} files: "))
    
    print(files[file_index])
    all_data = load_all_at_once(files[file_index])
    return all_data


model_path = ".\\model_cnn_lstm_ep_pre.zip"
frames_dir = ".\\examples\\PokemonRed\\PR_cnn_lstm_ep_pre\\checkpoints"  # Directory with image files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_size = (144, 160) 


frames = open_checkpoint(frames_dir)
model = RecurrentPPO.load(model_path)
model.policy.to(device)

def lstm_to_np(lstm_tensor):
    hidden_tensor = lstm_tensor[0][0]  # Shape (1, 1, 256)
    hidden_vector = hidden_tensor.squeeze()  # Shape (256,)
    image_tensor = hidden_vector.view(16, 16)  # Shape (16, 16)
    return image_tensor.cpu().detach().numpy()


def forward_func(input_tensor):
    global i, next_lstm

    batch = 1

    episode_starts = [True] * batch if i == 0 else [False] * batch
    episode_starts = th.tensor(episode_starts, dtype=th.float32, device=device)

            
    obs_tensor = input_tensor.unsqueeze(0)
    latent_pi, _, next_lstm, lstm_vf = model.policy.explainer_forward(obs_tensor, next_lstm, episode_starts)

    return latent_pi[0]
    # return lstm_vf[0][0][0]



# Visualization
def visualize_attribution(orig_tensor, attribution, title="Saliency Map", filename=None):

    
    orig_np = orig_tensor.permute(1, 2, 0).cpu().numpy()/256
    saliency_map = np.abs(attribution.cpu().numpy()).sum(axis=0)

    plt.figure(title, figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(orig_np)
    plt.title("Original Frame")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(saliency_map, cmap='hot')
    plt.title(title)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(lstm_to_np(next_lstm), cmap='hot')
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show(block=True)
    # plt.pause(0.1)
    # plt.close()

# Captum saliency
saliency = Saliency(forward_func)

# Run
i = 0
single_hidden_state_shape = (1, 1, 256) #(lstm.num_layers, self.n_envs, lstm.hidden_size)
next_lstm = RNNStates(
    (
        th.zeros(single_hidden_state_shape, device=device),
        th.zeros(single_hidden_state_shape, device=device),
    ),
    (
        th.zeros(single_hidden_state_shape, device=device),
        th.zeros(single_hidden_state_shape, device=device),
    ),
)


i = 0
while True:

    try:
        skip_n = int(input())
    except:
        skip_n = 1

    for _ in range(skip_n):
        frame = frames[i]
        frame_tensor = torch.tensor(frame).repeat(3, 1, 1).to(device)
        forward_func(frame_tensor)
        i += 1

    frame = frames[i]
    frame_tensor = torch.tensor(frame).repeat(3, 1, 1).to(device)
    attr_pi = saliency.attribute(frame_tensor, abs=False) # error
    visualize_attribution(frame_tensor, attr_pi, title=f"Saliency_cnn")

