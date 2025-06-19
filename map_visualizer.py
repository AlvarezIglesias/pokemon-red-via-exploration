
import io
import zipfile
from reward_engine import Engine
import json
import os
import glob
import cv2
import numpy as np

ep_length = 2048 * 10 // 8
config = {
            'rank': 0,
            'init_state': './states/SuperMarioLand2.state',
            'checkpoint_path': 'C:/Users/chisp/Documents/GitHub/TFM//files/checkpoints',
            'img_path': 'C:/Users/chisp/Documents/GitHub/TFM//files/imgs',
            'gb_path': './roms/SuperMarioLand2.gb',
            'input_resolution': (144,160),
            'headless': True, 
            'gbc_mode': False,
            'print_rewards': True,
            'debug': False,
            'action_freq': 10, 
            'random_movement': False,
            'max_steps': ep_length,
            'smooth_frames': 10,
            'smooth_frames_thresh': 0.004,
            # 'frame_comparision_margin': 16,
            'preamble_size': ep_length,
            'new_map_multiplier': 1/25,
            'use_distance': False,
            'distance_k' : 1/100,
            'save_checkpoints': False,
            'use_checkpoints': False,
            'prioritize_recent_checkpoints': True,
            'prioritize_checkpoint_value': 'total',
            'baby_ranks': [0], #list of ranks that, even when using checkpoints, start from the begining
            'initial_map_resolution': (500,500),
            'chunk_extension_size': 160,
            'inmediate_offsets' : [16,32],
            'inmediate_threshold': 0.1, #pokemon 0.04
            'stay_threshold_margin': 0.005,
            'inmediate_entry_point_threshold': 0.04,
            'close_entry_point_threshold': 0.04,
            'close_entry_point_offsets' : [16],
            'past_entry_point_threshold': 0.04,
            'past_entry_point_list_max_size' : 50,
            'repeat_ward_max_size' : 50,

        }

def crop_around_point(array, center, max_width, max_height):
    h, w = array.shape[:2]
    cx, cy = center


    x1 = max(cx - max_width // 2, 0)
    y1 = max(cy - max_height // 2, 0)
    x2 = min(x1 + max_width, w)
    y2 = min(y1 + max_height, h)

    # edge
    x1 = max(x2 - max_width, 0)
    y1 = max(y2 - max_height, 0)

    cropped = array[y1:y2, x1:x2]
    offset = (x1, y1)
    return cropped

def load_checkpoint_efficient(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True) 
    files = glob.glob(checkpoint_path + "/*")
    if len(files) == 0: 
        raise FileNotFoundError

    print("Select file from ", str(len(files)), "files")
    # Load the zip archive from the disk
    with open(files[int(input())], 'rb') as f:
        zip_buffer = io.BytesIO(f.read())

    with zipfile.ZipFile(zip_buffer, 'r') as zip_file:

        # Load the reward engine
        reward_engine = Engine(config)
        reward_engine.efficient_dict_load(zip_file)
    
    return reward_engine


def load_map(engine):

    co = (144//2 - 8 ,160//2 - 16)
    original = np.copy(engine.cm.canvas).astype(np.uint8)
    original = cv2.cvtColor(original,cv2.COLOR_GRAY2RGB)

    entries = list(engine.map_archive[engine.cm].keys())

    offx = engine.cm.extend_offset[1]
    offy = engine.cm.extend_offset[0]

    for e in entries:
        ex = e[1] + offx + co[1]
        ey = e[0] + offy + co[0]

        cv2.rectangle(original,(ex,ey),(ex+16,ey+16), (255,0,0), 2)

    x = engine.cm.last_coors[1]
    y = engine.cm.last_coors[0]

    return original, entries, offx, offy, x, y

def display(engine):

    original, entries, offx, offy, x, y = load_map(engine)
    co = (144//2 - 8 ,160//2 - 16)

    window_name = "Map visualizer"

    cv2.namedWindow(window_name)

    while True:
        copy = np.copy(original)
        cropped = crop_around_point(copy, (x + offx  + co[1], y + offy + co[0]), 300, 300)
        cv2.rectangle(cropped,(150,150),(150+16,150+16), (0,255,0), 2)

        num_ep = 0

        if (y,x) in engine.map_archive[engine.cm]:
            num_ep = len(engine.map_archive[engine.cm][(y,x)])
            cv2.putText(cropped, str(num_ep),(150,150-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, cropped)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break
        elif key == ord('a'):
            x -= 16
            print(x,y)
        elif key == ord('d'):
            x += 16
            print(x,y)
        elif key == ord('w'):
            y -= 16
            print(x,y)
        elif key == ord('s'):
            y += 16
            print(x,y)
        elif key == ord('e'):
        
            print((y,x) in engine.map_archive[engine.cm].keys())
            # print(engine.map_archive[engine.cm].keys())
            entry = engine.map_archive[engine.cm][(y,x)][0]
            engine.cm = entry[0]
            engine.cm.last_coors = entry[1]
            original, entries, offx, offy, x, y = load_map(engine)
        elif key >= 48 and key < 58:
            num_key = (key - 48)

            if num_key > 0 and num_key < num_ep + 1:
                entry = engine.map_archive[engine.cm][(y,x)][num_key - 1]
                engine.cm = entry[0]
                engine.cm.last_coors = entry[1]
                original, entries, offx, offy, x, y = load_map(engine)

            
        
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = load_checkpoint_efficient("./checkpoints/")
    display(engine)