import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger

from visual import visual_pos 





def main():
    """
    get input text
    ToDo skip if user input text in command
    current tasks:
         1 text 2 mtion
         2 motion transfer
         3 random sampling
         4 reconstruction

    ToDo 
    1 use one funtion for all expoert
    2 fitting smpl and export fbx in this file
    3 

    """
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")




    style_path = cfg.DEMO.style_motion_dir
    content_path = cfg.DEMO.content_motion_dir


    # 
    cfg.DEMO.TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                    'style_transfer' + cfg.DEMO.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda:0")
    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    # create mld model
    model = get_model(cfg, dataset)
    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    scale = cfg.DEMO.scale

    for content in os.listdir(content_path):
        if not content.endswith('.npy'):
            continue
        # prepare conent motion
        content_file_name = content.split('.')[0]
        content_file_path = os.path.join(content_path, content)
        content_motion = np.load(content_file_path)
        content_motion = np.array([content_motion])
        content_motion = torch.tensor(content_motion).to(device)
        
        # length is same as content motion
        length = content_motion.shape[1]
        lengths = [int(length)]


        for style in os.listdir(style_path):
            if not style.endswith('.npy'):
                continue
            # prepare style motion
            style_file_name = style.split('.')[0]
            style_file_path = os.path.join(style_path, style)
            style_motion = np.load(style_file_path)
            style_motion = np.array([style_motion])
            style_motion = torch.tensor(style_motion).to(device)

            # start
            with torch.no_grad():

                # prepare batch data
                batch = {"length": lengths, "style_motion": style_motion, "tag_scale": scale, "content_motion": content_motion}
                # joints,latents = model(batch)
                joints = model(batch)
                npypath = str(output_dir /
                            f"{content_file_name}_{style_file_name}_{str(lengths[0])}_scale_{str(scale).replace('.','-')}.npy")
                mp4path = npypath.replace('.npy', '.mp4')
                # with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                #     text_file.write('content {}'.format(content_file_name))
                #     text_file.write('#')
                #     text_file.write('style {}'.format(style_file_name))
                motion = joints[0].detach().cpu().numpy()
                np.save(npypath, motion)

                # visualization
                visual_pos(npypath, mp4path)

                logger.info(f"Motions are generated here:\n{npypath}")



if __name__ == "__main__":
    main()
