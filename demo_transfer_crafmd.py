import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path
import datetime
import pickle
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





    eval_name = "content"
    eval_id = 0

    # chekpoints_str = cfg.TEST.CHECKPOINTS.split("/")[-1].split(".")[0].split("=")[1]
    save_path = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME)))
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_path, eval_name+'-'+str(eval_id)+'_expname_'+str(cfg.NAME)+"_scale_"+str(cfg.DEMO.scale).replace('.','-') + '.pkl')
    

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





    # MOTION = cfg.DEMO.MOTION
    scale = cfg.DEMO.scale

    sum1 = 0

    save_all = {}
    save_all["joints"]=[]
    save_all["id"]=[]
    save_all["label_content"]=[]
    save_all["label_style"]=[]

    for content in os.listdir(content_path):



        content_file_name = content.split('.')[0]
        content_file_path = os.path.join(content_path, content)
        content_motion = np.load(content_file_path)
        content_motion = np.array([content_motion])
        # content_motion = np.expand_dims(content_motion,0)
        content_motion = torch.tensor(content_motion).to(device)

        length = content_motion.shape[1]
        lengths = [int(length)]


        for style in os.listdir(style_path):
            print('sum={}, total 1600'.format(sum1))
            sum1 = sum1 + 1


            style_file_name = style.split('.')[0]
            style_file_path = os.path.join(style_path, style)
            style_motion = np.load(style_file_path)
            style_motion = np.array([style_motion])
            style_motion = torch.tensor(style_motion).to(device)


            with torch.no_grad():
                rep_lst = []    
                rep_ref_lst = []
                texts_lst = []

                # prepare batch data
                batch = {"length": lengths, "style_motion": style_motion, "tag_scale": scale, "content_motion": content_motion}
                joints = model(batch)
                # npypath = str(output_dir /
                #             f"{content_file_name}_{style_file_name}_{str(lengths[0])}_scale_{str(scale).replace('.','-')}.npy")
                # np.save(npypath, joints[0].detach().cpu().numpy())
                idid = "content"+content_file_name+"_"+"style"+style_file_name+"_scale_"+str(scale).replace('.','-')
                save_all["joints"].append(joints[0].detach().cpu().numpy())
                save_all["id"].append(idid)
                save_all["label_content"].append(content_file_name.split("-")[-1])
                save_all["label_style"].append(style_file_name.split("-")[-1])


    
    # a = [1, 2, 3]
    with open(save_path, 'wb') as f:
	    pickle.dump(save_all, f)
                

                    
                    



    # if cfg.DEMO.RENDER:
    #     # plot with lines
    #     # from mld.data.humanml.utils.plot_script import plot_3d_motion
    #     # fig_path = Path(str(npypath).replace(".npy",".mp4"))
    #     # plot_3d_motion(fig_path, joints, title=text, fps=cfg.DEMO.FRAME_RATE)

    #     # single render
    #     # from mld.utils.demo_utils import render
    #     # figpath = render(npypath, cfg.DATASET.JOINT_TYPE,
    #     #                  cfg_path="./configs/render_cx.yaml")
    #     # logger.info(f"Motions are rendered here:\n{figpath}")

    #     from mld.utils.demo_utils import render_batch

    #     blenderpath = cfg.RENDER.BLENDER_PATH
    #     render_batch(os.path.dirname(npypath),
    #                  execute_python=blenderpath,
    #                  mode="sequence")  # sequence
    #     logger.info(f"Motions are rendered here:\n{os.path.dirname(npypath)}")


if __name__ == "__main__":
    main()
