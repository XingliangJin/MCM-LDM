import os
import numpy as np
import torch
import visual_motion.paramUtil as paramUtil
from visual_motion.plot_script import plot_3d_motion

import argparse

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--name', type=str, default='test1')
parser.add_argument('--motion_path', type=str, default=
'results/mld/test111/style_transfer2024-06-17-21-03'
)





def visual_pos(motion_path, save_path = './motion_output/59.mp4', caption = ' '):

#    motion_path = './datasets/cmu_new/test_file/000059.npy'
    skeleton = paramUtil.t2m_kinematic_chain
    #(F,22,3)
    motion = np.load(motion_path)[:, :22]




    print('generate video for', save_path)
    plot_3d_motion(save_path, skeleton, motion, caption, fps=20)




if __name__ == "__main__":

    #
    args = parser.parse_args()

    output_dir = 'motion_output'
    output_path = os.path.join(output_dir, args.name)

    # text_file_path = "/root/jxlcode/style_latent_diffusion/datasets/humanml3d/texts"
    # style_text_path = "/root/jxlcode/style_latent_diffusion/datasets/humanml3d/style_texts"

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    motion_path = args.motion_path
    file_list = os.listdir(motion_path)
    file_list.sort()

    for file in file_list:
        if '.npy' not in file:
            continue
        file_name = os.path.join(motion_path, file)
        save_path_1 = os.path.join(output_path, motion_path.split("/")[-1])
        if not os.path.exists(save_path_1):
            os.makedirs(save_path_1)
        save_path = os.path.join(save_path_1, '{}.mp4'.format(file.split('.')[0]))
        print('save video in {}'.format(save_path))


        #for origin
        # text_file = os.path.join(motion_path, file.split(".")[0] + '.txt')
        # with open(text_file, 'r') as f:
        #     action_desc = f.readline()



        # #for gt715
        # content = os.path.join(text_file_path, file.split(".")[0] + '.txt')
        # with open(content, "r") as text_file:
        #     text_line = text_file.readline().strip()
        #     text_line = text_line[:text_line.index("#")]
        # style = os.path.join(style_text_path, file.split(".")[0] + '.txt')
        # with open(style, "r") as style_text_file:
        #     style_text = style_text_file.read().strip()

        # caption = text_line + style_text




        visual_pos(file_name, save_path, "")
        # visual_pos(file_name, save_path, '')
        #visual_pos(file_name, save_path, caption)