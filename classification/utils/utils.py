import os
import glob
import torch
import random
import numpy as np

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   获取新的日志保存编号
#---------------------------------------------------#
def get_log_sub_dir(log_name):
    file_pattern = os.path.join(log_name,'train*')
    matching_files = glob.glob(file_pattern)
    new_name = f'train{len(matching_files)+1}'
    return new_name
