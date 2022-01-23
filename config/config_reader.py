import configparser
import numpy as np


def config_reader():
    # config = configparser.ConfigParser()
    # config.read('./config/config.ini')
    # param = config['param']
    # my_param = {}
    # model_id = param['modelID']
    # model = config['models']
    # my_model = {}
    # my_model['boxsize'] = int(model['boxsize'])
    # my_model['stride'] = int(model['stride'])
    # my_model['padValue'] = int(model['padValue'])
    #
    # my_param['octave'] = int(param['octave'])
    # my_param['use_gpu'] = int(param['use_gpu'])
    # my_param['starting_range'] = float(param['starting_range'])
    # my_param['ending_range'] = float(param['ending_range'])
    # my_param['scale_search'] = list(map(float, param['scale_search']))
    # my_param['thre1'] = float(param['thre1'])
    # my_param['thre2'] = float(param['thre2'])
    # my_param['thre3'] = float(param['thre3'])
    # my_param['mid_num'] = int(param['mid_num'])
    # my_param['min_num'] = int(param['min_num'])
    # my_param['crop_ratio'] = float(param['crop_ratio'])
    # my_param['bbox_ratio'] = float(param['bbox_ratio'])
    # my_param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])
    #
    # return my_param, my_model

    param = {
    "use_gpu": 1,
    "GPUdeviceNumber": 0,
    "modelID": 1,
    "octave": 3,
    "starting_range": 0.8,
    "ending_range": 2,
    "scale_search": [0.5, 1, 1.5, 2],
    "thre1": 0.1,
    "thre2": 0.05,
    "thre3": 0.5,
    "min_num": 4,
    "mid_num": 10,
    "crop_ratio": 2.5,
    "bbox_ratio": 0.25
    }
    model = {
    "caffemodel": './model/_trained_COCO/pose_iter_440000.caffemodel',
    "deployFile": './model/_trained_COCO/pose_deploy.prototxt',
    "description": 'COCO Pose56 Two-level Linevec',
    "boxsize": 368,
    "padValue": 128,
    "np" :12,
    "stride": 8,
    "part_str": ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Leye", "Reye", "Lear", "Rear", "pt19"]

    }
    return param, model

if __name__ == "__main__":
    config_reader()
