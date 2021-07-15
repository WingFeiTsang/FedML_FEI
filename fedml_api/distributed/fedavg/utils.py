import os
import sys, stat
import logging
import torch
import numpy as np


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "/home/zengrf/fedml/fedml_performance/tmp/"
    if not os.path.exists(pipe_path):
        os.mkdir(pip_path)
    pipe_path_file = pipe_path+"fedml.io"
    if not os.path.exists(pipe_path_file):
        os.mkfifo(pipe_path)

    logging.info("??????????????????????")
    logging.info(pipe_path)
    #pipe_fd = os.open(pipe_path, os.O_WRONLY|os.O_CREAT)
    pipe_fd = os.open(pipe_path_file, os.O_RDWR|os.O_CREAT, 777)
    logging.info("************************")
    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))
