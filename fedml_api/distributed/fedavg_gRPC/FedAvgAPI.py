import logging
import socket, fcntl, struct
import csv
from mpi4py import MPI

from .FedAVGAggregator import FedAVGAggregator
from .FedAVGTrainer import FedAVGTrainer
from .FedAvgClientManager import FedAVGClientManager
from .FedAvgServerManager import FedAVGServerManager

from ...standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from ...standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from ...standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG


def FedML_init(csvfile):
    host_name = socket.getfqdn(socket.gethostname())
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if_name = b'ens33'
    host_ip = socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,
        struct.pack('256s', if_name[:15])
    )[20:24])
    worker_number = 0
    with open(csvfile, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            worker_number = worker_number + 1
            if row[1] == host_ip:
                process_id = row[0]
    worker_number = worker_number - 1
    # cancel the number of the first line
    comm = None
    return comm, int(process_id), int(worker_number-1)


def FedML_FedAvg_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, model_trainer=None, preprocessed_sampling_lists=None):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                    model_trainer, preprocessed_sampling_lists)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict, test_data_local_dict, model_trainer)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, model_trainer, preprocessed_sampling_lists=None):
    if model_trainer is None:
        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model)
        else: # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                  worker_num, device, args, model_trainer)

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None :
        server_manager = FedAVGServerManager(args, aggregator, comm, rank, size, backend)
    else:
        server_manager = FedAVGServerManager(args, aggregator, comm, rank, size, backend,
            is_preprocessed=True, 
            preprocessed_client_lists=preprocessed_sampling_lists)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict,
                train_data_local_dict, test_data_local_dict, model_trainer=None):
    client_index = process_id - 1
    if model_trainer is None:
        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model)
        else: # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedAVGTrainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                            train_data_num, device, args, model_trainer)
    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size,backend)
    client_manager.run()
