#!/usr/bin/env bash

MODEL=$1
DATASET=$2
DATA_DIR=$3
DISTRIBUTION=$4
PARTITION_ALPHA=$5
CLIENT_NUM=$6
WORKER_NUM=$7
BATCH_SIZE=$8
CLIENT_OPTIMIZER=$9
BACKEND=${10}
LR=${11}
WD=${12}
EPOCH=${13}
COMM_ROUND=${14}
FREQUENCY_OF_THE_TEST=${15}
GPU_SERVER_NUM=${16}
GPU_NUM_PER_SERVER=${17}
GRPC_IPCONFIG_PATH=${18}
CI=${19}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

PS_MPI_HOST="192.168.232.128:1,192.168.232.129:1,192.168.232.130:1"

#hostname > mpi_host_file

# mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg_performance.py \
  mpirun -np $PROCESS_NUM -host $PS_MPI_HOST python3 ./main_fedavg_performance.py \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --partition_alpha $PARTITION_ALPHA \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --batch_size $BATCH_SIZE \
  --client_optimizer $CLIENT_OPTIMIZER \
  --backend $BACKEND \
  --lr $LR \
  --wd $WD \
  --epochs $EPOCH \
  --comm_round $COMM_ROUND \
  --is_mobile 0 \
  --frequency_of_the_test $FREQUENCY_OF_THE_TEST \
  --gpu_server_num $GPU_SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --grpc_ipconfig_path $GRPC_IPCONFIG_PATH \
  --ci $CI

  #--gpu_mapping_file "gpu_mapping.yaml" \
  #--gpu_mapping_key "mapping_default" \