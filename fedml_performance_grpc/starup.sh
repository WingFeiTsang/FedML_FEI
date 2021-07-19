sh run_fedavg_performance_grpc.sh cnn femnist "./../data/FederatedEMNIST/datasets" hetero 0.5 2 2 10 adam GRPC 0.1 0.01 10 2 1 2 2 grpc_ipconfig.csv 0
ssh 192.168.232.129
cd fedml/fedml_performance_grpc/
sh run_fedavg_performance_grpc.sh cnn femnist "./../data/FederatedEMNIST/datasets" hetero 0.5 2 2 10 adam GRPC 0.1 0.01 10 2 1 2 2 grpc_ipconfig.csv 0
exit
ssh 192.168.232.130
cd fedml/fedml_performance_grpc/
sh run_fedavg_performance_grpc.sh cnn femnist "./../data/FederatedEMNIST/datasets" hetero 0.5 2 2 10 adam GRPC 0.1 0.01 10 2 1 2 2 grpc_ipconfig.csv 0
exit
