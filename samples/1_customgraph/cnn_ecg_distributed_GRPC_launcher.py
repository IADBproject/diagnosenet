"""
Medical Care Purpose Classification for PMSI-ICU Dataset
User example for training DiagnoseNET on distributed platform
"""

## Makes diagnosenet library visible in samples folder
import sys
sys.path.append('../../')

from diagnosenet.resourcemanager import ResourceManager
import time

execution_start = time.time()

## Setting distributed training with resourcemanager:
distributed_training = ResourceManager()

if sys.argv[1] == "astro":
    distributed_training.between_graph_replication(device_replica_path="/home/mpiuser/cloud/0/diagnosenet/samples/1_customgraph/",
                                    device_replica_name="cnn_ecg_distributed_GRPC_replica.py",
#                                    ip_ps="astro0", ip_workers=",".join(["astro{}".format(i) for i in range(1, 5)]),
                                    ip_ps="134.59.132.161", ip_workers="134.59.132.166,134.59.132.168,134.59.132.170,134.59.132.173",
                                    num_ps=1, num_workers=4)
else:
    distributed_training.between_graph_replication(device_replica_path="/home/mpiuser/cloud/diagnosenet/samples/1_customgraph/",
                                    device_replica_name="cnn_ecg_distributed_GRPC_replica.py",
                                    ip_ps="134.59.132.135", ip_workers="134.59.132.20,134.59.132.21,134.59.132.23,134.59.132.26",
                                    num_ps=1, num_workers=4)


print("Resource manager execution time: {}".format((time.time()-execution_start)))
