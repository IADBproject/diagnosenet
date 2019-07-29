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

distributed_training.between_graph_replication(device_replica_path="/home/mpiuser/cloud/0/diagnosenet/samples/0_sequentialgraph/",
                                    device_replica_name="mlp_pmsi_distributed_GRPC_replica.py",
                                    ip_ps="134.59.132.185", ip_workers="134.59.132.190,134.59.132.192",	#134.59.132.23,134.59.132.26",
                                    num_ps=1, num_workers=2)

print("Resource manager execution time: {}".format((time.time()-execution_start)))
