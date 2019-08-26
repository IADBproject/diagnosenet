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
    distributed_training.between_graph_replication(device_replica_path="/home/mpiuser/cloud/0/diagnosenet/samples/0_sequentialgraph/",
                                    device_replica_name="mlp_pmsi_distributed_GRPC_replica.py",
                                    ip_ps="astro0", ip_workers=",".join(["astro{}".format(i) for i in range(1, 3)]),
                                    num_ps=1, num_workers=2)
else:
    distributed_training.between_graph_replication(device_replica_path="/home/mpiuser/cloud/diagnosenet/samples/0_sequentialgraph/",
                                    device_replica_name="mlp_pmsi_distributed_GRPC_replica.py",
                                    ip_ps="134.59.132.135", ip_workers="134.59.132.20,134.59.132.21,134.59.132.23,134.59.132.26,134.59.132.116,134.59.132.126",
                                    num_ps=1, num_workers=6)

print("Resource manager execution time: {}".format((time.time()-execution_start)))
