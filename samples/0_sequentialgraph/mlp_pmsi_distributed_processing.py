"""
Medical Care Purpose Classification for PMSI-ICU Dataset
User example for training DiagnoseNET on distributed platform
"""

import time
execution_start = time.time()

## Makes diagnosenet library visible in samples folder
import sys
sys.path.append('../../')


from diagnosenet.resourcemanager import ResourceManager



## Setting distributed training with resourcemanager:
distributed_training = ResourceManager()

distributed_training.between_graph_replication(device_replica_path="/home/mpiuser/cloud/diagnosenet/samples/0_sequentialgraph/",
                                    device_replica_name="mlp_pmsi_device_replica.py",
                                    ip_ps="134.59.132.135", ip_workers="134.59.132.20",
                                    num_ps=1, num_workers=1)



print("Execution Time: {}".format((time.time()-execution_start)))
