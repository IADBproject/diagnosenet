"""
Resource Manager
"""

import tensorflow as tf
from diagnosenet.datamanager import Dataset, Batching
from diagnosenet.io_functions import IO_Functions
from diagnosenet.monitor import enerGyPU, Metrics
from sklearn.metrics import f1_score

import os, time
import asyncio



import subprocess as sp
import psutil, datetime, os



class ResourceManager():
    """
    Create an orchestation for the DNN workflow according with the processing_mode
        + diagnosenet_datamining
        + diagnosenet_unsupervisedembedding
        + diagnosenet_supervisedlearning
    """

    def __init__(self) -> None:
		#, model, monitor: enerGyPU = None, datamanager: Dataset = None,
                #    max_epochs: int = 10, min_loss: float = 2.0,
                #    ip_ps: str = "localhost:2222",
                #    ip_workers: str = "localhost:2223") -> None:

        # super().__init__(model, monitor, datamanager, max_epochs, min_loss, ip_ps, ip_workers)
        #self.model = model
        #self.monitor = monitor
        #self.data = datamanager
        #self.max_epochs = max_epochs
        #self.min_loss = min_loss

        ## Distributed Setup
        #self.tf_cluster = self.set_tf_cluster(ip_workers, ip_ps)
        self.ip_ps = 0	#ip_ps
        self.ip_workers = 0	#ip_workers
        self.num_ps = 0
        self.num_workers = 0

        ## Testbed and Metrics
        self.processing_mode: str

    def set_tf_cluster(self) -> tf.Tensor:
        ## splitting the IP hosts
        self.ip_ps = self.ip_ps.split(",")
        self.ip_workers = self.ip_workers.split(",")

        ## Build a tf_ps collection
        tf_ps = []
        #[tf_ps.append(str(ip_ps[i]+":2222")) for i in range(self.num_ps)]
        [tf_ps.append(str(self.ip_ps[i]+":2222")) for i in range(self.num_ps)]        
        #tf_ps=','.join(tf_ps)
        print("++ tf_ps: ",tf_ps, type(tf_ps))

        ## Build a tf_workers collection
        tf_workers = []
        [tf_workers.append(str(self.ip_workers[i]+":2222")) for i in range(self.num_workers)]
        #tf_workers=','.join(tf_workers)
        print("++ tf_workers: ", tf_workers)

        ## A collection of tf_ps nodes
        return tf.train.ClusterSpec({"ps": tf_ps, "worker": tf_workers})


    def between_graph_replication(self, device_replica_path: str, device_replica_name: str,
                                        ip_ps: str = "localhost:2222",
                                        ip_workers: str = "localhost:2223",
                                        num_ps: int = 1,
                                        num_workers: int = 1) -> None:
        """
        Training the deep neural network exploit the memory on desktop machine
        https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md
        """
        ### Training Start
        training_start = time.time()

        ## Set processing_mode flat
        self.processing_mode = "distributed_processing"
        self.device_replica_path = device_replica_path
        self.device_replica_name = device_replica_name
        self.ip_ps = ip_ps
        self.ip_workers = ip_workers
        self.num_ps = num_ps
        self.num_workers = num_workers
        self.tf_cluster = self.set_tf_cluster()

        #print("cluster: {}".format(self.tf_cluster))
        #print("ps num: {} || workers num: {}".format(num_ps, num_workers))
        #print("ps: {} || worker: {}".format(self.ip_ps, self.ip_workers))


        ###################################################################
        ### Define role for distributed processing
        print("++ Issue: Define role for distributed processing ++")

        # "model": self.model,
        # "monitor": self.monitor,
        # "datamanager": self.datamanager,
        # "max_epochs": self.max_epochs,
        # "min_loss": self.min_loss,

        ##############################
        #Building ps job replicas
        job_PS_replicas = []
#        [job_PS_replicas.append({"node": self.ip_ps[i],
#                            #"tf_cluster": self.tf_cluster,
#                            "job_name": "ps",
#                            "task_index": i
#                            }
#                            )for i in range(num_ps)]

        [job_PS_replicas.append([self.ip_ps, self.ip_workers,  "ps", i]
                                                        )for i in range(num_ps)]




        job_WORKER_replicas = []
        [job_WORKER_replicas.append([self.ip_ps, self.ip_workers, "worker", i]
                                                        )for i in range(num_workers)]                            


#        [job_WORKER_replicas.append({ #"node": self.ip_workers[i],
#                            #"tf_cluster": self.tf_cluster,
#                            "job_name": "worker",
#                            "task_index": i}
#                            )for i in range(num_workers)]

        print("++ Job PS replicas: ", job_PS_replicas)
        print("++ Job WORKERS replicas: ", job_WORKER_replicas)



        print("----> sp run:")
        device_replica_ = str(self.device_replica_path + self.device_replica_name)

        print("----> Dv_replica: {}".format(device_replica_))
        sp.call(["ssh", "mpiuser@134.59.132.135", "python3.6", device_replica_, "{}".format(job_PS_replicas[0])])


        sp.call(["ssh", "mpiuser@134.59.132.20", "python3.6", device_replica_, "{}".format(job_WORKER_replicas[0])])

        #print(a.srdout.decode('utf-8')


# stdin=sp.PIPE,  stdout = sp.PIPE, universal_newlines=True, bufsize=0)
        #print("worker_prog: {}".format(worker_prog))


