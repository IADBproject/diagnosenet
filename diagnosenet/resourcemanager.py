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
        pass


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
        The training configuration, which involves multiple task in a `worker`,
        training the same model on different mini-batches of data, updating shared parameters hosted 
        in one or more task in a parameter server job.
        https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md
        """

        ### Training Start
        training_start = time.time()

        ## Set processing_mode flat
        self.processing_mode = "distributed_processing"
        self.device_replica_path = device_replica_path
        self.device_replica_name = device_replica_name
        self.ip_ps = ip_ps.split(",")
        self.ip_workers = ip_workers.split(",")
        self.num_ps = num_ps
        self.num_workers = num_workers
        #self.tf_cluster = self.set_tf_cluster()


        ###################################################################
        ### Define role for distributed processing

        ##############################
        #Building ps job replicas
        job_PS_replicas = []
        [job_PS_replicas.append([ip_ps, ip_workers, "ps", i]
                                                        )for i in range(num_ps)]


        job_WORKER_replicas = []
        [job_WORKER_replicas.append([ip_ps, ip_workers, "worker", i]
                                                        )for i in range(num_workers)]                            


        print("++ Job PS replicas: ", job_PS_replicas)
        print("++ Job WORKERS replicas: ", job_WORKER_replicas)


        ##############################
        # Device Replication
        print("Issue Launch the device replica with asyncio")
        device_replica_ = str(self.device_replica_path + self.device_replica_name)
        for i in range(num_ps):
            print("----> PS_IP: {}".format(self.ip_ps[i]))
            sp.call(["ssh", str("mpiuser@"+self.ip_ps[i]), "python3.6", device_replica_, "{}".format(job_PS_replicas[0])])

        for i in range(num_workers):
            print("----> WORKERS_IP: {}".format(self.ip_workers[i]))
            sp.call(["ssh", str("mpiuser@"+self.ip_workers[i]), "python3.6", device_replica_, "{}".format(job_WORKER_replicas[0])])


