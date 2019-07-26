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
from concurrent import futures
import platform

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
        #print("++ tf_ps: ",tf_ps, type(tf_ps))

        ## Build a tf_workers collection
        tf_workers = []
        [tf_workers.append(str(self.ip_workers[i]+":2222")) for i in range(self.num_workers)]
        #tf_workers=','.join(tf_workers)
        #print("++ tf_workers: ", tf_workers)

        ## A collection of tf_ps nodes
        return tf.train.ClusterSpec({"ps": tf_ps, "worker": tf_workers})


    def run_device_replica(self, user_name, host_name, device_replica, job_replica):
        """
        Subprocess by device job replica
        """
        ## If running on the array, force SSH to use mpiuser's SSH key
        if platform.node().startswith("astro"):
            print("Detected that we are running on the Astro array")
            sp.call(["ssh", "-i", "/home/{}/.ssh/id_rsa".format(user_name), str(user_name+"@"+host_name), "python3.6", device_replica, "{}".format(job_replica)])
        else:
            sp.call(["ssh", str(user_name+"@"+host_name), "python3.6", device_replica, "{}".format(job_replica)])


    async def queue_device_tasks(self, executor):
        """
        Asyncio create queue device tasks
        """

        ## Event loop run asynchronous tasks
        loop = asyncio.get_event_loop()

        ## Creating a queue executor tasks
        tasks = [loop.run_in_executor(executor, self.run_device_replica, "mpiuser", self.IP_HOSTS[i],
				 self.device_replica_, self.job_DEVICE_replicas[i]) for i in range(self.devices_num)]

        completed, peding = await asyncio.wait(tasks)


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

        ## Resource manager start
        training_start = time.time()

        ## Set processing_mode flat
        self.processing_mode = "distributed_processing"
        self.device_replica_ = str(device_replica_path + device_replica_name)

        ip_ps = ip_ps.split(",")
        ip_workers = ip_workers.split(",")
        self.IP_HOSTS = ip_ps + ip_workers

        #print("+++ new ip_ps: {} +++".format(ip_ps))
        #print("+++ new ip_worker: {} +++".format(ip_workers))

        ## Assigning the job role for a device replication
        self.job_DEVICE_replicas = []
        [self.job_DEVICE_replicas.append(["ps", i, ip_ps, ip_workers])for i in range(num_ps)]
        [self.job_DEVICE_replicas.append(["worker", i, ip_ps, ip_workers])for i in range(num_workers)]
        self.devices_num = len(self.job_DEVICE_replicas)

        #print("++ Job DEVICE replicas: ", self.job_DEVICE_replicas)
        #print("++ Devices: ", self.devices_num)
        #print("++ Hosts: ", self.IP_HOSTS)


        ## Execute a device replicate in a separate process
        executor = futures.ProcessPoolExecutor(max_workers=self.devices_num,)
        event_loop = asyncio.get_event_loop()
        event_loop.run_until_complete(self.queue_device_tasks(executor))
