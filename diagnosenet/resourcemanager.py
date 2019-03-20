#################################################################################
#   Towards a Green Intelligence Applied to Health Care and Well-Being          #
###-----------------------------------------------------------------------------#
# diagnosenet_resourcemanager.py                                                #
# Setting the computaional resources for training DNN                           #
# Create an orchestration scheduler                                             #
#                                                                               #
#################################################################################

from __future__ import print_function
import os, sys

# from asynciojobs import Scheduler
import asyncio

from asynciojobs import Scheduler, Sequence
from apssh import SshNode, SshJob, LocalNode
from apssh import Run, RunString, RunScript, Push
# from argparse import ArgumentParser

class ResourceManager(object):
    """
    Setting the computaional resources for training:
        + Pascal Jetson-TX2 Cluster
        + Octopus multiGPU
    Create an orchestation for the DNN workflow:
        + diagnosenet_datamining
        + diagnosenet_unsupervisedembedding
        + diagnosenet_supervisedlearning
    """

    def __init__(self, *args, **kwards):
        self.platform = args[0]
        self.gateway_user = args[1]
        self.node_user = args[2]

    def _set_Node(host, user, gateway):
        return SshNode(host, username=user, gateway=gateway,)

    def _set_Job(node, cmd, sched):
        return SshJob(node=node, command=cmd, scheduler=sched,)


def main(argv):
    if len(argv) == 3:
        print("!! Unfinished routines !!")
    else:
        print("++ Using default settings ++")
        ###########################
        ## Local Variables
        # platform='multiGPU'
        platform='distributed'
        gateway_user='gw_user'
		gateway_host='gw_host'
        node_username='node_user'

        #########################################################
        ## Distributed Requirements
        num_ps = 1
        num_workers = 2

        #########################################################
        gateway = SshNode(
                        gateway_host,
                        username=gateway_user
                        )

		##########################################################
        elif platform == 'distributed':
            
            ## Jetson-TX2 Cluster
            hosts = [cluster_ip_host]

            #########################################################
            ## Use the Server node for processing the first satge Data-mining
            server = ResourceManager._set_Node(master_host, master_user, gateway,)

            ############################
            # Push the launch file (run_splitpoint)
            # With the Parameters Connfiguration on the server
            # To execute the First Satege in this host
            job_launch_S1 = SshJob(
                        node = server,
                        commands = [
                                ## Run the script locate in the laptop
                                RunScript("run_dataspworkers_mlp.sh", platform, num_ps, num_workers),
                                Run("echo Split Data DONE"),
                                ],
                        )

            #############################
            ## A collection of the PS node
            ps = []
            [ps.append(ResourceManager._set_Node(hosts[i],
                                                node_username, gateway,))
                                                for i in range(num_ps)]

            #############################
            ## A collection of the workers node
            workers = []
            [workers.append(ResourceManager._set_Node(hosts[num_ps+i],
                                                    node_username, gateway,))
                                                    for i in range(num_workers)]

            #########################################################
            ## Setting Parameters for the First Stage
            FEATURES_NAME = "FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1"
            SANDBOX=str("/data_B/datasets/drg-PACA/healthData/sandbox-"+FEATURES_NAME)
            YEAR=str(2008)

            ## Stage 1
            # localdir = "/1_Mining-Stage/"
            # SP_Dir_X = str(SANDBOX+localdir+"BPPR-"+FEATURES_NAME+"-"YEAR)

            #############################
            ## Setting parameters for the Second Stage
            S_PLOINT = str(3072)    #1536)
            #SP_ARGV = str(S_PLOINT+"-"+platform)
            SP_ARGV = platform+"-"+str(num_workers)
            SP2=str(SANDBOX+"/2_Split-Point-"+SP_ARGV+"/")

            #############################
            ## BPPR Directories
            dir_train = "/data_training/"
            dir_valid = "/data_valid/"
            dir_test = "/data_test/"

            ############################
            ## Worker data management
            worker_healthData = "/opt/diagnosenet/healthData/"
            worker_sandbox = str(worker_healthData+"/sandbox-"+FEATURES_NAME)
            worker_splitpoint = str(worker_sandbox+"/2_Split-Point-"+SP_ARGV+"/")
            worker_train = str(worker_splitpoint+dir_train)
            worker_valid = str(worker_splitpoint+dir_valid)
            worker_test = str(worker_splitpoint+dir_test)

            ############################
            ## Worker commands
            mkd_worker_sandbox = str("mkdir"+" "+worker_sandbox)
            mkd_worker_splitpoint = str("mkdir"+" "+worker_splitpoint)
            mkd_worker_train = str("mkdir"+" "+worker_train)
            mkd_worker_valid = str("mkdir"+" "+worker_valid)
            mkd_worker_test = str("mkdir"+" "+worker_test)

            #############################
            ## Create a JOB to build the sandbox for each Worker
            job_build_sandbox = []

            [ job_build_sandbox.append(SshJob(
                            node = workers[i],
                            commands = [
                                RunString(mkd_worker_sandbox),
                                RunString(mkd_worker_splitpoint),
                                RunString(mkd_worker_train),
                                RunString(mkd_worker_valid),
                                RunString(mkd_worker_test),
                                Run("echo SANDBOX ON WORKER DONE"), ],
                                )) for i in range(len(workers)) ]


            #############################
            ## Create a command for transfer data
            scp = "scp"
            cmd_X_train_transfer = []
            cmd_y_train_transfer = []
            cmd_X_valid_transfer = []
            cmd_y_valid_transfer = []
            cmd_X_test_transfer = []
            cmd_y_test_transfer = []

            for i in range(num_workers):
                worker_host = str(node_user+"@"+ hosts[num_ps+i] +":")
                num_file = str(i+1)
                ## Commands to transfer Training dataset
                X_train_splitted = str(SP2+dir_train+"X_training-"+FEATURES_NAME+"-"+YEAR+"-"+num_file+".txt")
                cmd_X_train_transfer.append(str(scp+" "+X_train_splitted+" "+worker_host+worker_train))
                y_train_splitted = str(SP2+dir_train+"y_training-"+FEATURES_NAME+"-"+YEAR+"-"+num_file+".txt")
                cmd_y_train_transfer.append(str(scp+" "+y_train_splitted+" "+worker_host+worker_train))

                ## Commands to transfer Validation dataset
                X_valid_splitted = str(SP2+dir_valid+"X_valid-"+FEATURES_NAME+"-"+YEAR+"-"+num_file+".txt")
                cmd_X_valid_transfer.append(str(scp+" "+X_valid_splitted+" "+worker_host+worker_valid))
                y_valid_splitted = str(SP2+dir_valid+"y_valid-"+FEATURES_NAME+"-"+YEAR+"-"+num_file+".txt")
                cmd_y_valid_transfer.append(str(scp+" "+y_valid_splitted+" "+worker_host+worker_valid))

                ## Commands to transfer Test dataset
                X_test_splitted = str(SP2+dir_test+"X_test-"+FEATURES_NAME+"-"+YEAR+"-"+num_file+".txt")
                cmd_X_test_transfer.append(str(scp+" "+X_test_splitted+" "+worker_host+worker_test))
                y_test_splitted = str(SP2+dir_test+"y_test-"+FEATURES_NAME+"-"+YEAR+"-"+num_file+".txt")
                cmd_y_test_transfer.append(str(scp+" "+y_test_splitted+" "+worker_host+worker_test))


            ############################
            ## Build a JOB for transfering data to each worker sandbox
            job_data_transfer = []
            [job_data_transfer.append(SshJob(
                        node = server,
                        commands = [
                                    RunString(cmd_X_train_transfer[i]),
                                    RunString(cmd_y_train_transfer[i]),
                                    Run("echo SENDER TRAINING DATA DONE"),
                                    RunString(cmd_X_valid_transfer[i]),
                                    RunString(cmd_y_valid_transfer[i]),
                                    Run("echo SENDER VALID DATA DONE"),
                                    RunString(cmd_X_test_transfer[i]),
                                    RunString(cmd_y_test_transfer[i]),
                                    Run("echo SENDER TEST DATA DONE"),
                                    ],)
                                    ) for i in range(len(workers))]

            #########################################################
            ## Create a sequence orchestration scheduler instance upfront
            worker_seq = []

            ## Add the Stage-1 JOB into Scheduler
            worker_seq.append(Scheduler(Sequence(
                                job_launch_S1)))

            ## Add the worker JOBs into Scheduler
            [worker_seq.append(Scheduler(Sequence(
                                job_build_sandbox[i],
                                job_data_transfer[i], ))
                                ) for i in range(len(workers))]

            #############################
            ## Old method
            ## Add the JOB PS Replicas into Scheduler
            # worker_seq.append(Scheduler(Sequence(
            #                     job_PS_replicas)))
            #
            # ## Add the JOB WORKER Replicas into Scheduler
            # worker_seq.append(Scheduler(Sequence(
            #                     job_WORKER_replicas)))


            #############################
            ## Run the Sequence JOBS
            # [seq.orchestrate() for seq in worker_seq]


            #########################################################
            #########################################################
            ## Push the launch file (run_secondstage_distributed)
            ## With the Distributed Parameters for each worker replicas
            ## To distributed training of Unsupervised Embedding

            #############################
            ## Build a collection of TensorFlow Hosts for PS
            tf_ps = []
            [tf_ps.append(str(hosts[i]+":2222")) for i in range(num_ps)]
            # print("+++ tf_ps: {}".format(tf_ps))
            tf_ps=','.join(tf_ps)

            #############################
            ## Build a collection of TensorFlow Hosts for workers
            tf_workers = []
            [tf_workers.append(str(hosts[num_ps+i]+":2222")) for i in range(num_workers)]
            # print("+++ tf_workers: {}".format(tf_workers))
            tf_workers=','.join(tf_workers)

            job_PS_replicas = []
            [job_PS_replicas.append(SshJob(
                        node = ps[i],
                        commands = [
                                ## Launches local script to execute on cluster
                                # RunScript("run_secondstage_distributed.sh",
                                #             platform, tf_ps, tf_workers,
                                #             num_ps, num_workers, "ps", i),
                                RunScript("run_thirdstage_distributed_mlp.sh",
                                            platform, tf_ps, tf_workers,
                                            num_ps, num_workers, "ps", i),
                                Run("echo PS REPLICA DONE"),
                                ],)
                                ) for i in range(len(ps))]


            job_WORKER_replicas = []
            [job_WORKER_replicas.append(SshJob(
                        node = workers[i],
                        commands = [
                                ## Launches local script to execute on cluster
                                # RunScript("run_secondstage_distributed.sh",
                                #             platform, tf_ps, tf_workers,
                                #             num_ps, num_workers, "worker", i),
                                RunScript("run_thirdstage_distributed_mlp.sh",
                                            platform, tf_ps, tf_workers,
                                            num_ps, num_workers, "worker", i),
                                Run("echo WORKER REPLICA DONE"),
                                ], )
                                ) for i in range(len(workers))]

            #############################
            ### Simultaneous jobs
            s_distraining = Scheduler()
            [s_distraining.add(job_PS_replicas[i]) for i in range(len(ps))]
            [s_distraining.add(job_WORKER_replicas[i]) for i in range(len(workers))]

            s_distraining.run(jobs_window = int(num_ps+num_workers+1))

            ## run the scheduler
            # ok = scheduler.orchestrate()

            ## give details if it failed
            # ok or scheduler.debrief()

            ## return something useful to your OS
            # exit(0 if ok else 1)

        else:
            print("!! Platform don't selected !!")


if __name__ == '__main__':
    main(sys.argv[1:])
