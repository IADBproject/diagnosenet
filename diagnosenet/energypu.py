"""
Energy-monitoring launcher for workload characterization
"""

import subprocess as sp
import psutil, datetime
import numpy as np

from diagnosenet.metrics import Testbed, Metrics


class enerGyPU(Testbed):
    """
    This module deploys an energy monitor to collect the energy consumption metrics
    while the DNN model is executed on the target platform.
    """
    def __init__(self, model, data, platform_name, max_epochs) -> None:
        super().__init__(model, data, platform_name, max_epochs)
        self.idgpu_available: list = []
        self.resources_metrics: list = []
        self.recording_stop: bool = False

    def _get_available_GPU(self) -> list:
        """
        Returns a list of ID GPUs available in the computational platform
        """
        ## Identification of GPUs
        idgpu = []
        cmd_nvidia_smi = "nvidia-smi | grep '0000' | awk '{if($7 ~ '0000') print $7; else if($8 ~ '0000') print $8}'"
        idgpu.append(sp.Popen(cmd_nvidia_smi, stdout=sp.PIPE, shell=True).stdout.readlines())

        ## Get ID GPUs available
        if not idgpu:
            print("++ no NVIDIA GPU detected ++")
        else:
            for i in range(len(idgpu)):
                ## Convert byte sp.Popen output to string
                gpu_tmp = str(idgpu[0][i].decode("utf-8").rstrip('\n'))
                ## Uses the "nvidia-smi pmon" command-line for monitoring the availeble GPU
                cmd_nvidia_pmon = "nvidia-smi pmon -i "+gpu_tmp+" -c 1 | awk '{if(NR == 3) print $1}'"
                tmp_idgpu = sp.Popen(cmd_nvidia_pmon, stdout=sp.PIPE, shell=True).stdout.readline()
                self.idgpu_available.append(tmp_idgpu.decode("utf-8").rstrip('\n'))

        return self.idgpu_available

    def start_power_recording(self, testbed_path, exp_id) -> None:
        """
        Launches a subprocess for recording the global GPU factors
        to power consumption measures.
        """
        sp.run(["enerGyPU/dataCapture/enerGyPU_record.sh", testbed_path, exp_id])

    def end_power_recording(self) -> None:
        """
        Kill the subprocess enerGyPU_record.
        """
        sp.call(["killall", "-9", "nvidia-smi"])

    def start_platform_recording(self, pid) -> None:
        """
        Subprocess recording for memory and cpu usage while the models are training
        This function uses the library psutil-5.4.8
        """

        p = psutil.Process(int(pid))
        resources_metrics: list = []

        while True:
            core_usage = p.cpu_percent(interval=1.0)
            num_threads = p.num_threads()
            memory_usage = np.round(p.memory_percent(), 2)
            memory_rss = str(p.memory_info().rss / 1024)
            memory_vms = str(p.memory_info().vms / 1024)
            memory_shr = str(p.memory_info().shared / 1024)
            data = str(p.memory_info().data)
            io_reads = str(p.io_counters().read_bytes / 1024)
            io_writes = str(p.io_counters().write_bytes / 1024)
            time = str(datetime.datetime.now().time())

            self.resources_metrics.append((time, memory_usage, memory_rss, memory_vms, memory_shr,
                            core_usage, num_threads, data, io_reads, io_writes))

            if self.recording_stop == True:
                break

    def end_platform_recording(self) -> list:
        """
        Send a signal to stop the recording process
        """
        self.recording_stop = True
        return self.resources_metrics
