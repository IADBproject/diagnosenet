"""
Energy-monitoring launcher for workload characterization
"""

import subprocess as sp

from diagnosenet.metrics import Testbed, Metrics


class enerGyPU(Testbed):
    """
    This module deploys an energy monitor to collect the energy consumption metrics
    while the DNN model is executed on the target platform.
    """
    def __init__(self, model, data, platform_name, max_epochs) -> None:
        super().__init__(model, data, platform_name, max_epochs)
        self.idgpu_available: list = []

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
