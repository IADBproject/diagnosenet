"""
Energy-monitoring launcher for workload characterizatio
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
        self.gpus: list

    def _get_available_GPU(self) -> list:
        """
        Returns a list of ID GPUs available in the computational platform
        """

        ## Identification of GPUs
        idgpu = []
        cmd_nvidia_smi = "nvidia-smi | grep 0000 | awk '{print $8}'"
        idgpu.append(sp.Popen(cmd_nvidia_smi, stdout=sp.PIPE, shell=True).stdout.readlines())

        ## Get ID GPUs available
        if not idgpu:
            print("++ no NVIDIA GPU detected ++")
        else:
            self.gpus = [g.decode("utf-8").rstrip('\n') for g in idgpu[0] if '0000' in g.decode("utf-8")]

        return self.gpus

    def start_power_recording(self, testbed_path, exp_id) -> None:
        """
        Launches a subprocess for recording the global GPU factors
        to power consumption measures.
        """
        # testbed_Dir="enerGyPU/testbed/"
        # argv="diagNET"
        # pwd = sp.check_output(["ls", "enerGyPU/dataCapture/enerGyPU_record.sh"])

        # sp.call(["enerGyPU/dataCapture/enerGyPU_record.sh", testbed_Dir, argv])
        sp.run(["enerGyPU/dataCapture/enerGyPU_record.sh", testbed_path, exp_id])

    def end_power_recording(self) -> None:
        """
        Kill the subprocess enerGyPU_record.
        """
        sp.call(["killall", "-9", "nvidia-smi"])


## enerGyPU

# gpus = enerGyPU()._get_available_GPU()
# print("idgpu: {}".format(gpus))
#
# enerGyPU().start_power_recording()
# print("Next task")
#
# enerGyPU().end_power_recording()
