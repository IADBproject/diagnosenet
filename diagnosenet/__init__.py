"""
This module will add the public DiagnoseNET interface
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from diagnosenet.loggingconf import Config

#######################################################################
### Set general logging configuration for using in _dIAgnoseNET_DataMining
logging_config = Config()
logging_config._setup_logger('_DiagnoseNET_',
            str('.diagnosenet.log'), logging.DEBUG)
