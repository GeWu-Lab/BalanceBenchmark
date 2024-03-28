from balancemm.utils.logger import setup_logger
from lightning import fabric
import yaml

def only_test(args: dict):
    sys_logger = setup_logger(args['out_dir'] + "/test.log")