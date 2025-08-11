import logging

from .drain3.template_miner import TemplateMiner
from .drain3.template_miner_config import TemplateMinerConfig
from .drain3.persistence_handler import PersistenceHandler

def Drain_Init(init_file: str) -> TemplateMiner:
    """
    Create drain3 model
    
    This function will init a drain cluster model. This  model will be used for all drain log parsing progress.
    """
    ph = PersistenceHandler()
    drain3Config = TemplateMinerConfig()
    drain3Config.load(init_file)
    drain3Config.profiling_enabled = True

    tmp = TemplateMiner(config=drain3Config, persistence_handler=ph)
    return tmp


def TT_service_name_extrace_from_pod(pod_name: str):
    if "service" in pod_name:
        return pod_name.split("service")[0] + "service"
    else:
        return pod_name
    
def OB_service_name_extract_from_pod(pod_name: str):
    svc_info = pod_name.split("-")
    if len(svc_info) > 2:
        pod_name = svc_info[0]
        return pod_name
    else:
        if "frontend" in pod_name:
            return "frontend"
        return None
    
def logger_init(logger: logging.Logger):
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
def RemoveSignals(line: str):
    """
    Remove all signals, numbers and single alpha from log line

    This function will remove the meaningless signals and numbers from the str.

    Parameter:
    line: The string value which is need to remove signals.
    """
    remove_list = list("~`!@#$%^&*()-_=+[{]};:'\",<.>/?|\\0123456789")
    res = line
    for signal in remove_list:
        res = res.replace(signal, ' ')
    res_list = res.split()
    alpha = "abcdefghijklmnopqrstuvwxyz"
    alpha_upper = alpha.upper()
    alpha_lower = alpha.lower()
    alpha_list = list(alpha_upper + alpha_lower)
    for a in alpha_list:
        while a in res_list:
            res_list.remove(a)
    res = ' '.join(res_list)
    return res