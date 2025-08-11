import os 
import sys
sys.path.append("/data/nfs/rongyuan/Journal_MAAD")

from .utils.drain3.template_miner import TemplateMiner
from .utils.drain3.template_miner_config import TemplateMinerConfig
from .utils.drain3.persistence_handler import PersistenceHandler

from .utils.utils import Drain_Init, RemoveSignals
from tqdm import tqdm
import pandas as pd

import time


def drain_mining():
    tmp = Drain_Init("./config/drain3.ini")
    data_path = "../baselines/Extracted_dataset/TT/"
    
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        print(f"Start to analyze {folder_path}")
        trace_files = os.listdir(folder_path)
        for i in tqdm(range(len(trace_files))):
            trace_file = trace_files[i]
            trace_name = os.path.join(folder_path, trace_file)
            pd_data = pd.read_csv(trace_name, keep_default_na=False)
            for j in range(pd_data.shape[0]):
                tmp.add_log_message(pd_data.loc[j]['OperationName'].rstrip())
                if pd_data.loc[j]['logs'] != "":
                    logs = pd_data.loc[j]['logs'].split("%")
                    for log in logs:
                        tmp.add_log_message(log.rstrip())
        tmp.save_state("tt")
        
def GloveCorpusConstruction():
    """
    Generate world vector from templates
    """
    save_dir = "./glove/corpus"
    tmp = Drain_Init("./config/drain3.ini")
    tmp.load_state("tt")
    save_file = open(save_dir, 'a+')
    for cluster in tmp.drain.clusters:
        template = cluster.get_template()
        save_file.write(RemoveSignals(template))
    save_file.close()        
    
def get_svcs():
    data_path = "/data/nfs/rongyuan/Journal_MAAD/baselines/Nezha/rca_data/2023-01-29_TT/metric"
    svcs = os.listdir(data_path)
    svcs.remove("front_service.csv")
    for i in range(len(svcs)):
        svc = svcs[i].split("service")[0] + "service"
        svcs[i] = svc
    print(svcs)