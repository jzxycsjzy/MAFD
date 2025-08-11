import os 
import sys

import time
import logging
from tqdm import tqdm

from .utils.utils import TT_service_name_extrace_from_pod, OB_service_name_extract_from_pod, logger_init
from .data_preprocessing.TT_dataset import TT_dataset
from .data_preprocessing.OB_dataset import OB_dataset

from .framework.MAAD_pipeline import MAFDRCA
from .framework.decision_merge import decision_merger

import click

logger = logging.getLogger(__name__)

def load_dataset(dataset_name: str, dataset_path: str, fault_path: str, save_path: str) -> tuple:
    """
    
    """
    service_list = []

    if dataset_name == "nezha":
        if "OB" in dataset_path:
            logger.info(f"DataSet: Load dataset: {dataset_name}_OB")
            metrics_list = os.listdir(os.path.join(dataset_path, "metric"))
            for f_name in metrics_list:
                service_name = OB_service_name_extract_from_pod(f_name)
                if service_name != None:
                    service_list.append(service_name)
            OB_dataset(dataset_path, logger, save_path, fault_path)
            logger.info(f"DataSet: Load dataset: {dataset_name}_OB successful.")
        else:
            logger.info(f"DataSet: Load dataset: {dataset_name}_TT")
            metrics_list = os.listdir(os.path.join(dataset_path, "metric"))
            for f_name in metrics_list:
                service_name = TT_service_name_extrace_from_pod(f_name)
                service_list.append(service_name)
            TT_dataset(dataset_path, logger, save_path, fault_path)
            logger.info(f"DataSet: Load dataset: {dataset_name}_TT successful.")

    else:
        pass

@click.command()
@click.option('--dataset_name', default='nezha', help='Name of the dataset to load)')
@click.option('--dataset_path', default='/data/nfs02/rongyuan/Journal_MAAD/baselines/Nezha/rca_data/2023-01-29_TT', help='Path to the dataset directory')
@click.option('--fault_path', default='/data/nfs02/rongyuan/Journal_MAAD/baselines/Nezha/construct_data/root_cause_ts.json', help='Path to the fault data file')
@click.option('--save_path', default='/data/nfs02/rongyuan/Journal_MAAD/baselines/Extracted_dataset/TT/', help='Path to save the extracted dataset')
def main(dataset_name, dataset_path, fault_path, save_path):
    """
    Main function to load the dataset and run the MAFD RCA workflow.
    """
    logger_init(logger)
    load_dataset(dataset_name, dataset_path, fault_path, save_path)
    rca = MAFDRCA("./config/drain3.ini", "./model", logger=logger, mode="test")
    rca.workflow(save_path)
    
if __name__ == '__main__':
    main()
    