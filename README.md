# MAFD
MAFD is  a multi-agent-based fault diagnosis architecture that operates in a distributed manner. MAFD comprises a collection of agents that collaboratively identify and localize anomalous behaviors through decentralized coordination. Each agent is co-located with an individual service and employs lightweight machine learning techniques to diagnose localized faults. This process leverages the service's native traces, logs, metrics, local context, and information extracted from its parent span in the call chain.

# Environment
OS: Ubuntu 22.04
Python 3.7

Using the following command to install necessary python packages.
```bash
conda create -m mafd python=3.7
conda activate mafd pip install requirementes.txt
```
We provide the functions in [preparation](./prepare_model.py) to illustrate how to prepare Drain3 model and GloVe models. Please following these functions to prepare Drain3 and GloVe model based on the dataset that will be used.

# Execution Guidance
We provide an example script to run an example of MAFD. Note that please prepare the dataset from [``Nezha``](https://github.com/IntelligentDDS/Nezha) before execute the script. In our example, ``Nezha`` is under `/data/baselines/Nezha`.

```bash
cd ..
python -B -m MAFD.main \
    --dataset_name nezha \
    --dataset_path /data/baselines/Nezha/rca_data/2023-01-29_TT \
    --fault_path /data/baselines/Nezha/construct_data/root_cause_ts.json \
    --save_path /data/baselines/Extracted_dataset/TT/ 
```