import pandas as pd
import numpy as np
import os
import logging
import copy
import json
from datetime import datetime

from tqdm import tqdm

from ..utils.utils import RemoveSignals

class TT_dataset:
    def __init__(self, data_path: str, logger: logging.Logger, save_path, fault_path: str):
        self.logger = logger
        self.data_path = data_path

        self.trace_data = {}
        
        self.save_path = save_path
        self.fault_dict = self._get_fault_dict(fault_path)
        self.fault_idxs = list(self.fault_dict.keys())
        self.processed_files = [
        ]
        self._load_all()
        
    def _get_fault_dict(self, fault_path: str) -> dict:
        with open(fault_path, 'r') as f:
            fault_dict = json.load(f)
        return fault_dict
    
    def _get_fault_type(self, inject_pod: str):
        inject_service = inject_pod.split("service")[0] + "service"
        return inject_service
    
    def _get_metric_data(self, metric_path) -> dict:
        """
        Obtain metric data of the current sub-dataset
        """
        metric_data = {}
        for metric_file in os.listdir(metric_path):
            if "front" in metric_file:
                continue
            else:
                svc_name = metric_file.split("service")[0] + "service"
                metric_pd = pd.read_csv(os.path.join(metric_path, metric_file))
                metric_data[svc_name] = metric_pd[['TimeStamp', 'CpuUsageRate(%)', 'PodClientLatencyP90(s)', 'PodServerLatencyP90(s)']]
        return metric_data
    
    def _get_anomaly_time(self, inject_time: str):
        """
        Reference from Nazha
        """
        min = int(inject_time.split(":")[1]) + 2

        if min >= 60:
            hour_min = inject_time.split(" ")[1]
            hour = int(hour_min.split(":")[0])
            if hour < 9:
                abnormal_time = inject_time.split(
                    " ")[0] + " 0" + str(hour+1) + ":0" + str(min-60)
            else:
                abnormal_time = inject_time.split(
                    " ")[0] + " " + str(hour+1) + ":0" + str(min-60)
        elif min < 10:
            abnormal_time = inject_time.split(
                ":")[0] + ":0" + str(min)
        else:
            abnormal_time = inject_time.split(
                ":")[0] + ":" + str(min)
        abnormal_time = abnormal_time.split(" ")[1].replace(":")
        return abnormal_time
    
    def label_abnormal(self, pd_data: pd.DataFrame, err: str):
        err_svc = err.split("#")[0]
        err_time = float(err.split("#")[1])
        err_type = err.split("#")[2]
        
        true_err = "normal_normal"
        
        for i in range(pd_data.shape[0]):
            if pd_data.loc[i]['StartTime'] > err_time and err_svc in pd_data.loc[i]['PodName']:
                true_err = f"{err_svc}_{err_type}"
                
        err_folder = os.path.join(self.save_path, true_err)
        if not os.path.exists(err_folder):
            os.mkdir(err_folder)
        return err_folder
            
            
    def _load_all(self):
        log_path = os.path.join(self.data_path, "log")
        metric_path = os.path.join(self.data_path, "metric")
        trace_path = os.path.join(self.data_path, "trace")
        traceid_path = os.path.join(self.data_path, "traceid")
        fault_list_filename = os.path.basename(self.data_path).replace("_TT", "-fault_list.json")
        fault_path = os.path.join(self.data_path, fault_list_filename)
        all_traceid_file = sorted(os.listdir(traceid_path))
        
        metric_data = self._get_metric_data(metric_path)
        neccessary_metrics = ['CpuUsageRate(%)', 'PodClientLatencyP90(s)', 'PodServerLatencyP90(s)']
        
        abnormal_dict = {}
        idx = 0
        if os.path.exists(fault_path):
            with open(fault_path, 'r') as f:
                fault_list = json.load(f)
            for key in fault_list.keys():
                for i in range(len(fault_list[key])):
                    fault = fault_list[key][i]
                    inject_time = fault['inject_time']
                    inject_timestamp = datetime.strptime(inject_time, "%Y-%m-%d %H:%M:%S").timestamp()
                    fault_type = self._get_fault_type(fault["inject_pod"])
                    abnormal_dict[f"{fault_type}#{inject_timestamp}#{fault['inject_type']}"] = all_traceid_file[idx:idx+3]
                    idx += 3
    
        else:
            fault_list = {}
        print(abnormal_dict)
        
        # mark the most close index in metric data to reduce the overhead.
        if abnormal_dict == {}:
            for traceid_idx in tqdm(range(len(all_traceid_file))):
                traceid_filename = all_traceid_file[traceid_idx]
                traceid_filepath = os.path.join(traceid_path, traceid_filename)
                trace_filepath = os.path.join(trace_path, traceid_filename.replace("traceid", "trace"))
                log_filepath = os.path.join(log_path, traceid_filename.replace("traceid", "log"))
                # Have not contained metrics yet.
                
                traceids = pd.read_csv(traceid_filepath, header=None, names=['traceid'])
                traces = pd.read_csv(trace_filepath)
                logs = pd.read_csv(log_filepath)

                curr_trace_template = pd.DataFrame(columns=['SpanID', 'PodName', 'OperationName', 'StartTime', 'EndTime', 'ChildSpans', 'logs', 'CpuUsageRate(%)', 'PodClientLatencyP90(s)', 'PodServerLatencyP90(s)'])
                curr_traceid = ""
                curr_trace = None
                most_close_metric_idx = -1
                for i in range(traces.shape[0]):
                # for i in range(0, 100):
                    # print(traces.loc[i]['ParentID'])
                    if traces.loc[i]['ParentID'] == "root":
                        if curr_traceid != "":
                            self.trace_data[curr_traceid] = curr_trace
                        curr_traceid = traces.loc[i]['TraceID']
                        curr_trace = copy.deepcopy(curr_trace_template)
                    curr_spanid = traces.loc[i]['SpanID']
                    logs_df = logs[logs['SpanID'] == curr_spanid]
                    if logs_df.shape[0] > 0:
                        loglines = '#'.join([eval(log)['log'].rstrip() for log in logs_df['Log'].tolist()])
                    else:
                        loglines = ""
                        
                    start_time = traces.loc[i]['StartTimeUnixNano'] // 1000000000
                    svc_name = traces.loc[i]['PodName'].split("service")[0] + "service"
                    if most_close_metric_idx == -1:
                        metric_start_time = metric_data[svc_name].loc[0]['TimeStamp']
                        time_diff = start_time - metric_start_time
                        most_close_metric_idx = int(time_diff // 60)
                    # define metrics
                    metrics = []
                    # Find metrics
                    curr_time_diff = start_time - metric_data[svc_name].loc[most_close_metric_idx]['TimeStamp']
                    if abs(curr_time_diff) <= 30:
                        for metric in neccessary_metrics:
                            metrics.append(metric_data[svc_name].loc[most_close_metric_idx][metric])
                    else:
                        if curr_time_diff > 0:
                            index_change = 1
                        else:
                            index_change = -1
                        while True:
                            most_close_metric_idx += index_change
                            curr_time_diff = start_time - metric_data[svc_name].loc[most_close_metric_idx]['TimeStamp']
                            if abs(curr_time_diff) <= 30:
                                for metric in neccessary_metrics:
                                    metrics.append(metric_data[svc_name].loc[most_close_metric_idx][metric])
                                break
                    new_data = [curr_spanid, traces.loc[i]['PodName'], traces.loc[i]['OperationName'], traces.loc[i]['StartTimeUnixNano'] // 1000000, traces.loc[i]['EndTimeUnixNano'] // 1000000, "", loglines] + metrics
                    curr_trace.loc[-1] = new_data
                    curr_trace.index += 1
                    # Add childspan
                    parentID = traces.loc[i]['ParentID']
                    if parentID != "root":
                        curr_trace.loc[curr_trace[curr_trace['SpanID'] == parentID].index, 'ChildSpans'] += curr_spanid
                        # curr_trace.to_csv(os.path.join(self.save_path, f"{trace_id}.csv"), index=False)
            if "" in self.trace_data.keys():
                del self.trace_data[""]
            # for trace_id in self.trace_data.keys():
            #     self.trace_data[trace_id].to_csv(os.path.join(self.save_path, f"{trace_id}.csv"), index=False)
        else:
            # Process abnormal data
            for err in abnormal_dict.keys():
                for traceid_filename in abnormal_dict[err]:
                    if traceid_filename in self.processed_files:
                        continue
                    self.logger.info(f"Start extrace file: {traceid_filename}")
                    traceid_filepath = os.path.join(traceid_path, traceid_filename)
                    trace_filepath = os.path.join(trace_path, traceid_filename.replace("traceid", "trace"))
                    log_filepath = os.path.join(log_path, traceid_filename.replace("traceid", "log"))
                    
                    traceids = pd.read_csv(traceid_filepath, header=None, names=['traceid'])
                    traces = pd.read_csv(trace_filepath)
                    logs = pd.read_csv(log_filepath)

                    curr_trace_template = pd.DataFrame(columns=['SpanID', 'PodName', 'OperationName', 'StartTime', 'EndTime', 'ChildSpans', 'logs', 'CpuUsageRate(%)', 'PodClientLatencyP90(s)', 'PodServerLatencyP90(s)'])
                    curr_traceid = ""
                    curr_trace = None
                    most_close_metric_idx = -1
                    for i in range(traces.shape[0]):
                    # for i in range(0, 100):
                        # print(traces.loc[i]['ParentID'])
                        if traces.loc[i]['ParentID'] == "root":
                            if curr_traceid != "":
                                save_dir = self.label_abnormal(curr_trace, err)
                                # self.trace_data[curr_traceid + trace_label] = curr_trace
                                curr_trace.to_csv(os.path.join(save_dir, f"{curr_traceid}.csv"), index=False)
                            curr_traceid = traces.loc[i]['TraceID']
                            curr_trace = copy.deepcopy(curr_trace_template)
                        curr_spanid = traces.loc[i]['SpanID']
                        logs_df = logs[logs['SpanID'] == curr_spanid]
                        if logs_df.shape[0] > 0:
                            loglines = '%'.join([eval(log)['log'].rstrip() for log in logs_df['Log'].tolist()])
                        else:
                            loglines = ""
                        start_time = traces.loc[i]['StartTimeUnixNano'] // 1000000000
                        svc_name = traces.loc[i]['PodName'].split("service")[0] + "service"
                        if most_close_metric_idx == -1:
                            metric_start_time = metric_data[svc_name].loc[0]['TimeStamp']
                            time_diff = start_time - metric_start_time
                            most_close_metric_idx = int(time_diff // 60)
                        
                        # define metrics
                        metrics = []
                        # Find metrics
                        curr_time_diff = start_time - metric_data[svc_name].loc[most_close_metric_idx]['TimeStamp']
                        if abs(curr_time_diff) <= 30:
                            for metric in neccessary_metrics:
                                metrics.append(metric_data[svc_name].loc[most_close_metric_idx][metric])
                        else:
                            if curr_time_diff > 0:
                                index_change = 1
                            else:
                                index_change = -1
                            while True:
                                most_close_metric_idx += index_change
                                curr_time_diff = start_time - metric_data[svc_name].loc[most_close_metric_idx]['TimeStamp']
                                if abs(curr_time_diff) <= 30:
                                    for metric in neccessary_metrics:
                                        metrics.append(metric_data[svc_name].loc[most_close_metric_idx][metric])
                                    break
                                    
                        new_data = [curr_spanid, traces.loc[i]['PodName'], traces.loc[i]['OperationName'], start_time, traces.loc[i]['EndTimeUnixNano'] // 1000000000, "", loglines] + metrics
                        curr_trace.loc[-1] = new_data
                        curr_trace.index += 1
                        # Add childspan
                        parentID = traces.loc[i]['ParentID']
                        if parentID != "root":
                            curr_trace.loc[curr_trace[curr_trace['SpanID'] == parentID].index, 'ChildSpans'] += curr_spanid
                            # curr_trace.to_csv(os.path.join(self.save_path, f"{trace_id}.csv"), index=False)
                # for trace_id in self.trace_data.keys():
                #     self.trace_data[trace_id]
            
            
        