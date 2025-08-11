from ..model.MAADModel import MAADAgent

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from multiprocessing import Pool

from ..utils.drain3.template_miner import TemplateMiner

from ..utils.SIF.src import params, data_io, SIF_embedding

from ..utils.utils import RemoveSignals, Drain_Init

import os
import time
import torch
import logging
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# Init SIF parameters
wordfile = "./MAFD/glove/vectors.txt" # word vector file, can be downloaded from GloVe website
weightfile = "./MAFD/glove/vocab.txt" # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
# load word vectors
(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# set parameters
param = params.params()
param.rmpc = rmpc

def Logs2Vectors(cur_logs, tmp: TemplateMiner) -> list:
    """
    Transfer logs and spans to sentence vectors
    """
    # sentences = None
    # if type(cur_logs) == str:
    #     cur_logs = eval(cur_logs)
    #     sentences = [cur_logs[i][2] for i in range(len(cur_logs))]
    # else:
    #     sentences = cur_logs
    if cur_logs == "/*" or cur_logs == "":
        return np.array([[0.0 for i in range(300)]])
    sentences = cur_logs.split("%")
    embedding = []
    if cur_logs != []:
        for i in range(len(sentences)):
            sentence = RemoveSignals(tmp.add_log_message(sentences[i].strip())['template_mined'])
            sentences[i] = sentence
        x, m = data_io.sentences2idx(sentences=sentences, words=words)
        w = data_io.seq2weight(x, m, weight4ind)
        embedding = SIF_embedding.SIF_embedding(We, x, w, param)
    return embedding

class MAFDRCA:
    def __init__(self, drain_config: str, model_save_path: str, mode: str = "train", logger: logging.Logger = None):
        self.svcs = ['ts-price-service', 'ts-notification-service', 'ts-ticket-office-service', 'ts-payment-service', 'ts-auth-service', 'ts-delivery-service', 'ts-admin-travel-service', 'ts-security-service', 'ts-admin-user-service', 'ts-admin-route-service', 'ts-verification-code-service', 'ts-train-service', 'ts-config-service', 'ts-user-service', 'ts-food-delivery-service', 'ts-inside-payment-service', 'ts-gateway-service', 'ts-contacts-service', 'ts-rebook-service', 'ts-order-service', 'ts-news-service', 'ts-consign-price-service', 'ts-route-plan-service', 'ts-wait-order-service', 'ts-travel-plan-service', 'ts-ui-dashboard-5477b965f-2chkr_metric.csvservice', 'ts-admin-order-service', 'ts-travel-service', 'ts-voucher-service', 'ts-cancel-service', 'ts-basic-service', 'ts-route-service', 'ts-preserve-service', 'ts-seat-service', 'ts-station-service', 'ts-order-other-service', 'ts-admin-basic-info-service', 'ts-avatar-service', 'ts-execute-service', 'ts-travel2-service', 'ts-assurance-service', 'ts-train-food-service', 'ts-station-food-service', 'ts-food-service', 'ts-preserve-other-service', 'ts-consign-service']
        self.maad_dict, self.optim_dict = self._create_maad()
        self.model_save_path = model_save_path

        self.tmp = Drain_Init(drain_config)
        self.tmp.load_state("tt")

        self.mode = mode
        if self.mode == "test":
            self.load_maad()

        # self.root_cause = {
        #                     0: "normal", 
        #                     1: "return",
        #                     2: "exception",
        #                     3: "cpu_contention",
        #                     4: "network_delay"
        #                    }
        self.root_cause = {
                            "normal": 0, 
                            "return": 1,
                            "exception": 2,
                            "cpu_contention": 3,
                            "network_delay": 4
                           }
        
        self.logger = logger
        self.loss_func = CrossEntropyLoss()

    def _create_maad(self) -> dict:
        model_dict = {}
        optim_dict = {}
        for svc in self.svcs:
            model_dict[svc] = MAADAgent(corresponds_svc=svc)
            optim_dict[svc] = torch.optim.Adam(lr=0.0001, params=model_dict[svc].parameters())
        return model_dict, optim_dict
    
    def save_maad(self):
        """
        Save MAAD agents to disk
        """
        for svc in self.maad_dict.keys():
            torch.save(self.maad_dict[svc].state_dict(), os.path.join(self.model_save_path, f"{svc}.pt"))

    def load_maad(self):
        """
        Load MAAD agents from disk
        """
        for svc in self.maad_dict.keys():
            self.maad_dict[svc].load_state_dict(torch.load(os.path.join(self.model_save_path, f"{svc}.pt")))

    def split_dataset(self, data_path: str):
        trainset = {}
        testset = {}
        self.logger.info("Start to split trainset and testset")
        for fault_type in os.listdir(data_path):
            self.logger.info(f"Current root cause type: {fault_type}")
            fault_path = os.path.join(data_path, fault_type)
            trainset[fault_type] = []
            testset[fault_type] = []
            traces = os.listdir(fault_path)
            split_idx = int(0.7 * len(traces))
            for i in range(split_idx):
                trace_file = os.path.join(fault_path, traces[i])
                trainset[fault_type].append(trace_file)
            for i in range(split_idx, len(traces)):
                trace_file = os.path.join(fault_path, traces[i])
                testset[fault_type].append(trace_file)
        
        return trainset, testset

    def workflow(self, data_path: str = "../baselines/Extracted_dataset/TT"):
        train_set, test_set = self.split_dataset(data_path)

        # training phase
        trains = []
        for fault_type in train_set.keys():
            if fault_type == "normal_normal":
                random.shuffle(train_set[fault_type])
                trains += train_set[fault_type][:100]
            else:
                trains += train_set[fault_type]
        for epoch in tqdm(range(15)):
            random.shuffle(trains)
            # for trace in trains:
            for i in tqdm(range(len(trains))):
                trace = trains[i]
                fault_type = self._get_fault_type_from_path(trace)
                losses, err_data = self.process_trace(trace, fault_type)
                if i % 10 == 0:
                    self.logger.info(f"curr label: {fault_type}, curr loss: {losses}, err_softmax: {err_data}")
                if i % 100 == 0 and i != 0:
                    self.save_maad()
            self.save_maad()
            
        # testing phase
        tests = []
        for fault_type in test_set.keys():
            if fault_type == "normal_normal":
                random.shuffle(test_set[fault_type])
                tests += test_set[fault_type][:100]
            else:
                tests += test_set[fault_type]
        for epoch in tqdm(range(1)):
            random.shuffle(tests)
            # for trace in trains:
            for i in tqdm(range(len(tests))):
                trace = tests[i]
                fault_type = self._get_fault_type_from_path(trace)
                final_out, _ = self.process_trace(trace, fault_type)  

                for svc in final_out.keys():
                    data = final_out[svc].detach().numpy().tolist()
                    final_out[svc] = data[0]

    # def test_workflow(self, data_path: str = "../baselines/Extracted_dataset/TT"):
    #     test_set, _ = self.split_dataset(data_path)

    #     tests = []
    #     for fault_type in test_set.keys():
    #         if fault_type == "normal_normal":
    #             random.shuffle(test_set[fault_type])
    #             tests += test_set[fault_type][:100]
    #         else:
    #             tests += test_set[fault_type]
    #     for epoch in tqdm(range(1)):
    #         random.shuffle(tests)
    #         # for trace in trains:
    #         for i in tqdm(range(len(tests))):
    #             trace = tests[i]
    #             fault_type = self._get_fault_type_from_path(trace)
    #             final_out, _ = self.process_trace(trace, fault_type)  

    #             for svc in final_out.keys():
    #                 data = final_out[svc].detach().numpy().tolist()
    #                 final_out[svc] = data[0]
                
    #             # with open("train.txt", "a+") as f:
    #             #     res = []
    #             #     for svc in self.svcs:
    #             #         if svc not in final_out.keys():
    #             #             res.append([0,0,0,0,0])
    #             #         else:
    #             #             res.append(final_out[svc])
    #             #     if "normal" in fault_type:
    #             #         f.write(f"{0},{0};{res}\n")
    #             #     else:
    #             #         label_info = fault_type.split("_")
    #             #         err_svc = label_info[0]
    #             #         err_type = self.root_cause['_'.join(label_info[1:])]
    #             #         f.write(f"{self.svcs.index(err_svc)},{err_type};{res}\n")


    def process_trace(self, trace_path: str, label: str):
        # Read current trace
        start_time = time.time()
        events_num = 0
        child_spans_input = {}
        final_out = {}
        
        label_info = label.split("_")

        err_svc = label_info[0]
        err_type = self.root_cause['_'.join(label_info[1:])]

        pd_data = pd.read_csv(trace_path, keep_default_na=False)
        for i in range(pd_data.shape[0]):
            spanid = pd_data.loc[i]['SpanID']
            svc = pd_data.loc[i]['PodName'].split("service")[0] + "service"
            trace = pd_data.loc[i]['OperationName']
            child_spans = pd_data.loc[i]['ChildSpans']
            childs = []
            for i in range(int(len(child_spans) / 16)):
                childs.append(child_spans[i * 16: (i + 1) * 16])
            logs = pd_data.loc[i]['logs']
            cpu = pd_data.loc[i]['CpuUsageRate(%)']
            clientp90 = pd_data.loc[i]['PodClientLatencyP90(s)']
            clientp90 = 0 if clientp90 == "" else clientp90
            serverp90 = pd_data.loc[i]['PodServerLatencyP90(s)']
            serverp90 = 0 if serverp90 == "" else serverp90

            vectors = np.concatenate((Logs2Vectors(trace, self.tmp), Logs2Vectors(logs, self.tmp)), axis=0, dtype=np.float32)
            metric_vector = np.array([[cpu, clientp90, serverp90]] * vectors.shape[0], dtype=np.float32)
            vectors = np.concatenate((vectors, metric_vector), axis=1, dtype=np.float32)
            events_num += vectors.shape[0]
            curr_tensor = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            if spanid in child_spans_input.keys():
                prev_out = child_spans_input[spanid]
                curr_tensor = torch.cat([prev_out, curr_tensor], dim=-2)
                del child_spans_input[spanid]

            csvc, out_tensor, _, rca = self.maad_dict[svc](curr_tensor)
            final_out[svc] = rca

            for child in childs:
                child_spans_input[child] = out_tensor
       
        losses = {}
        err_data = None
        
        end_time = time.time()
        self.logger.info(f"Current trace: {trace_path}, events num: {events_num}, time cost: {(end_time - start_time) * 10000 / events_num} ms")
        if self.mode == "train":
            for svc in final_out.keys():
                if err_svc == "normal":
                    tgt = torch.tensor([0], dtype=torch.long)
                    loss = self.loss_func(final_out[svc], tgt) * 0.01
                    loss.backward(retain_graph=True)
                    losses[svc] = loss
                    continue
                if svc == err_svc:
                    tgt = torch.tensor([err_type], dtype=torch.long)
                    loss = self.loss_func(final_out[svc], tgt)
                    loss.backward(retain_graph=True)
                    err_data = final_out[svc]
                else:
                    tgt = torch.tensor([0], dtype=torch.long)
                    loss = self.loss_func(final_out[svc], tgt) * 0.01
                    loss.backward(retain_graph=True)
                losses[svc] = loss
            for svc in final_out.keys():
                self.optim_dict[svc].step()
                self.optim_dict[svc].zero_grad()
        else:
            return final_out, None
        
        
        return losses, err_data

    def _get_fault_type_from_path(self, trace_path):
        return trace_path.split("/")[4]