import os
from datetime import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import pickle

import sys
project_path = os.path.abspath(__file__).split("/experiments")[0]
sys.path.append(project_path)

# original packages in src
from src import utils
from src import data_handler as dh
from src.tuning import KAN_Tuner, MLP_Tuner
""" === change based on the model you want to use === """
from src.models import *

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, help="datasets name you want to use")
parser.add_argument("--label_columns", type=utils.parse_str_list, help="label columns names")
parser.add_argument("--model", type=str, help="Use KAN or MLP")
parser.add_argument("--mode", type=str, default="classification", help="classification or regression")

args = parser.parse_args()

cfg = vars(args)

dataset_dict = {"BACE": "bace",
                                "ClinTox": "clintox_M",
                                "CYP2C9": "cyp2c9_inhib",
                                "CYP3A4": "cyp3a4_inhib",
                                "hERG": "herg_karim",
                                "LD50": "ld50",
                                "SIDER": "sider",
                                "Tox21": "tox21_M",
                                "ToxCast": "toxcast_M"}

cfg["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cfg["outdir"] = os.path.join(os.path.dirname(__file__).replace("experiments", "results_and_logs"), f"{cfg["datasets"]}_{cfg["mode"]}", f"{datetime.now().strftime("%y%m%d")}_{cfg["model"]}")
os.makedirs(cfg["outdir"], exist_ok=True)


if __name__ == "__main__":

    data_name = dataset_dict[cfg["datasets"]]
    if cfg["mode"] == "classification":
        metrics = ["accuracy", "AUROC", "sensitivity", "precision", "MCC"]
    elif cfg["mode"] == "regression":
        metrics = ["R2", "RMSE", "MAE"]
    else:
        raise NameError("check mode argument. classification or regression ?")

    results = pd.DataFrame(columns=metrics)

    # 3 seeds
    for seed in [42, 222, 2025]:
        # TfVAE_repr or unimol_repr
        for repr in ["TfVAE_repr", "unimol_repr"]:
            
            logger = utils.init_logger(cfg["outdir"])
            trial_note = f"seed{seed}_{repr}"

            logger.info(f"===== {trial_note} =====")

            # prepare datasets
            utils.fix_seed(seed)
            if repr == "TfVAE_repr":
                x = np.load(project_path + f"/data/{repr}/{data_name}_mu.npy")
            else:
                x = np.load(project_path + f"/data/{repr}/{data_name}.npy")
            y = pd.read_csv(project_path + f"/data/tox_csv/{data_name}.csv", index_col=0)
            y = np.array(y[cfg["label_columns"]])
            num_labels = len(cfg["label_columns"])
            dset = dh.prep_dataset(x, y)
            transform = dh.array_to_tensors_flaot()
            if len(cfg["label_columns"]) == 1 and cfg["mode"] == "classification":
                train_set, valid_set, test_set = dh.split_dataset_stratified(dset, [0.8, 0.1, 0.1], shuffle=True, transform=transform)
            else:
                train_set, valid_set, test_set = dh.split_dataset(dset, [0.8, 0.1, 0.1], shuffle=True, transform=transform)
            
            if cfg["model"] == "KAN":
                tuner = KAN_Tuner(seed, cfg["mode"], train_set, valid_set, test_set, num_labels, cfg["device"], metrics, logger, cfg["outdir"], trial_note)
            elif cfg["model"] == "MLP":
                tuner = MLP_Tuner(seed, cfg["mode"], train_set, valid_set, test_set, num_labels, cfg["device"], metrics, logger, cfg["outdir"], trial_note)
            else:
                raise NameError("check model argument. KAN or MLP ?")

            test_score, params_names, params_values = tuner.optimize_and_test()
            
            for metric, score in zip(metrics, test_score):
                logger.info(f"test_{metric}: {score:.3f}")

            try:
                params.loc[trial_note] = params_values
            except:
                params = pd.DataFrame(columns=params_names)
                params.loc[trial_note] = params_values
            
            results.loc[trial_note] = test_score
    
    params.to_csv(os.path.join(cfg["outdir"], "parameters.csv"))
    results.to_csv(os.path.join(cfg["outdir"], "results.csv"))

    logger.info("===== ALL FINISHED =====")







