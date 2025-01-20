import numpy as np
import pandas as pd
from kan import *
from torch import nn, optim, cuda, amp

import optuna
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset, Subset

import os
from pathlib import Path
import shutil
import pickle

from ToxPred.MolKAN.molkan.src.utils import *

class KAN_predictor(nn.Module):
    def __init__(self, width:list, device, working_dir:Path, results_dir_name:str, LOGGER):
        super(KAN_predictor, self).__init__()
        self.model = KAN(width=width, grid=20, k=3, auto_save=False, device=device, seed=42)
        self.width = width
        self.device = device
        self.dir = working_dir
        self.results_dir_name = results_dir_name
        os.makedirs(self.dir / self.results_dir_name, exist_ok=True)
        self.LOGGER = LOGGER
        self.dtype = torch.get_default_dtype()


    def forward(self, input):
        return self.model.forward(input, singularity_avoiding=False, y_th=1000)
    

    def make_objective(self):
        def obj(trial):

            # set params for optimization
            params = {
                "lr": trial.suggest_float("lr", 1e-5, 5e-1, log=True),
                "lamb": trial.suggest_float("lamb", 1e-5, 5e-1, log=True),
                }
            
            # define KFold & metrics & loss function
            if self.mode == "classification":
                self.kf = StratifiedKFold(self.fold, shuffle=True, random_state=42)
                self.metrics = [roc_auc_score, accuracy_score, precision_score, recall_score, f1_score]
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.kf = KFold(n_splits=self.fold, shuffle=True, random_state=42)
                self.metrics = [r2_score, root_mean_squared_error]
                self.loss_fn = lambda x, y: torch.mean((x - y) ** 2)

            # CrossValidation
            evals = []
            for i, (tr_idx, val_idx) in enumerate(self.kf.split(self.x_tr_val, self.y_tr_val)):
                self.LOGGER.info(f"-----Fold-{i}-----")

                # preparing data
                x_tr, x_val, y_tr, y_val = self.x_tr_val[tr_idx], self.x_tr_val[val_idx], self.y_tr_val[tr_idx], self.y_tr_val[val_idx]
                tr = TensorDataset(torch.from_numpy(x_tr).type(self.dtype), torch.from_numpy(y_tr).type(self.dtype))
                val = TensorDataset(torch.from_numpy(x_val).type(self.dtype), torch.from_numpy(y_val).type(self.dtype))
                tr_loader = DataLoader(dataset=tr, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
                val_loader = DataLoader(dataset=val, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
                
                # train & eval
                results = self.train(tr_loader, val_loader, **params)

                # appending score to evals
                evals.append(results["val_loss"][results["best_epoch"]])
            
            self.LOGGER.info(f"-----{self.fold}Fold finished. Value: {np.mean(evals):.2e} +- {np.std(evals):.1e}-----")
            return np.mean(evals)
        
        return obj
                

    def train(self, tr_loader, val_loader, lr, lamb, grids_refine=False,
              reg_metric="edge_forward_spline_n", lamb_l1=1., lamb_entropy=1., lamb_coefdiff=0., lamb_coef=0., test=False, test_with_grids_refine=False, fold_idx=None):
        
        # init model
        if not grids_refine:
            self.model = KAN(width=self.width, grid=20, k=3, device=self.device, seed=42, auto_save=False)

        # update grid
        self.model.update_grid(torch.cat([x for x, y in tr_loader]).to(self.device))      # more general way is needed... 

        # init optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # init scaler
        self.scaler = cuda.amp.GradScaler(init_scale=2**15, enabled=self.use_amp)

        # set sigmoid func for pred
        self.sigmoid = nn.Sigmoid().to(self.device)

        # init results
        results = {}
        results["tr_loss"] = []
        results["val_loss"] = []
        results["tr_metrics"] = {}
        results["val_metrics"] = {}
        for f in self.metrics:
            results["tr_metrics"][f.__name__] = []
            results["val_metrics"][f.__name__] = []

        pbar = tqdm(range(self.epochs), desc="train_start", dynamic_ncols=True)
        est_cnt = 0
        if not (test_with_grids_refine and "best_val_loss" in self.__dict__.keys()):
            self.best_val_loss = float("inf")

        # training & validation
        # torch.manual_seed(42) # fix shuffle for tr_loader
        for _ in pbar:
            tr_loss_batch = []
            tr_metrics_batch = {}
            for f in self.metrics:
                tr_metrics_batch[f.__name__] = []
                
            # train
            # Profiling
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
            ) as p:
                for x, y in tr_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # autocast (if self.use_amp is True)
                    with cuda.amp.autocast(enabled=self.use_amp):
                        pred = self.forward(x)
                        loss = self.loss_fn(pred, y)
                        reg = self.model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                        objective = loss + lamb * reg
                    self.optimizer.zero_grad()
                    self.scaler.scale(objective).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    tr_loss_batch.append(loss.cpu().detach().item())
                    for f in self.metrics:
                        if f.__name__ == "roc_auc_score" or f.__name__ == "r2_score" or f.__name__ == "root_mean_squared_error":
                            try:
                                tr_metrics_batch[f.__name__].append(f(y.cpu().detach().numpy(), pred.cpu().detach().numpy()))
                            except:
                                pass
                        elif f.__name__ == "accuracy_score":
                            tr_metrics_batch[f.__name__].append(f(y.cpu().detach().numpy(), [1 if p >= 0.5 else 0 for p in self.sigmoid(pred).cpu().detach().numpy()]))
                        else:
                            try:
                                tr_metrics_batch[f.__name__].append(f(y.cpu().detach().numpy(), [1 if p >= 0.5 else 0 for p in self.sigmoid(pred).cpu().detach().numpy()], zero_division=0.0))
                            except:
                                tr_metrics_batch[f.__name__].append(f(y.cpu().detach().numpy(), [1 if p >= 0.5 else 0 for p in self.sigmoid(pred).cpu().detach().numpy()]))
            
            self.LOGGER.info(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
            self.LOGGER.info(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

            # get valid prediction
            val_pred = []
            val_label = []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.forward(x)
                    val_pred.append(pred)
                    val_label.append(y)
                val_pred = torch.cat(val_pred)
                val_label = torch.cat(val_label)    
                
            # resultsへappend
            results["tr_loss"].append(np.mean(tr_loss_batch))
            results["val_loss"].append(self.loss_fn(val_pred, val_label).cpu().detach().item())
            for f in self.metrics:
                try:
                    results["tr_metrics"][f.__name__].append(np.mean(tr_metrics_batch[f.__name__]))
                except:
                    results["tr_metrics"][f.__name__].append(0.0)
                if f.__name__ == "roc_auc_score" or f.__name__ == "r2_score" or f.__name__ == "root_mean_squared_error":
                    try:
                        results["val_metrics"][f.__name__].append(f(val_label.cpu().detach().numpy(), val_pred.cpu().detach().numpy()))
                    except:
                        results["val_metrics"][f.__name__].append(0.0)
                elif f.__name__ == "accuracy_score":
                    results["val_metrics"][f.__name__].append(f(val_label.cpu().detach().numpy(), [1 if p >= 0.5 else 0 for p in self.sigmoid(val_pred).cpu().detach().numpy()]))
                else:
                    try:
                        results["val_metrics"][f.__name__].append(f(val_label.cpu().detach().numpy(), [1 if p >= 0.5 else 0 for p in self.sigmoid(val_pred).cpu().detach().numpy()], zero_division=0.0))
                    except:
                        results["val_metrics"][f.__name__].append(f(val_label.cpu().detach().numpy(), [1 if p >= 0.5 else 0 for p in self.sigmoid(val_pred).cpu().detach().numpy()]))

            if self.mode == "classification":
                pbar.set_description(f"epoch {_} | tr loss: {results['tr_loss'][-1]:.2e} | tr auroc: {results['tr_metrics']["roc_auc_score"][-1]:.2e} | val loss: {results["val_loss"][-1]:.2e} | val auroc: {results['val_metrics']["roc_auc_score"][-1]:.2e}")
                self.LOGGER.debug(f"epoch {_} | tr loss: {results['tr_loss'][-1]:.2e} | tr auroc: {results['tr_metrics']["roc_auc_score"][-1]:.2e} | val loss: {results["val_loss"][-1]:.2e} | val auroc: {results['val_metrics']["roc_auc_score"][-1]:.2e}")
            else:
                pbar.set_description(f"epoch {_} | tr loss: {results['tr_loss'][-1]:.2e} | tr r2: {results['tr_metrics']["r2_score"][-1]:.2e} | val loss: {results["val_loss"][-1]:.2e} | val r2: {results['val_metrics']["r2_score"][-1]:.2e}")
                self.LOGGER.debug(f"epoch {_} | tr loss: {results['tr_loss'][-1]:.2e} | tr r2: {results['tr_metrics']["r2_score"][-1]:.2e} | val loss: {results["val_loss"][-1]:.2e} | val r2: {results['val_metrics']["r2_score"][-1]:.2e}")
                
            # EarlyStopping
            if self.best_val_loss == float("inf"):
                self.best_val_loss = results["val_loss"][-1]
                results["best_epoch"] = _
                if (test or test_with_grids_refine) and fold_idx is not None:
                    torch.save(self.model.state_dict(), self.dir / self.results_dir_name / "best_models" / f"fold_{fold_idx}.pth")
            elif self.best_val_loss - results["val_loss"][-1] > self.best_val_loss * 0.001:
                self.best_val_loss = results["val_loss"][-1]
                results["best_epoch"] = _
                est_cnt = 0
                if (test or test_with_grids_refine) and fold_idx is not None:
                    torch.save(self.model.state_dict(), self.dir / self.results_dir_name / "best_models" / f"fold_{fold_idx}.pth")
            else:
                est_cnt += 1
            if est_cnt == self.patience:
                self.LOGGER.info(f"EarlyStopping at epoch {_}")
                pbar.close()
                if test:
                    del self.best_val_loss
                break

            if _ == self.epochs-1 and test:
                del self.best_val_loss

        return results


    def optimize(self, trial_num, x_tr_val:np.ndarray, y_tr_val:np.ndarray, mode, fold, batch_size, epochs, patience, use_amp):
        self.x_tr_val = x_tr_val
        self.y_tr_val = y_tr_val
        self.mode = mode
        self.fold = fold
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.use_amp = use_amp

        self.LOGGER.info("|-----Hyperparameters Optimization Start-----|")
        self.LOGGER.info(f"-----n_trial: {trial_num}  fold: {self.fold}  batch_size: {self.batch_size}  epochs: {self.epochs}  patience: {self.patience}-----")
        self.study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.SuccessiveHalvingPruner(), direction="minimize")
        obj = self.make_objective()
        self.study.optimize(obj, n_trials=trial_num, timeout=86400, n_jobs=1)
        self.LOGGER.info("|-----Hyperparameters Optimization Finished-----|")
        self.LOGGER.info(f"|-----lr: {self.study.best_params["lr"]:.3e} & lamb: {self.study.best_params["lamb"]:.3e}")


    def test(self, x_test, y_test, lr=None, lamb=None, use_amp=False):
        self.x_test = x_test
        self.y_test = y_test
        self.use_amp = use_amp

        self.LOGGER.info("|-----TEST(emsemble)-----|")

        # make results dirs
        os.makedirs(self.dir / self.results_dir_name / "best_models", exist_ok=True)
        os.makedirs(self.dir / self.results_dir_name / "train_history", exist_ok=True)
        
        # get best params
        if lr is None and lamb is None:
            best_params = self.study.best_params

        # preparing test data
        test = TensorDataset(torch.from_numpy(self.x_test).type(self.dtype), torch.from_numpy(self.y_test).type(self.dtype))
        test_loader = DataLoader(dataset=test, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # CrossValidation training
        for i, (tr_idx, val_idx) in enumerate(self.kf.split(self.x_tr_val, self.y_tr_val)):
            self.LOGGER.info(f"-----Fold-{i}-----")

            # preparing train data
            x_tr, x_val, y_tr, y_val = self.x_tr_val[tr_idx], self.x_tr_val[val_idx], self.y_tr_val[tr_idx], self.y_tr_val[val_idx]
            tr = TensorDataset(torch.from_numpy(x_tr).type(self.dtype), torch.from_numpy(y_tr).type(self.dtype))
            val = TensorDataset(torch.from_numpy(x_val).type(self.dtype), torch.from_numpy(y_val).type(self.dtype))
            tr_loader = DataLoader(dataset=tr, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            val_loader = DataLoader(dataset=val, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

            # train
            if lr is None and lamb is None:
                results = self.train(tr_loader, val_loader, **best_params, test=True, fold_idx=i)
            else:
                results = self.train(tr_loader, val_loader, lr=lr, lamb=lamb, test=True, fold_idx=i)

            # test
            self.model.load_state_dict(torch.load(self.dir / self.results_dir_name / "best_models" / f"fold_{i}.pth"))
            with torch.no_grad():
                pred_list = []
                for x, y in test_loader:
                    x = x.to(self.device)
                    pred = self.forward(x)
                    pred_list.append(pred)
                pred = torch.cat(pred_list)
                if i == 0:
                    test_pred = pred
                else:
                    test_pred = test_pred + pred
            
            # saving train history graph
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            ax1 = axes[0]
            ax2 = axes[1]
            ax1.plot(results["tr_loss"], label="train")
            ax1.plot(results["val_loss"], label="valid")
            ax1.set_xlabel("epoch")
            ax1.set_ylim([0, 1])
            ax1.legend()
            if self.mode == "classification":
                ax1.set_title(f"Loss (logloss)")
                ax2.plot(results["tr_metrics"]["roc_auc_score"], label="train")
                ax2.plot(results["val_metrics"]["roc_auc_score"], label="valid")
                ax2.set_xlabel("epoch")
                ax2.set_ylim([0, 1])
                ax2.legend()
                ax2.set_title("AUROC")
            else:
                ax1.set_title(f"Loss (MSE)")
                ax2.plot(results["tr_metrics"]["r2_score"], label="train")
                ax2.plot(results["val_metrics"]["r2_score"], label="valid")
                ax2.set_xlabel("epoch")
                ax2.set_ylim([0, 1])
                ax2.legend()
                ax2.set_title("R2")
            plt.tight_layout()
            plt.savefig(self.dir / self.results_dir_name / "train_history" / f"fold_{i}.png")
            plt.close()
        
        self.LOGGER.info(f"-----{self.fold}-Fold finished-----")

        # emsemble, calculating test scores
        self.LOGGER.info("-----scores-----")
        df = pd.DataFrame(columns=[f.__name__ for f in self.metrics])
        test_pred = self.sigmoid((test_pred / self.fold)).cpu().detach().numpy()
        test_results = {}
        for f in self.metrics:
            if f.__name__ == "roc_auc_score" or f.__name__ == "r2_score" or f.__name__ == "root_mean_squared_error":
                test_results[f.__name__] = f(self.y_test, test_pred)
            else:
                test_results[f.__name__] = f(self.y_test, [1 if p >= 0.5 else 0 for p in test_pred])
            self.LOGGER.info(f"{f.__name__}: {test_results[f.__name__]:.2e}")
            df.loc[0, f.__name__] = f"{test_results[f.__name__]:.2e}"
        df.to_csv(self.dir / self.results_dir_name / "test_scores.csv")
        
        self.LOGGER.info("|-----TEST FINISHED-----|")


    def test_with_grids_refine(self, x_test, y_test, lr=None, lamb=None, use_amp=False):
        self.x_test = x_test
        self.y_test = y_test
        self.use_amp = use_amp

        self.LOGGER.info("|-----TEST(emsemble) with grids refine-----|")

        # make results dirs
        os.makedirs(self.dir / self.results_dir_name / "best_models", exist_ok=True)
        os.makedirs(self.dir / self.results_dir_name / "train_history", exist_ok=True)
        
        # get best params
        if lr is None and lamb is None:
            best_params = self.study.best_params

        # preparing test data
        test = TensorDataset(torch.from_numpy(self.x_test).type(self.dtype), torch.from_numpy(self.y_test).type(self.dtype))
        test_loader = DataLoader(dataset=test, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # CrossValidation training
        for i, (tr_idx, val_idx) in enumerate(self.kf.split(self.x_tr_val, self.y_tr_val)):
            self.LOGGER.info(f"-----Fold-{i}-----")

            # preparing train data
            x_tr, x_val, y_tr, y_val = self.x_tr_val[tr_idx], self.x_tr_val[val_idx], self.y_tr_val[tr_idx], self.y_tr_val[val_idx]
            tr = TensorDataset(torch.from_numpy(x_tr).type(self.dtype), torch.from_numpy(y_tr).type(self.dtype))
            val = TensorDataset(torch.from_numpy(x_val).type(self.dtype), torch.from_numpy(y_val).type(self.dtype))
            tr_loader = DataLoader(dataset=tr, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            val_loader = DataLoader(dataset=val, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

            # train
            grids = [10, 20]
            if lr is None and lamb is None:
                results = self.train(tr_loader, val_loader, **best_params, test_with_grids_refine=True, fold_idx=i)
                self.LOGGER.info(f"---grids refine 5 -> 10")
                self.model = self.model.refine(new_grid=grids[0])
                best_params["lamb"] *= 10
                results2 = self.train(tr_loader, test_loader, **best_params, grids_refine=True, test_with_grids_refine=True, fold_idx=i)
                self.model.auto_save=False
                self.LOGGER.info(f"---grids refine 10 -> 20")
                self.model = self.model.refine(new_grid=grids[1])
                best_params["lamb"] *= 10
                results3 = self.train(tr_loader, test_loader, **best_params, grids_refine=True, test_with_grids_refine=True, fold_idx=i)
                self.model.auto_save=False
            else:
                results = self.train(tr_loader, val_loader, lr=lr, lamb=lamb, test_with_grids_refine=True, fold_idx=i)
                self.LOGGER.info(f"---grids refine 5 -> 10")
                self.model = self.model.refine(new_grid=grids[0])
                results2 = self.train(tr_loader, test_loader, lr=lr, lamb=lamb*10, grids_refine=True, test_with_grids_refine=True, fold_idx=i)
                self.model.auto_save=False
                self.LOGGER.info(f"---grids refine 10 -> 20")
                self.model = self.model.refine(new_grid=grids[1])
                results3 = self.train(tr_loader, test_loader, lr=lr, lamb=lamb*100, grids_refine=True, test_with_grids_refine=True, fold_idx=i)
                self.model.auto_save=False

            # test
            self.model.load_state_dict(torch.load(self.dir / self.results_dir_name / "best_models" / f"fold_{i}.pth"))
            with torch.no_grad():
                pred_list = []
                for x, y in test_loader:
                    x = x.to(self.device)
                    pred = self.forward(x)
                    pred_list.append(pred)
                pred = torch.cat(pred_list)
                if i == 0:
                    test_pred = pred
                else:
                    test_pred = test_pred + pred
            
            # saving train history graph
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            ax1 = axes[0]
            ax2 = axes[1]
            ax1.plot(results["tr_loss"]+results2["tr_loss"]+results3["tr_loss"], label="train")
            ax1.plot(results["val_loss"]+results2["val_loss"]+results3["val_loss"], label="valid")
            ax1.set_xlabel("epoch")
            ax1.set_ylim([0, 1])
            ax1.legend()
            if self.mode == "classification":
                ax1.set_title(f"Loss (logloss)")
                ax2.plot(results["tr_metrics"]["roc_auc_score"]+results2["tr_metrics"]["roc_auc_score"]+results3["tr_metrics"]["roc_auc_score"], label="train")
                ax2.plot(results["val_metrics"]["roc_auc_score"]+results2["val_metrics"]["roc_auc_score"]+results3["val_metrics"]["roc_auc_score"], label="valid")
                ax2.set_xlabel("epoch")
                ax2.set_ylim([0, 1])
                ax2.legend()
                ax2.set_title("AUROC")
            else:
                ax1.set_title(f"Loss (MSE)")
                ax2.plot(results["tr_metrics"]["r2_score"]+results2["tr_metrics"]["r2_score"]+results3["tr_metrics"]["r2_score"], label="train")
                ax2.plot(results["val_metrics"]["r2_score"]+results2["val_metrics"]["r2_score"]+results3["val_metrics"]["r2_score"], label="valid")
                ax2.set_xlabel("epoch")
                ax2.set_ylim([0, 1])
                ax2.legend()
                ax2.set_title("R2")
            plt.tight_layout()
            plt.savefig(self.dir / self.results_dir_name / "train_history" / f"fold_{i}_grids_refine.png")
            plt.close()

            # delete best_val_loss
            del self.best_val_loss
        
        self.LOGGER.info(f"-----{self.fold}-Fold finished-----")

        # emsemble, calculating test scores
        self.LOGGER.info("-----scores-----")
        df = pd.DataFrame(columns=[f.__name__ for f in self.metrics])
        test_pred = self.sigmoid((test_pred / self.fold)).cpu().detach().numpy()
        test_results = {}
        for f in self.metrics:
            if f.__name__ == "roc_auc_score" or f.__name__ == "r2_score" or f.__name__ == "root_mean_squared_error":
                test_results[f.__name__] = f(self.y_test, test_pred)
            else:
                test_results[f.__name__] = f(self.y_test, [1 if p >= 0.5 else 0 for p in test_pred])
            self.LOGGER.info(f"{f.__name__}: {test_results[f.__name__]:.2e}")
            df.loc[0, f.__name__] = f"{test_results[f.__name__]:.2e}"
        df.to_csv(self.dir / self.results_dir_name / "test_scores_grids_refine.csv")
        
        self.LOGGER.info("|-----TEST FINISHED-----|")



if __name__ == "__main__":
    file = os.path.basename(__file__).split(".")[0]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CURRENT_DIR = Path.cwd()
    LOGGER = init_logger(module_name=file, outdir=CURRENT_DIR)