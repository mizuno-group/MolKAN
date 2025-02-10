import os
import numpy as np
import pandas as pd
import torch
from schedulefree import RAdamScheduleFree
from tqdm import tqdm
from src.layers import *
import src.utils as utils
import src.data_handler as dh
from src.trainer import Trainer
import optuna

# using RAdamScheduleFree

class KAN_Tuner:
    def __init__(self, seed, mode, train_ds, valid_ds, test_ds, outdim, device, metrics, logger, outdir, note):
        self.mode = mode
        self.seed = seed
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.outdim = outdim
        self.device = device
        self.metrics = metrics
        self.logger = logger
        self.outdir = outdir
        self.note = note

    def _make_objective(self):
        # train_epoch
        def train_epoch(model, optimizer, loss_func, trainloader):
            model.train()
            optimizer.train()
            total_loss = 0
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = model(x)

                mask = (~torch.isnan(y)).float()
                y = torch.where(torch.isnan(y), torch.zeros_like(y), y)
                
                loss = loss_func(pred, y, weight=mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x)
            return total_loss / len(trainloader.dataset)
        
        # evaluation
        def evaluate(model, optimizer, loss_func, validloader):
            with torch.no_grad():
                model.eval()
                optimizer.eval()
                total_loss = 0
                for x, y in validloader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x)

                    mask = (~torch.isnan(y)).float()
                    y = torch.where(torch.isnan(y), torch.zeros_like(y), y)

                    loss = loss_func(pred, y, mask)
                    total_loss += loss.item() * len(x)
            
                valid_loss = total_loss / len(validloader.dataset)
                return valid_loss

        # objective object (return best valid loss)
        def objective(trial):
            utils.fix_seed(self.seed)

            hidden_layer_dim = trial.suggest_int("hidden_layer_dim", 1, 1024)
            num_grids = trial.suggest_int("num_grids", 1, 8)
            batch_size = trial.suggest_categorical("batch_size", [4, 16, 64, 256, 1024])
            learning_rate = trial.suggest_float("learning_rate", 1.0e-5, 1.0e-2, log=True)
            dropout = trial.suggest_categorical("dropout", [0, 0.2, 0.5])

            # prepare DataLoader
            trainloader = dh.prep_dataloader(self.train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
            validloader = dh.prep_dataloader(self.valid_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)

            # prepare model, optimizer, loss_func
            model = FourierKAN_Layer([512, hidden_layer_dim, self.outdim], mode=self.mode, num_grids=num_grids, dropout=dropout)
            model = model.to(self.device)
            optimizer = RAdamScheduleFree(params=model.parameters(), lr=learning_rate, weight_decay=0)
            if self.mode == "classification":
                loss_func = utils.BCELoss()
            else:
                loss_func = utils.MSE()
            
            # Train and validation
            self.logger.info("=== train start ===")
            pbar = tqdm(range(50), desc="train_start", dynamic_ncols=True)
            
            train_losses = []
            valid_losses = []
            for i in pbar:
                train_loss = train_epoch(model, optimizer, loss_func, trainloader)
                valid_loss = evaluate(model, optimizer, loss_func, validloader)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                pbar.set_description(f"epoch {i+1} | train loss: {train_loss:.2e} | valid loss: {valid_loss:.2e}")
                self.logger.debug(f"epoch {i+1} | train loss: {train_loss:.2e} | valid loss: {valid_loss:.2e}")

            trial.set_user_attr("model_state_dict", model.state_dict())
            trial.set_user_attr("train_losses", train_losses)
            trial.set_user_attr("valid_losses", valid_losses)
            self.logger.info(f"=== train finished ===")

            return valid_loss
        
        return objective
    
    def _test_func(self, testloader, model, metrics):
        with torch.no_grad():
            model.eval()
            test_scores = []
            pred_list = []
            y_list = []
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                pred_list.append(pred.detach().cpu().numpy())
                y_list.append(y.detach().cpu().numpy())
        
            pred = np.concatenate(pred_list)
            y = np.concatenate(y_list)
            for f in metrics:
                if hasattr(utils.Metrics, f):
                    func = getattr(utils.Metrics, f)
                    test_scores.append(func(pred, y))
                else:
                    raise AttributeError(f"Metric not found({f}), check your metric name.")

            return test_scores
    
    def optimize_and_test(self):
        self.logger.info("===== Hyperparameters tuning start =====")
        objective = self._make_objective()
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed),
                                    pruner=optuna.pruners.SuccessiveHalvingPruner(), direction="minimize")
        study.optimize(objective, n_trials=150)
        best_trial = study.best_trial
        os.makedirs(os.path.join(self.outdir, "plots"), exist_ok=True)
        utils.plot_progress(os.path.join(self.outdir, "plots"), best_trial.user_attrs["train_losses"], best_trial.user_attrs["valid_losses"], 50, note=self.note)
        best_model_state = best_trial.user_attrs["model_state_dict"]
        os.makedirs(os.path.join(self.outdir, "models"), exist_ok=True)
        torch.save(best_model_state, os.path.join(self.outdir, "models", f"{self.note}_best_model.pt"))
        best_params = study.best_params

        utils.fix_seed(self.seed)
        # prepare testloader
        testloader = dh.prep_dataloader(self.test_ds, best_params["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        # prepare model
        model = FourierKAN_Layer([512, best_params["hidden_layer_dim"], self.outdim],self.mode, best_params["num_grids"], best_params["dropout"])
        model.load_state_dict(torch.load(os.path.join(self.outdir, "models", f"{self.note}_best_model.pt")))
        model = model.to(self.device)
        # test
        test_scores = test_func(model, testloader, self.metrics, self.device)
        return test_scores, list(best_params.keys()), list(best_params.values())


class MLP_Tuner:
    def __init__(self, seed, mode, train_ds, valid_ds, test_ds, outdim, device, metrics, logger, outdir, note):
        self.mode = mode
        self.seed = seed
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.outdim = outdim
        self.device = device
        self.metrics = metrics
        self.logger = logger
        self.outdir = outdir
        self.note = note

    def _make_objective(self):
        # train_epoch
        def train_epoch(model, optimizer, loss_func, trainloader):
            model.train()
            optimizer.train()
            total_loss = 0
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = model(x)

                mask = (~torch.isnan(y)).float()
                y = torch.where(torch.isnan(y), torch.zeros_like(y), y)

                loss = loss_func(pred, y, mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x)
            return total_loss / len(trainloader.dataset)
        
        # evaluation
        def evaluate(model, optimizer, loss_func, validloader):
            with torch.no_grad():
                model.eval()
                optimizer.eval()
                total_loss = 0
                for x, y in validloader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x)

                    mask = (~torch.isnan(y)).float()
                    y = torch.where(torch.isnan(y), torch.zeros_like(y), y)

                    loss = loss_func(pred, y, mask)
                    total_loss += loss.item() * len(x)
            
                valid_loss = total_loss / len(validloader.dataset)
                return valid_loss

        # objective object (return best valid loss)
        def objective(trial):
            utils.fix_seed(self.seed)

            hidden_layer_dim = trial.suggest_int("hidden_layer_dim", 1, 1024)
            batch_size = trial.suggest_categorical("batch_size", [4, 16, 64, 256, 1024])
            learning_rate = trial.suggest_float("learning_rate", 1.0e-5, 1.0e-2, log=True)
            dropout = trial.suggest_categorical("dropout", [0, 0.2, 0.5])

            # prepare DataLoader
            trainloader = dh.prep_dataloader(self.train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
            validloader = dh.prep_dataloader(self.valid_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)

            # prepare model, optimizer, loss_func
            model = MLP_Layer([512, hidden_layer_dim, self.outdim], mode=self.mode, dropout=dropout)
            model = model.to(self.device)
            optimizer = RAdamScheduleFree(model.parameters(), lr=learning_rate, weight_decay=0)
            if self.mode == "classification":
                loss_func = utils.BCELoss()
            else:
                loss_func = utils.MSE()
            
            # Train and validation
            self.logger.info("=== train start ===")
            pbar = tqdm(range(50), desc="train_start", dynamic_ncols=True)
            
            train_losses = []
            valid_losses = []
            for i in pbar:
                train_loss = train_epoch(model, optimizer, loss_func, trainloader)
                valid_loss = evaluate(model, optimizer, loss_func, validloader)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                pbar.set_description(f"epoch {i+1} | train loss: {train_loss:.2e} | valid loss: {valid_loss:.2e}")
                self.logger.debug(f"epoch {i+1} | train loss: {train_loss:.2e} | valid loss: {valid_loss:.2e}")

            trial.set_user_attr("model_state_dict", model.state_dict())
            trial.set_user_attr("train_losses", train_losses)
            trial.set_user_attr("valid_losses", valid_losses)
            self.logger.info(f"=== train finished ===")

            return valid_loss
        
        return objective
    
    def optimize_and_test(self):
        self.logger.info("===== Hyperparameters tuning start =====")
        objective = self._make_objective()
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed),
                                    pruner=optuna.pruners.SuccessiveHalvingPruner(), direction="minimize")
        study.optimize(objective, n_trials=120)
        best_trial = study.best_trial
        os.makedirs(os.path.join(self.outdir, "plots"), exist_ok=True)
        utils.plot_progress(os.path.join(self.outdir, "plots"), best_trial.user_attrs["train_losses"], best_trial.user_attrs["valid_losses"], 50, note=self.note)
        best_model_state = best_trial.user_attrs["model_state_dict"]
        os.makedirs(os.path.join(self.outdir, "models"), exist_ok=True)
        torch.save(best_model_state, os.path.join(self.outdir, "models", f"{self.note}_best_model.pt"))
        best_params = study.best_params

        utils.fix_seed(self.seed)
        # prepare testloader
        testloader = dh.prep_dataloader(self.test_ds, best_params["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        # prepare model
        model = MLP_Layer([512, best_params["hidden_layer_dim"], self.outdim],self.mode, best_params["dropout"])
        model.load_state_dict(torch.load(os.path.join(self.outdir, "models", f"{self.note}_best_model.pt")))
        model = model.to(self.device)
        # test
        test_scores = test_func(model, testloader, self.metrics, self.device)
        return test_scores, list(best_params.keys()), list(best_params.values())


def test_func(model, testloader, metrics, device):
    with torch.no_grad():
        model.eval()
        test_scores = []
        pred_list = []
        y_list = []
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_list.append(pred.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())
        
        pred = np.concatenate(pred_list)
        y = np.concatenate(y_list)
        for f in metrics:
            if hasattr(utils.Metrics, f):
                func = getattr(utils.Metrics, f)
                test_scores.append(func(pred, y))
            else:
                raise AttributeError(f"Metric not found({f}), check your metric name.")

        return test_scores
            