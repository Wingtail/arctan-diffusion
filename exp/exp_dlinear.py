from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, save_full_checkpoint
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Exp_DLinear(Exp_Basic):
    def __init__(self, args):
        if hasattr(args, "input_len") and hasattr(args, "seq_len"):
            if args.seq_len != args.input_len:
                args.seq_len = args.input_len
        super(Exp_DLinear, self).__init__(args)
        self._linear_probe_best = float("inf")

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_head_module(self):
        return self._unwrap_model()

    def _apply_linear_probe_freeze(self):
        if not getattr(self.args, "linear_probe", False):
            return
        head = self._get_head_module()
        if head is None:
            return
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in head.parameters():
            p.requires_grad_(True)

    def _maybe_save_linear_probe_head(self, path, epoch, val_loss):
        if not getattr(self.args, "linear_probe", False):
            return
        head = self._get_head_module()
        if head is None:
            return
        if val_loss >= self._linear_probe_best:
            return
        self._linear_probe_best = val_loss
        save_dir = getattr(self.args, "linear_probe_save_dir", None) or path
        os.makedirs(save_dir, exist_ok=True)
        fname = "linear_probe_best_head.pth"
        if os.path.abspath(save_dir) != os.path.abspath(path):
            tag = os.path.basename(path.rstrip("/")) or "run"
            fname = f"linear_probe_best_head_{tag}.pth"
        save_path = os.path.join(save_dir, fname)
        torch.save(
            {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "state_dict": head.state_dict(),
            },
            save_path,
        )
        print(f"Saved linear-probe head to: {save_path}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            transfer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = transfer_weights(
                self.args.load_checkpoints,
                model,
                exclude_head=False,
                device=transfer_device,
            )

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(
            "number of model params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def pretrain(self):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        self._apply_linear_probe_freeze()
        model_optim = self._select_optimizer()
        model_criteria = self._select_criterion()
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model_optim, gamma=self.args.lr_decay
        )

        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()
            train_loss = []
            train_loader = tqdm(train_loader, desc="Pretrain")

            print(
                "Current learning rate: {:.7f}".format(
                    model_scheduler.get_last_lr()[0]
                )
            )

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(outputs, batch_y)
                loss.backward()
                model_optim.step()

                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            vali_loss = self.valid(vali_loader, model_criteria)
            self._maybe_save_linear_probe_head(path, epoch + 1, vali_loss)

            end_time = time.time()
            print(
                "Epoch: {}/{}, Time: {:.2f}, Train Loss: {:.4f}, Vali Loss: {:.4f}".format(
                    epoch + 1,
                    self.args.train_epochs,
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                )
            )

            if not min_vali_loss or vali_loss <= min_vali_loss:
                min_vali_loss = vali_loss
                save_full_checkpoint(path, "ckpt_best.pth", self.model, args=self.args, epoch=epoch)

            if (epoch + 1) % 10 == 0:
                save_full_checkpoint(path, f"ckpt{epoch + 1}.pth", self.model, args=self.args, epoch=epoch)

            model_scheduler.step()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        self._apply_linear_probe_freeze()
        model_optim = self._select_optimizer()
        model_criteria = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_loader = tqdm(train_loader, desc="Training")

            print(
                "Current learning rate: {:.7f}".format(
                    model_optim.param_groups[0]["lr"]
                )
            )

            self.model.train()
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(outputs, batch_y)
                loss.backward()
                model_optim.step()

                train_loss.append(loss.item())

                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

            train_loss = np.mean(train_loss)
            vali_loss = self.valid(vali_loader, model_criteria)
            test_loss = self.valid(test_loader, model_criteria)
            self._maybe_save_linear_probe_head(path, epoch + 1, vali_loss)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                    test_loss,
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}\n".format(
                        epoch + 1,
                        len(train_loader),
                        end_time - start_time,
                        train_loss,
                        vali_loss,
                        test_loss,
                    )
                )

            early_stopping(vali_loss, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))
        save_full_checkpoint(path, "finetune_best.pth", self.model, args=self.args)
        return self.model

    def valid(self, vali_loader, model_criteria):
        vali_loss = []
        self.model.eval()
        vali_loader = tqdm(vali_loader, desc="Validation")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(outputs, batch_y)
                vali_loss.append(loss.item())

        vali_loss = np.mean(vali_loss)
        self.model.train()
        return vali_loss

    def test(self):
        test_data, test_loader = self._get_data(flag="test")

        preds = []
        trues = []
        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                preds.append(outputs.detach().cpu())
                trues.append(batch_y.detach().cpu())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, _, _, _ = metric(preds, trues)
        print(
            "{0}->{1}, mse:{2:.3f}, mae:{3:.3f}".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        with open(folder_path + "/score.txt", "a") as f:
            f.write(
                "{0}->{1}, {2:.3f}, {3:.3f} \n".format(
                    self.args.input_len, self.args.pred_len, mse, mae
                )
            )
