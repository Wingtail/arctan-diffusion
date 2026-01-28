from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    transfer_weights,
    show_series,
    show_matrix,
    visual,
    save_full_checkpoint,
)
from utils.augmentations import masked_data
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
from tensorboardX import SummaryWriter
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


class Exp_TimeDART(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeDART, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self._linear_probe_best = float("inf")

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_head_module(self):
        model = self._unwrap_model()
        if hasattr(model, "head"):
            return model.head
        if hasattr(model, "projection"):
            return model.projection
        return None

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

    def _maybe_load_linear_probe_head(self, model, device):
        head_path = getattr(self.args, "linear_probe_head_path", None)
        if not head_path:
            return model
        if not os.path.exists(head_path):
            raise FileNotFoundError(f"Linear-probe head not found: {head_path}")
        head = getattr(model, "head", None)
        if head is None:
            raise ValueError("Model has no head to load linear-probe weights.")
        state = torch.load(head_path, map_location=device)
        state_dict = state.get("state_dict", state)
        missing, unexpected = head.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"Linear-probe head load: missing={missing} unexpected={unexpected}"
            )
        print(f"Loaded linear-probe head from: {head_path}")
        return model

    def _build_model(self):
        if self.args.downstream_task == "forecast":
            model = self.model_dict[self.args.model].Model(self.args).float()
        elif self.args.downstream_task == "classification":
            model = self.model_dict[self.args.model].ClsModel(self.args).float()

        transfer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            model = transfer_weights(
                self.args.load_checkpoints, model, device=transfer_device
            )

        model = self._maybe_load_linear_probe_head(model, transfer_device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
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
        if self.args.task_name == "finetune" and self.args.downstream_task == "classification":
            criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss")
        else:
            criterion = nn.MSELoss()
            print("Using MSELoss")
        return criterion

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        # model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        # model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,T_max=self.args.train_epochs)
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model_optim, gamma=self.args.lr_decay
        )

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            # current learning rate
            print("Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0]))

            train_loss = self.pretrain_one_epoch(
                train_loader, model_optim, model_scheduler
            )
            vali_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
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

            loss_scalar_dict = {
                "train_loss": train_loss,
                "vali_loss": vali_loss,
            }

            # Avoid leading slash so logs stay inside ./outputs/logs instead of trying to create /pretrain_loss
            self.writer.add_scalars("pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model epoch{}...".format(
                        min_vali_loss, vali_loss, epoch
                    )
                )
                min_vali_loss = vali_loss

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if "encoder" in k or "enc_embedding" in k:
                        if "module." in k:
                            k = k.replace("module.", "")  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.encoder_state_dict,
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if "encoder" in k or "enc_embedding" in k:
                        if "module." in k:
                            k = k.replace("module.", "")
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.encoder_state_dict,
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        train_loss = []
        model_criterion = self._select_criterion()

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
        ):
            model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            pred_x = self.model(batch_x)
            diff_loss = model_criterion(pred_x, batch_x)
            diff_loss.backward()

            model_optim.step()
            train_loss.append(diff_loss.item())

        model_scheduler.step()
        train_loss = np.mean(train_loss)

        return train_loss

    def valid_one_epoch(self, vali_loader):
        vali_loss = []
        model_criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)
                diff_loss = model_criterion(pred_x, batch_x)
                vali_loss.append(diff_loss.item())

        vali_loss = np.mean(vali_loss)

        return vali_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        self._apply_linear_probe_freeze()

        # optimizer
        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loader = tqdm(train_loader, desc="Training")

            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))

            self.model.train()
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(pred_x, batch_y)
                loss.backward()
                model_optim.step()
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                train_loss.append(loss.item())

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

        self.lr = model_scheduler.get_last_lr()[0]

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

                pred_x = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                loss = model_criteria(pred_x, batch_y)
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

                pred_x = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)

        if len(preds) == 0:
            raise RuntimeError(
                "No test batches were produced. Check dataset length and loader settings."
            )

        # Concatenate along batch dimension to avoid ragged/object arrays.
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        mae, mse, _, _, _ = metric(preds, trues)
        print(
            "{0}->{1}, mse:{2:.3f}, mae:{3:.3f}".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        # append plain text log (kept for backward compatibility)
        with open(folder_path + "/score.txt", "a") as f:
            f.write(
                "{0}->{1}, {2:.3f}, {3:.3f} \n".format(
                    self.args.input_len, self.args.pred_len, mse, mae
                )
            )

        # tidy CSV summary (creates header once)
        metrics_csv = os.path.join(folder_path, "metrics_timedart.csv")
        header_needed = not os.path.exists(metrics_csv)
        with open(metrics_csv, "a") as f:
            if header_needed:
                f.write("input_len,pred_len,mse,mae\n")
            f.write(f"{self.args.input_len},{self.args.pred_len},{mse:.6f},{mae:.6f}\n")

        # save raw preds/trues for later inspection
        np.savez(
            os.path.join(folder_path, f"preds_pl{self.args.pred_len}.npz"),
            preds=preds,
            trues=trues,
        )

        # quick visualization of first sample / first dimension
        try:
            visual(
                trues[0, :, 0],
                preds[0, :, 0],
                name=os.path.join(folder_path, f"plot_pl{self.args.pred_len}_idx0_dim0.png"),
            )
        except Exception as e:
            print(f"Plotting failed: {e}")

    def cls_train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        self._apply_linear_probe_freeze()
        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_acc = []
            train_f1 = []

            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))

            self.model.train()
            train_loader = tqdm(train_loader)
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                loss.backward()
                model_optim.step()
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')

                train_loss.append(loss.item())
                train_acc.append(acc)
                train_f1.append(f1)

            train_loss = np.mean(train_loss)
            train_acc = np.mean(train_acc)
            train_f1 = np.mean(train_f1)

            vali_loss, vali_acc, vali_f1 = self.cls_valid(vali_loader, model_criteria)
            test_loss, test_acc, test_f1 = self.cls_valid(test_loader, model_criteria)
            self.cls_test(write_log=False)
            self._maybe_save_linear_probe_head(path, epoch + 1, vali_loss)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | ".format(
                    epoch + 1, len(train_loader), end_time - start_time
                ) +
                "Train Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f} | ".format(
                    train_loss, train_acc, train_f1
                ) +
                "Vali Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f} | ".format(
                    vali_loss, vali_acc, vali_f1
                ) +
                "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}".format(
                    test_loss, test_acc, test_f1
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f}, Acc: {4:.4f}, F1: {5:.4f} | Vali Loss: {6:.7f}, Acc: {7:.4f}, F1: {8:.4f} | Test Loss: {9:.7f}, Acc: {10:.4f}, F1: {11:.4f}\n".format(
                        epoch + 1, len(train_loader), end_time - start_time, train_loss, train_acc, train_f1, vali_loss, vali_acc, vali_f1, test_loss, test_acc, test_f1
                    )
                )

            early_stopping(-vali_acc, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        save_full_checkpoint(path, "finetune_best.pth", self.model, args=self.args)
        return self.model

    def cls_valid(self, vali_loader, model_criteria):
        vali_acc = []
        vali_f1 = []
        vali_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                vali_loss.append(loss.item())
                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')
                vali_acc.append(acc)
                vali_f1.append(f1)
        
        vali_loss = np.mean(vali_loss)
        vali_acc = np.mean(vali_acc)
        vali_f1 = np.mean(vali_f1)

        return vali_loss, vali_acc, vali_f1

    def cls_test(self, write_log=True):
        test_data, test_loader = self._get_data(flag="test")
        model_criteria = self._select_criterion()

        preds_all = []
        trues_all = []
        test_loss = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                test_loss.append(loss.item())
                preds_all.extend(preds)
                trues_all.extend(trues)

        test_loss = np.mean(test_loss)
        test_acc = accuracy_score(trues_all, preds_all)
        test_f1 = f1_score(trues_all, preds_all, average='macro')

        print(
            "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}".format(
                test_loss, test_acc, test_f1
            )
        )
        if write_log:
            f = open(folder_path + "/score.txt", "a")
            f.write(
                "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}\n".format(test_loss, test_acc, test_f1)
            )
            f.close()
