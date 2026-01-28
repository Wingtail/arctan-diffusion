from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, show_series, show_matrix, save_full_checkpoint
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

warnings.filterwarnings('ignore')


class Exp_SimMTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_SimMTM, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self._linear_probe_best = float("inf")

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_head_module(self):
        model = self._unwrap_model()
        if hasattr(model, "head"):
            return model.head
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
        model = self.model_dict[self.args.model].Model(self.args).float()

        transfer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            model = transfer_weights(self.args.load_checkpoints, model, device=transfer_device)

        model = self._maybe_load_linear_probe_head(model, transfer_device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # show cases
        self.train_show = next(iter(train_loader))
        self.valid_show = next(iter(vali_loader))

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        #model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                     T_max=self.args.train_epochs)

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss, train_cl_loss, train_rb_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss, valid_cl_loss, valid_rb_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f} Val Loss: {6:.4f}/{7:.4f}/{8:.4f}"
                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,
                        train_rb_loss, vali_loss, valid_cl_loss, valid_rb_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_cl_loss': train_cl_loss,
                'train_rb_loss': train_rb_loss,
                'vali_loss': vali_loss,
                'valid_cl_loss': valid_cl_loss,
                'valid_rb_loss': valid_rb_loss,
            }

            # Avoid leading slash so tensorboardX writes to the relative logdir (./outputs/logs)
            self.writer.add_scalars("pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

                self.show(5, epoch + 1, 'train')
                self.show(5, epoch + 1, 'valid')

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []
        train_cl_loss = []
        train_rb_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            if self.args.select_channels < 1:

                # random select channels
                B, S, C = batch_x.shape
                random_c = int(C * self.args.select_channels)
                if random_c < 1:
                    random_c = 1

                index = torch.LongTensor(random.sample(range(C), random_c))
                batch_x = torch.index_select(batch_x, 2, index)

            # data augumentation
            batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm, self.args.positive_nums)
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            batch_x = batch_x.float().to(self.device)
            # batch_x_mark = batch_x_mark.float().to(self.device)

            # masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # to device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # encoder
            loss, loss_cl, loss_rb, _, _, _, _ = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

            # backward
            loss.backward()
            model_optim.step()

            # record
            train_loss.append(loss.item())
            train_cl_loss.append(loss_cl.item())
            train_rb_loss.append(loss_rb.item())

        model_scheduler.step()

        train_loss = np.average(train_loss)
        train_cl_loss = np.average(train_cl_loss)
        train_rb_loss = np.average(train_rb_loss)

        return train_loss, train_cl_loss, train_rb_loss

    def valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_cl_loss = []
        valid_rb_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            # data augumentation
            batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,
                                                          self.args.positive_nums)
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            # masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # to device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # encoder
            loss, loss_cl, loss_rb, _, _, _, _ = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

            # Record
            valid_loss.append(loss.item())
            valid_cl_loss.append(loss_cl.item())
            valid_rb_loss.append(loss_rb.item())

        vali_loss = np.average(valid_loss)
        valid_cl_loss = np.average(valid_cl_loss)
        valid_rb_loss = np.average(valid_rb_loss)

        self.model.train()
        return vali_loss, valid_cl_loss, valid_rb_loss

    def train(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        self._apply_linear_probe_freeze()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loader = tqdm(train_loader, desc="Training")

            # print lr
            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.select_channels < 1:
                    # Random select channels
                    B, S, C = batch_x.shape
                    random_c = int(C * self.args.select_channels)
                    if random_c < 1:
                        random_c = 1

                    index = torch.LongTensor(random.sample(range(C), random_c))
                    batch_x = torch.index_select(batch_x, 2, index)
                    batch_y = torch.index_select(batch_y, 2, index)

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # loss
                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                # record
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)
            self._maybe_save_linear_probe_head(path, epoch + 1, vali_loss)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        save_full_checkpoint(path, "finetune_best.pth", self.model, args=self.args)

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        vali_loader = tqdm(vali_loader)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                # loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                # record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f = open("./outputs/score.txt", 'a')
        f.write('{0}->{1}, {2:.3f}, {3:.3f} \n'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f.close()

        metrics_csv = os.path.join(folder_path, "metrics_simmtm.csv")
        header_needed = not os.path.exists(metrics_csv)
        with open(metrics_csv, "a") as f:
            if header_needed:
                f.write("input_len,pred_len,mse,mae\n")
            f.write(f"{self.args.input_len},{self.args.pred_len},{mse:.6f},{mae:.6f}\n")

    def show(self, num, epoch, type='valid'):

        # show cases
        if type == 'valid':
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.valid_show
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.train_show

        # data augumentation
        batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,
                                                      self.args.positive_nums)
        batch_x_om = torch.cat([batch_x, batch_x_m], 0)

        # masking matrix
        mask = mask.to(self.device)
        mask_o = torch.ones(size=batch_x.shape).to(self.device)
        mask_om = torch.cat([mask_o, mask], 0).to(self.device)

        # to device
        batch_x = batch_x.float().to(self.device)
        batch_x_om = batch_x_om.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        # Encoder
        with torch.no_grad():
            loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

        for i in range(num):

            if i >= batch_x.shape[0]:
                continue

        fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix = show_matrix(logits, positives_mask, rebuild_weight_matrix)
        self.writer.add_figure(f"/{type} show logits_matrix", fig_logits, global_step=epoch)
        self.writer.add_figure(f"/{type} show positive_matrix", fig_positive_matrix, global_step=epoch)
        self.writer.add_figure(f"/{type} show rebuild_weight_matrix", fig_rebuild_weight_matrix, global_step=epoch)
