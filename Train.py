import pickle
import time
import matplotlib.pyplot as plt
from torch import optim
import pandas as pd
from Models.STFEB_WPF import *
from utils import *
from copy import deepcopy
from tqdm import tqdm
import torch.nn.functional as F
torch.manual_seed(42)
np.random.seed(42)

def loss_fn(y_pred, y_true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(y_true, mask_value)
        y_pred = torch.masked_select(y_pred, mask)
        y_true = torch.masked_select(y_true, mask)
    return F.smooth_l1_loss(y_pred, y_true, reduction='mean', beta=1.0)


class STMAML(nn.Module):

    def __init__(self, data_args, task_args, model_args, model='GRU', device='cuda:0'):
        super(STMAML, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.update_lr = model_args['update_lr']
        self.model_name = model

        if model == 'STFEB_WPF':
            self.model = STFEB_WPF()
            print("MAML Model: STFEB_WPF")
        self.model.to(device)
        print("model params: ", count_parameters(self.model))
        self.loss_criterion = loss_fn

    def forward(self, data, matrix):
        out, meta_graph = self.model(data, matrix)
        return out, meta_graph

    def finetuning(self, target_dataloader, valid_dataloader, test_dataloader, target_epochs, scaler, device):
        maml_model = deepcopy(self.model)
        optimizer = optim.AdamW(maml_model.parameters(), lr=self.update_lr, weight_decay=10 * 1e-4)
        min_MAE = 10000000
        bestid = 1
        patience = 0
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        vi_train_losses = []
        vi_valid_losses = []
        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            train_losses1 = []
            valid_losses = []
            start_time = time.time()
            maml_model.train()
            for step, data in enumerate(target_dataloader):
                x = data.x.to(device=device)
                y = data.y.to(device=device)
                out, query, pos, neg = maml_model(x)#out shape:[batchsize, nodenum, predLong]
                out = scaler.inverse_transform(out)

                loss = self.loss_criterion(out, y)
                separate_loss = nn.TripletMarginLoss(margin=1.0)
                compact_loss = nn.MSELoss()
                loss2 = separate_loss(query, pos, neg)
                loss3 = compact_loss(query, pos)
                loss1 = loss + 2 * loss2 + 2 * loss3
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_losses1.append(loss1.item() - loss.item())
            avg_train_loss = np.mean(train_losses)  # sum(train_losses)/len(train_losses)
            avg_train_loss1 = np.mean(train_losses1)
            vi_train_losses.append(avg_train_loss)

            maml_model.eval()
            for step, data in enumerate(valid_dataloader):
                with torch.no_grad():
                    x = data.x.to(device=device)
                    y = data.y.to(device=device)
                    # out = maml_model(x)
                    out, query, pos, neg = maml_model(x)
                    out = scaler.inverse_transform(out)
                    loss = self.loss_criterion(out, y)
                    valid_losses.append(loss.item())
            avg_valid_loss = np.mean(valid_losses)  # sum(valid_losses) / len(valid_losses)
            vi_valid_losses.append(avg_valid_loss)
            lr_scheduler.step()
            if min_MAE > avg_valid_loss:
                patience = 0
                if os.path.exists('save/' + str(self.model_name) + str(self.model_args['update_lr']) + '_bs_' + str(
                        target_dataloader.batch_size) + 'pred_num' + str(
                    self.task_args['pred_num']) + '_finetuning_epoch_' + str(bestid) + '_' + str(
                    round(min_MAE, 2)) + '.pth'):
                    # 删除之前的最佳模型
                    os.remove('save/' + str(self.model_name) + str(self.model_args['update_lr']) + '_bs_' + str(
                        target_dataloader.batch_size) + 'pred_num' + str(
                        self.task_args['pred_num']) + '_finetuning_epoch_' + str(bestid) + '_' + str(
                        round(min_MAE, 2)) + '.pth')

                torch.save(maml_model.state_dict(), 'save/' +
                           str(self.model_name) + str(self.model_args['update_lr']) + '_bs_' + str(
                    target_dataloader.batch_size) + 'pred_num' + str(
                    self.task_args['pred_num']) + '_finetuning_epoch_' + str(epoch) + '_' + str(
                    round(avg_valid_loss, 2)) + '.pth')
                bestid = epoch
                min_MAE = avg_valid_loss
            else:
                patience += 1
                if patience >= 20:
                    break
            end_time = time.time()
            if epoch % 10 == 0 or epoch < 50:
                print("epoch #{}/{}: loss: {}, Bank loss:{}, avg_valid_loss: {},patience:{}, time={}S".format(epoch + 1,
                                                                                                              target_epochs,
                                                                                                              round(
                                                                                                                  avg_train_loss,
                                                                                                                  2),
                                                                                                              round(
                                                                                                                  avg_train_loss1,
                                                                                                                  5),
                                                                                                              round(
                                                                                                                  avg_valid_loss,
                                                                                                                  2),
                                                                                                              patience,
                                                                                                              start_time - end_time))
        maml_model.load_state_dict(
            torch.load('save/' + str(self.model_name) + str(self.model_args['update_lr']) + '_bs_' + str(
                target_dataloader.batch_size) + 'pred_num' + str(
                self.task_args['pred_num']) + '_finetuning_epoch_' + str(bestid) + '_' + str(
                round(min_MAE, 2)) + '.pth'))

        maml_model.eval()
        with torch.no_grad():
            test_start = time.time()
            outputs, realy = [], []
            test_result = pd.DataFrame(columns=['Horizon', 'MAE', 'RMSE', 'MAPE'])
            for step, data in enumerate(test_dataloader):
                with torch.no_grad():
                    x = data.x.to(device=device)
                    y = data.y.to(device=device)
                    data.node_num = data.node_num
                    batch_size, node_num, seq_len, _ = x.shape
                    out, query, pos, neg = maml_model(x)  # 256，134，12
                    if step == 0:
                        outputs = out
                        y_label = y
                    else:
                        outputs = torch.cat((outputs, out))  # 10494，134，12
                        y_label = torch.cat((y_label, y))
            test_loss = []
            test_mape = []
            test_rmse = []
            res = []
            for k in range(12):
                pred = scaler.inverse_transform(outputs[:, :, k])
                real = y_label[:, :, k]
                metrics = metric(pred, real)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(k + 1, metrics[0], metrics[2], metrics[1]))
                test_result = test_result.append(
                    {'Horizon': k + 1, 'MAE': round(metrics[0], 4), 'RMSE': round(metrics[2], 4),
                     'MAPE': round(metrics[1], 4)}, ignore_index=True)
                test_loss.append(metrics[0])
                test_mape.append(metrics[1])
                test_rmse.append(metrics[2])
                if k in [2, 5, 11]:
                    res += [metrics[0], metrics[2], metrics[1]]
            mtest_loss = np.mean(test_loss)
            mtest_mape = np.mean(test_mape)
            mtest_rmse = np.mean(test_rmse)

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            print(log.format(mtest_loss, mtest_rmse, mtest_mape))
            res += [mtest_loss, mtest_rmse, mtest_mape]
            res = [round(r, 4) for r in res]
            print(res)

            test_end = time.time()

            # result_print(result, info_name='Evaluate')
            print("[Target Test] testing time is {}".format(test_end - test_start))
            with open('save/lossesMydata_36.pkl', 'wb') as f:
                pickle.dump({'train_losses': vi_train_losses, 'valid_losses': vi_valid_losses}, f)
            print("loss config saved in lossesSDWPF36.pkl")
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(vi_train_losses) + 1), vi_train_losses, label='Training Loss', linewidth=2.5)
            plt.plot(range(1, len(vi_valid_losses) + 1), vi_valid_losses, label='Validation Loss', linewidth=2.5)
            plt.xlabel('Epoch', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.legend(prop={'size': 15},
                       handlelength=2,
                       columnspacing=1.8)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.show()
