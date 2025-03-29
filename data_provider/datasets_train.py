from torch_geometric.data import Data, Dataset
from utils import *
import random
torch.manual_seed(42)
np.random.seed(42)
class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) 
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

class traffic_dataset(Dataset):

    def __init__(self, data_args, task_args, stage='Train', test_data='SDWPF134', Train_ratio=0.6):
        super(traffic_dataset, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.test_data = test_data
        self.Train_ratio = Train_ratio
        self.load_data(stage, test_data)

    def load_data(self, stage, test_data):
        self.x_list, self.y_list = {}, {}
        self.means_list, self.stds_list = {}, {}

        if stage == 'target' or stage == 'target_maml':
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        elif stage == 'valid':
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError('Error: Unsupported Stage')

        for dataset_name in self.data_list:
            if dataset_name=='SDWPF134':
                X = np.load(self.data_args[dataset_name]['dataset_path'])['data'][:,:,[0,2,3,5,6,7,9]]
                times = np.load(self.data_args[dataset_name]['dataset_path'])['times'][:,:,:].astype(np.float32)
            else:
                X = np.load(self.data_args[dataset_name]['dataset_path'])['data'][:,:,:]
                times = np.load(self.data_args[dataset_name]['dataset_path'])['times'].astype(np.float32)
            X = X.astype(np.float32)
            month_of_year = times[..., 0] - 1  # 0 ~ 11
            day_of_year = times[..., 1] - 1  # 0 ~ 365
            time_of_day = (times[..., 2] * 3600 + times[..., 3] * 60 + times[..., 4]) // 600
            X = np.concatenate([
                X,
                time_of_day.reshape([*time_of_day.shape, 1]),
                day_of_year.reshape([*day_of_year.shape, 1]),
                month_of_year.reshape([*month_of_year.shape, 1]),
            ],
                axis=-1
            )
            X = X.transpose((1, 2, 0))

            x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'])
            torch.manual_seed(42)
            np.random.seed(42)
            random_idx = np.random.permutation(x_inputs.shape[0])
            x_inputs = x_inputs[random_idx, ...].permute(1,2,0,3)
            y_outputs = y_outputs[random_idx, ...].permute(1,2,0)

            x_inputs_Train = x_inputs[:, :, :int(x_inputs.shape[2] * self.Train_ratio)]
            mean = x_inputs_Train[:, :, :, :7].mean(dim=(0, 1, 2), keepdim=True)
            std = x_inputs_Train[:, :, :, :7].std(dim=(0, 1, 2), keepdim=True)
            if stage == 'target' or stage == 'target_maml':
                # X_inputs = X_inputs[:, :, :288 * 6]
                x_inputs = x_inputs[:, :, :int(x_inputs.shape[2]*self.Train_ratio)]
                y_outputs = y_outputs[:, :, :int(y_outputs.shape[2]*self.Train_ratio)]
                x_inputs[:, :, :, :7] = (x_inputs[:, :, :, :7] - mean) / std
            elif stage == 'test':
                x_inputs = x_inputs[:, :, int(x_inputs.shape[2]*self.Train_ratio):int(x_inputs.shape[2]*0.8)]
                y_outputs = y_outputs[:, :, int(y_outputs.shape[2]*self.Train_ratio):int(y_outputs.shape[2]*0.8)]
                x_inputs[:, :, :, :7] = (x_inputs[:, :, :, :7] - mean) / std
            elif stage == 'valid':
                x_inputs = x_inputs[:, :, int(x_inputs.shape[2]*0.8): ]
                y_outputs = y_outputs[:, :, int(y_outputs.shape[2]*0.8): ]
                x_inputs[:, :, :, :7] = (x_inputs[:, :, :, :7] - mean) / std
            else:
                raise BBDefinedError('Error: Unsupported Stage')


            x_inputs = x_inputs.permute(2,0,1,3)
            y_outputs = y_outputs.permute(2,0,1)
            self.x_list[dataset_name] = x_inputs
            self.y_list[dataset_name] = y_outputs
            self.means_list[dataset_name] = mean[...,-1].squeeze().numpy()
            self.stds_list[dataset_name] = std[...,-1].squeeze().numpy()
            print("{},{}_data:{}, mean:{}, std:{}".format(self.data_list,stage,x_inputs.shape, self.means_list[dataset_name], self.stds_list[dataset_name]))

    def get(self, index):
        select_dataset = self.data_list[0]
        x_data = self.x_list[select_dataset][index: index+1]
        y_data = self.y_list[select_dataset][index: index+1]
        node_num = x_data.shape[1]
        data_i = Data(node_num=node_num, x=x_data, y=y_data)
        data_i.data_name = select_dataset
        return data_i
    
    def get_maml_task_batch(self):
        spt_task_data, qry_task_data = [], []
        select_dataset = random.choice(self.data_list)
        return spt_task_data, self.means_list[select_dataset], qry_task_data, self.stds_list[select_dataset]
    
    def len(self):
        return self.x_list[self.data_list[0]].shape[0]
