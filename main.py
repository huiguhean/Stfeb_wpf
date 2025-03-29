import random
from torch_geometric.loader import DataLoader
from data_provider.datasets_train import traffic_dataset
from utils import *
import argparse
import yaml
import time
from Train import STMAML
from data_provider.normalization import StandardScaler
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='MAML-based')
parser.add_argument('--config_filename', default='config.yaml', type=str,
                    help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='SDWPF134', type=str, help='MyData,SDWPF134')
parser.add_argument('--target_epochs', default=120, type=int)
parser.add_argument('--update_lr', default=0.0015, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--Train_ratio', default=0.6, type=int)
parser.add_argument('--model', default='STFEB_WPF', type=str,
                    help='TransformerEnOnly,MLP,CNN_LSTM,AGCRN,STAEformer，model2D， ASFB_TFB,')
# parser.add_argument('--loss_lambda', default=1.5, type=float)
parser.add_argument('--memo', default='revise', type=str)
args = parser.parse_args()

print(time.strftime('%Y-%m-%d %H:%M:%S'), "Train_ratio = ", args.Train_ratio)

if __name__ == '__main__':

    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        print("INFO: GPU," + str(args.device))
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    model_args['update_lr'] = args.update_lr  # 梯度下降学习率
    task_args['batch_size'] = args.batch_size

    model = STMAML(data_args, task_args, model_args, model=args.model, device=args.device).to(device=args.device)
    print(args)

    target_dataset = traffic_dataset(data_args, task_args, "target", test_data=args.test_dataset,
                                     Train_ratio=args.Train_ratio)
    target_dataloader = DataLoader(target_dataset, batch_size=task_args['batch_size'], shuffle=True, drop_last=True)#,
                                   #num_workers=4, pin_memory=True)
    valid_dataset = traffic_dataset(data_args, task_args, "valid", test_data=args.test_dataset,
                                    Train_ratio=args.Train_ratio)
    valid_dataloader = DataLoader(valid_dataset, batch_size=task_args['test_batch_size'], shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True)
    test_dataset = traffic_dataset(data_args, task_args, "test", test_data=args.test_dataset,
                                   Train_ratio=args.Train_ratio)
    test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=True, drop_last=True,
                                 num_workers=4, pin_memory=True)
    print("批次大小 (target_dataloader):", target_dataloader.batch_size)
    num_batches = 0
    for _ in target_dataloader:
        num_batches += 1
    print("target_dataloader批次总数:", num_batches)
    num_batches = 0
    for _ in test_dataloader:
        num_batches += 1
    print("test_dataloader批次总数:", num_batches)

    _, target_mean, _, target_std = target_dataset.get_maml_task_batch()
    target_mean = torch.from_numpy(target_mean).to(args.device)
    target_std = torch.from_numpy(target_std).to(args.device)
    scaler = StandardScaler(target_mean, target_std)

    model.finetuning(target_dataloader, valid_dataloader, test_dataloader, args.target_epochs, scaler, args.device)  # , args.test_dataset)

    print(args.memo)
    print(time.strftime('%Y-%m-%d %H:%M:%S'), "Train_ratio = ", args.Train_ratio)
    print(args)
