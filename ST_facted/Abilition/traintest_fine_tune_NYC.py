import os
import time
from datetime import datetime
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from lib.utils import data_gen, gen_batch, get_metric

from model_tune_2 import ModalMoE

from model_tune_2 import LLMFineTuner ,Prediction


from ST_facted.signal_facted.SignalFacted_model_AUTO import SignalFactedVAE
from ST_facted.multi_facted.MultiFacted_model import MultiFacedVAE



# 在程序开始处


model_name = "/data/dzl2023/LLM-Factory/Qwen-2.5-1.5B"
device_ids=[4]




def predictModel(fine_tuner,pre_trained_vae,prediction_head, seq, dynamic_batch=True):
    fine_tuner.eval()
    prediction_head.eval()
    pre_trained_vae.eval()
    pred_list = []
    for i in gen_batch(seq, min(args.batch_size, len(seq[0])), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:args.input_length, :, :, :])
        step_list = []
        test_seq_th = torch.tensor(test_seq, dtype=torch.float32).to(device)

        with torch.no_grad():

            batch_size = test_seq_th.shape[0]

            rep = test_seq_th.squeeze(-1).permute(0, 2, 1, 3).reshape(batch_size, args.num_nodes, -1)

            rep = fine_tuner.module.embedding_to_hidden(rep)

            rep = fine_tuner(rep).view(batch_size, -1, args.num_modals, args.num_nodes).unsqueeze(-1)

            pred = prediction_head(rep)

            pred = pred.permute(0, 1, 3, 2, 4)

        pred = pred.data.cpu().numpy()
        pred_list.append(pred)
    pred_array = np.concatenate(pred_list, axis=0)
    return pred_array, pred_array.shape[0]


def modelInference(fine_tuner,pre_trained_vae,prediction_head, inputs):
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()
    if args.input_length + args.horizon > x_val[0].shape[0]:
        raise ValueError(f'ERROR: the value of horizon "{args.horizon}" exceeds the length limit.')
    # evaluation
    y_val, len_val = predictModel(fine_tuner,pre_trained_vae,prediction_head, x_val)
    mae_val, rmse_val = get_metric(x_val[0:len_val, args.input_length:args.horizon + args.input_length, :, :, :],
                                   y_val[:, :, :, :, :], x_stats)
    # test
    y_test, len_test = predictModel(fine_tuner,pre_trained_vae,prediction_head,x_test)
    mae_test, rmse_test = get_metric(x_test[0:len_test, args.input_length:args.horizon + args.input_length, :, :, :],
                                     y_test[:, :, :, :, :], x_stats)
    return mae_val, rmse_val, mae_test, rmse_test


def traintest_model(dataset):

    fine_tuner,prediction_head,pre_trained_vae = get_model()

    save_dir = "./model/fine_tune_aux"
    os.makedirs(save_dir, exist_ok=True)
    print('=' * 10)
    print("training and testing model...")
    criterion = nn.MSELoss()
    if torch.cuda.device_count() > 1:
        fine_tuner = nn.DataParallel(fine_tuner, device_ids=device_ids)
        prediction_head = nn.DataParallel(prediction_head, device_ids=device_ids)
        pre_trained_vae = nn.DataParallel(pre_trained_vae, device_ids=device_ids)
        # 注意：即使只有一个GPU，我们也要将模型移到设备上
    else:
        fine_tuner = fine_tuner.to(device)
        prediction_head = prediction_head.to(device)
        pre_trained_vae = pre_trained_vae.to(device)

    optimizer = optim.Adam([
        {'params': fine_tuner.parameters(), 'lr': 0.0001},
        {'params': prediction_head.parameters(), 'lr': 0.01},
        # {'params': cross_model.parameters(), 'lr': 0.001}
    ])


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)


    min_rmse = float('inf')

    nb_epoch = 120
    for epoch in range(nb_epoch):
        start_time = time.time()
        fine_tuner.train()
        prediction_head.train()
        pre_trained_vae.eval()
        losses = []
        for j, x_batch in enumerate(gen_batch(dataset.get_data('train'), args.batch_size, dynamic_batch=True, shuffle=True)):

            optimizer.zero_grad()
            x = x_batch[:, 0:args.input_length]
            y = x_batch[:, args.input_length:args.input_length + args.horizon, :, :, :]
            x, y = prepare_x_y(x, y)

            batch_size = x.shape[0]

            rep = x.squeeze(-1).permute(0, 2, 1, 3).reshape(batch_size, args.num_nodes, -1)


            rep = fine_tuner.module.embedding_to_hidden(rep)

            rep = fine_tuner(rep)

            rep=rep.view(batch_size, -1, args.num_modals, args.num_nodes).unsqueeze(-1)


            pred= prediction_head(rep)

            pred=pred.permute(0, 1, 3, 2, 4)


            loss = criterion(pred, y)

            print("loss:", loss.item())

            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']],
                args.max_grad_norm
            )  # gradient clipping - this does it in place

        train_loss = np.mean(losses)

        lr_scheduler.step()
        end_time = time.time()
        mae_val, rmse_val, mae_test, rmse_test = modelInference(fine_tuner,pre_trained_vae,prediction_head, dataset)
        print('=' * 80)
        print('Epoch {}: train_loss: {:.3f} ; lr: {:.5f}; {:.1f}s'.format(
            epoch, train_loss,  optimizer.param_groups[0]['lr'],
            (end_time - start_time)))
        for i in range(args.num_modals):
            print('Modality {}:'.format(i))
            print('Horizon 4 Hour| MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                  .format(mae_val[3, i], mae_test[3, i], rmse_val[3, i], rmse_test[3, i]))
            print('Horizon 8 Hour| MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                  .format(mae_val[7, i], mae_test[7, i], rmse_val[7, i], rmse_test[7, i]))
            print('Horizon 16 Hour| MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                  .format(mae_val[15, i], mae_test[15, i], rmse_val[15, i], rmse_test[15, i]))
        total_rmse = rmse_val.sum()
        if total_rmse < min_rmse:
            print('Toal RMSE decrease from {:.2f} to {:.2f} ({:.2f})'.format(min_rmse, total_rmse,
                                                                             (min_rmse- total_rmse)))
            fine_tuner.eval()
            torch.save(fine_tuner.state_dict(), os.path.join(save_dir, f'fine_tune.pth'))
            torch.save(prediction_head.state_dict(), os.path.join(save_dir, f'prediction_head.pth'))
            min_rmse= total_rmse

    fine_tuner.load_state_dict(torch.load(os.path.join(save_dir, f'fine_tune.pth')))
    prediction_head.load_state_dict(torch.load(os.path.join(save_dir, f'prediction_head.pth')))
    mae_val, rmse_val, mae_test, rmse_test = modelInference(fine_tuner,pre_trained_vae,prediction_head,dataset)
    print('=' * 20 + 'Best model performance' + '=' * 20)
    for i in range(args.num_modals):
        print('Modality {}:'.format(i))
        print('Horizon 4 Hour| MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mae_val[3, i], mae_test[3, i], rmse_val[3, i], rmse_test[3, i]))
        print('Horizon 8 Hour| MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mae_val[7, i], mae_test[7, i], rmse_val[7, i], rmse_test[7, i]))
        print('Horizon 16 Hour| MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mae_val[15, i], mae_test[15, i], rmse_val[15, i], rmse_test[15, i]))

def get_model():
    fine_tuner = LLMFineTuner(model_name, model_name, device).to(device)


    prediction_head=ModalMoE().to(device)

    #prediction_head=EventMoE().to(device)
    #prediction_head=Prediction(384,16).to(device)



    pre_trained_multi_vae = MultiFacedVAE(4, 4, 192).to(device)

    pre_trained_model_path = "/data/dzl2023/MoSSL/ST_facted/multi_facted/MODEL/MoSSL_NYC_num_comp4_hc48_l4_his16_pred3_v0.h5"

    pretrained_dict = torch.load(pre_trained_model_path, map_location=device)

    pre_trained_multi_vae.load_state_dict(pretrained_dict, strict=False)

    pred_trained_signal_vae = SignalFactedVAE(pre_trained_multi_vae.encoder,args.num_modals, args.hidden_channels).to(device)

    pre_trained_signal_path="/data/dzl2023/MoSSL/ST_facted/signal_facted/MODEL/SignalFactedVAE/SignalFactedVAE_NYC.h5"

    pred_trained_signal_dict = torch.load(pre_trained_signal_path, map_location=device)

    pred_trained_signal_vae.load_state_dict(pred_trained_signal_dict, strict=False)


    #

    return fine_tuner,prediction_head, pred_trained_signal_vae


def prepare_x_y(x, y):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x.to(device), y.to(device)


# Params #
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='MoSSL', type=str, help='model name')
parser.add_argument('--data_name', default='NYC', type=str, help='NYC dataset')
parser.add_argument('--num_nodes', default=98, type=int, help='number of nodes')
parser.add_argument('--num_modals', default=4, type=int, help='number of modalities')
parser.add_argument('--input_length', default=16, type=int, help='input length')
parser.add_argument('--horizon', default=16, type=int, help='output length')
parser.add_argument('--indim', default=1, type=int, help='input dimension')
parser.add_argument('--num_comp', default=4, type=int, help='number of clusters')
parser.add_argument('--hidden_channels', default=48, type=int, help='number of hidden channels')
parser.add_argument('--batch_size', default=16, type=int, help='number of batch size')
parser.add_argument("--patience", default=15, type=int, help="patience used for early stop")
parser.add_argument("--lr", default=0.01, type=float, help="base learning rate")
parser.add_argument("--epsilon", default=1e-3, type=float, help="optimizer epsilon")
parser.add_argument("--steps", default=[50,100], type=eval, help="steps")
parser.add_argument("--lr_decay_ratio", default=0.1, type=float, help="lr_decay_ratio")
parser.add_argument("--max_grad_norm", default=5, type=int, help="max_grad_norm")
parser.add_argument('--version', default=0, type=int, help='index of repeated experiments')
parser.add_argument('--cuda', default=4, type=int, help='cuda name')
args = parser.parse_args()
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
layers = int(np.log2(args.input_length))
cpu_number = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_number)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_number)
os.environ['MKL_NUM_THREADS'] = str(cpu_number)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_number)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_number)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_num_threads(cpu_number)
###########################################################
print('=' * 10)
print('| Model: {0} | Dataset: {1} | History: {2} | Horizon: {3}'.format(args.model_name, args.data_name,
                                                                         args.input_length, args.horizon))
print("version: ", args.version)
print("number of clusters: ", args.num_comp)
print("channel in: ", args.indim)
print("hidden channels: ", args.hidden_channels)
print("layers: ", layers)
# load data
print('=' * 10)
print("loading data...")
if args.data_name == 'NYC':
    n_train, n_val, n_test = 81, 5, 5
    args.num_nodes = 98
    args.num_modals = 4
    dataset = data_gen('/data/dzl2023/MoSSL/data/NYC.h5', (n_train, n_val, n_test), args.num_nodes, args.input_length + args.horizon,
                       args.num_modals, day_slot=48)




#######################################
def main():
    print('=' * 10)
    print("compiling model...")
    print('=' * 10)
    print("init model...")
    start = time.time()
    traintest_model(dataset)
    end = time.time()
    print('Total running {:.1f} hours.'.format((end - start) / 3600))


if __name__ == '__main__':
    main()