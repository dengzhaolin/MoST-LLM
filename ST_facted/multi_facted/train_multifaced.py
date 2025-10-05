import sys


from MultiFacted_model import MultiFacedVAE
sys.path.append('../')
import os
import torch.optim as optim
import torch.nn as nn
import argparse
from lib.utils import data_gen, gen_batch, get_metric
from ST_facted.util import *
device_ids=[5,4]
def vae_masked_loss_function(recon_data, original_data, mask= None):
    recon_loss = nn.MSELoss()(recon_data, original_data)
    masked_recon_loss = (recon_loss * mask).sum() / mask.sum()
    return masked_recon_loss

def prepare_x_y(x, y):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x.to(device), y.to(device)


def multifaced_train(dataset, epochs=1000, lr=0.001, device=None):


    vae = MultiFacedVAE(4,4, hidden_dim=192).to(device)

    #vae = CrossST_pre(device).to(device)

    if torch.cuda.device_count() > 1:
        vae = nn.DataParallel(vae,device_ids=device_ids)


    optimizer = optim.Adam(vae.parameters(), lr=lr)

    temporal_epochs = 25
    spatial_epochs = 25
    agnostic_epochs = epochs - temporal_epochs - spatial_epochs
    save_dir = "./model"
    os.makedirs(save_dir, exist_ok=True)

    #data_loder = DataLoader(dataset.get_data('train'), args.batch_size, shuffle=True)

    spatial_points=50

    temporal_hours=8

    val_losses=[]

    for epoch in range(epochs):
        total_loss = 0
        vae.train()
        min_loss = float('inf')

        for j, x_batch in enumerate((gen_batch(dataset.get_data('train'), args.batch_size, dynamic_batch=True, shuffle=True))):#):
            batch_data = x_batch[:, 0:args.input_length]

            batch_data=torch.tensor(batch_data, dtype=torch.float32).to(device)

            optimizer.zero_grad()

            epoch_within_100 = epoch % 100
            if epoch_within_100 < temporal_epochs:
                masked_data, mask = apply_mask(batch_data, mask_type="temporal",temporal_hours=temporal_hours,spatial_points=spatial_points, return_mask=True)

                # if epoch_within_100 == 0:
                #     print("Entering Temporal Masking Strategy")

            elif epoch_within_100 < temporal_epochs + spatial_epochs:
                masked_data, mask = apply_mask(batch_data, mask_type="spatial", temporal_hours=temporal_hours,spatial_points=spatial_points,return_mask=True)
                # if epoch_within_100 == 25:
                #     print("Entering Spatial Masking Strategy")

            else:
                masked_data, mask = apply_mask(batch_data, mask_type="agnostic", temporal_hours=temporal_hours,spatial_points=spatial_points,return_mask=True)
                # if epoch_within_100 == 50:
                #     print("Entering Agnostic Masking Strategy")

            recon_data,_= vae(masked_data)

            loss = vae_masked_loss_function(recon_data, batch_data, mask=mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (j+1)}")

        if (epoch + 1) % 100 == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f'multi_modality_mask_{temporal_hours}_{spatial_points}_epoch_{epoch + 1}_modile.pth'))


    return vae


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='MoSSL', type=str, help='model name')
parser.add_argument('--data_name', default='NYC', type=str, help='NYC dataset')
parser.add_argument('--num_nodes', default=98, type=int, help='number of nodes')
parser.add_argument('--num_modals', default=4, type=int, help='number of modalities')
parser.add_argument('--input_length', default=16, type=int, help='input length')
parser.add_argument('--horizon', default=3, type=int, help='output length')
parser.add_argument('--indim', default=1, type=int, help='input dimension')
parser.add_argument('--num_comp', default=4, type=int, help='number of clusters')
parser.add_argument('--hidden_channels', default=48, type=int, help='number of hidden channels')
parser.add_argument('--batch_size', default=16, type=int, help='number of batch size')
parser.add_argument("--patience", default=15, type=int, help="patience used for early stop")
parser.add_argument("--lr", default=0.01, type=float, help="base learning rate")
parser.add_argument("--epsilon", default=1e-3, type=float, help="optimizer epsilon")
parser.add_argument("--steps", default=[50, 100], type=eval, help="steps")
parser.add_argument("--lr_decay_ratio", default=0.1, type=float, help="lr_decay_ratio")
parser.add_argument("--max_grad_norm", default=5, type=int, help="max_grad_norm")
parser.add_argument('--version', default=0, type=int, help='index of repeated experiments')
parser.add_argument('--cuda', default=5, help='cuda name')
args = parser.parse_args()
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
layers = int(np.log2(args.input_length))
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
def main():

    trained_model = multifaced_train(dataset, device=device)

if __name__ == '__main__':
    main()
