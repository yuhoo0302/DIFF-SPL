import os
import sys
import time
import shutil
import argparse

sys.path.append('./source')

import wandb
import torch

from torch.utils.data import DataLoader

from models import load_model
from data_func import load_dataset
from plane_func import Slicer, tangent2plane
from help_func import read_yaml, init_seed, init_folder, Recorder


# This only be used when debug the code
# The wandb won't upload the log into the server
# os.environ['WANDB_MODE'] = 'offline'


def train_loop(model, optimizer, dataset):
    model.train()

    volume, tangent = dataset.pop_data()
    slicer = Slicer(im=volume, out_size=dataset.plane_size, device=model.model_device)

    fwd = model(tangent, slicer, dataset.batch_size)
    optimizer.zero_grad()

    loss_total, loss_info, print_info = model.loss_func(fwd)

    loss_total.backward()
    optimizer.step()

    return loss_info, print_info


def val_loop(model, dataset, sess, iteration):
    model.eval()
    recorder = Recorder()
    for idx in range(dataset.num):
        volume, tangent, _ = dataset.pop_data_idx(idx)
        slicer = Slicer(im=volume, out_size=dataset.plane_size, device=model.model_device)

        with torch.no_grad():
            eval_dict = model.evaluate(tangent, slicer, dataset.batch_size)
            recorder.update(eval_dict)

    sess.log(recorder.average(), step=iteration)
    val_print = ' '.join([f'{key}: {value:.3f}' for key, value in recorder.average().items()])
    return val_print, recorder.average()['TangentDistance']


def main(cfg):
    session = wandb.init(project="DiffusionSPL",
                         config=cfg,
                         name=os.path.basename(cfg['SavePath']))

    device = torch.device('cuda:{}'.format(cfg['GPUId'])) if torch.cuda.is_available() else torch.device('cpu')

    init_seed(cfg['Seed'])

    dataset_dict = load_dataset(cfg)
    train_set = dataset_dict['Train']
    val_set = dataset_dict['Val']

    model = load_model(cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LearningRate'], weight_decay=cfg['WeightDecay'])

    if cfg['InitWeight']:
        model.load_weight(cfg['InitWeight'])
        print('>>> Load from', cfg['InitWeight'])

    open(os.path.join(cfg['SavePath'], 'Logs', 'record.txt'), 'w+').close()

    best_metric = 100.

    for iteration in range(cfg['NumIterations']):
        loss_info, print_info = train_loop(model, optimizer, train_set)
        session.log(loss_info, step=iteration)
        info = f'Iteration {iteration:05d} >>> {print_info}'
        print(info)
        if iteration % cfg['ValFreq'] == 0:

            val_info, val_metric = val_loop(model, val_set, session, iteration)

            info = f'Iteration {iteration:05d} >>> {print_info} {val_info}'
            print(info)
            open(os.path.join(cfg['SavePath'], 'Logs', 'record.txt'), 'a+').write(f'{info}\n')

            if val_metric <= best_metric:
                torch.save(model.state_dict(), os.path.join(cfg['SavePath'], 'Weights', 'model_best.pth.gz'))
                best_metric = val_metric
                best_iteration = iteration
                info = f'Save best model with metric of {best_metric:.3f} in epoch #{best_iteration:03d}'

                print(info)
                open(os.path.join(cfg['SavePath'], 'Logs', 'record.txt'), 'a+').write(f'{info}-\n')

            if not cfg['EfficientSave']:
                torch.save(model.state_dict(), os.path.join(cfg['SavePath'], 'Weights', f'model_{iteration}.pth.gz'))

    torch.save(model.state_dict(), os.path.join(cfg['SavePath'], 'Weights', 'model_final.pth.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='./configure/Prototyping.yaml')
    args = parser.parse_args()

    config = read_yaml([args.config_path])
    init_folder(config['SavePath'])
    shutil.copy(args.config_path, os.path.join(config['SavePath'], 'config.yaml'))

    main(config)
