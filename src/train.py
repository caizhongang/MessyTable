from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import time
from config import parse_config
import os
from data import *
from model import *
from utils import *


def main(args):
    print('Configuration file in', args.config_dir)

    config = parse_config(args)

    np.random.rand(0)

    tic = time.time()
    train_samples, train_scenes = globals()[config['split_samples_func']](config)
    print('time spent in sample spliting {:0.1f}'.format(time.time() - tic))
    device = torch.device('cuda')
    config['device'] = device
    print('gpu count', torch.cuda.device_count())

    train_set = MessyTableTripletDataset(train_samples, config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'])

    model = globals()[config['model_class']](config)
    if args.resume:
        model_pathname = os.path.join(config['config_dir'], '{}_{}.pth'.format(config['model_class'],args.resume))
        model.load_state_dict(torch.load(os.path.join(model_pathname)))
        print('Resuming from', model_pathname)
    model.to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_func = config['loss_func']
    triplet_margin = config['triplet_margin']
    num_epoch = config['num_epoch']

    loss_function = globals()[loss_func](config).to(device)

    model.train()

    scheduler = MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])

    if args.resume:
        scheduler.step(args.resume + 1)

    print('Training starts...{:0.1f}'.format(time.time() - tic))
    tic = time.time()
    # epoch
    for i in range(args.resume + 1 if args.resume else 0, num_epoch):

        epoch_start = time.time()

        epoch_loss = 0
        train_loader_cnt = 0

        for sample in train_loader:

            config['curr_cnt'] = i * len(train_loader) + train_loader_cnt
            optimizer.zero_grad()

            if config['mode'] == 'Triplet':
                sample_dict = {
                    'a': sample['a_crop'][0].to(device),
                    'p': sample['p_crop'][0].to(device),
                    'n': sample['n_crop'][0].to(device)
                }
            elif config['mode'] in ['Siamese', 'PatchMatching']:
                sample_dict = {
                    'a': sample['main_crop'][0].to(device),
                    'b': sample['sec_crop'][0].to(device),
                    'labels': sample['if_same_subcls'][0].to(device)
                }
            else:
                raise ValueError('Undefined mode:', mode)

            output_dict = model(sample_dict)
            loss = loss_function(output_dict, sample_dict)

            loss.backward()
            optimizer.step()

            iter_loss = loss.item()
            epoch_loss += iter_loss
            if train_loader_cnt % 10 == 0:
                delta_t = time.time() - tic
                tic = time.time()
                print('Epoch {}: Iter {} / {}, Loss = {:0.4f},    Elapsed: {:0.1f}'.format(i, train_loader_cnt,
                                                                                           len(train_loader), iter_loss,
                                                                                           delta_t))
            train_loader_cnt += 1
        scheduler.step()

        print('Epoch', i, 'Loss = ', epoch_loss / len(train_loader), 'Epoch time = ', time.time() - epoch_start)

        model_pathname = os.path.join(config['config_dir'], '{}_{}.pth'.format(config['model_class'],i))  
        torch.save(model.module.state_dict(), model_pathname)
        print('Model saved as', os.path.basename(model_pathname))

    model_pathname = os.path.join(config['config_dir'], '{}.pth'.format(config['model_class']))
    torch.save(model.module.state_dict(), model_pathname)
    print('Model saved as', os.path.basename(model_pathname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', required=True)
    parser.add_argument('--resume', required=False, type=int,
                        help='Resume the model trained after x epoch, only remaining epochs will be run.')
    main(parser.parse_args())
