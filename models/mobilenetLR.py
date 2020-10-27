#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020. Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo           #
# Pellegrini, Davide Maltoni. All rights reserved.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-04-2020                                                             #
# Authors: Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo Pellegrini, Davide   #
# Maltoni.                                                                     #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" This file contains the model Class used for the exps"""

import torch
import torch.nn as nn
from models import random_memory
from utils import *
import copy
from data_loader import CORE50
import time
from tqdm.auto import tqdm

try:
    from pytorchcv.models.mobilenet import DwsConvBlock
except:
    from pytorchcv.models.common import DwsConvBlock
from pytorchcv.model_provider import get_model


def remove_sequential(network, all_layers):

    for layer in network.children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            #print(layer)
            remove_sequential(layer, all_layers)
        else: # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)


def remove_DwsConvBlock(cur_layers):

    all_layers = []
    for layer in cur_layers:
        if isinstance(layer, DwsConvBlock):
           #  print("helloooo: ", layer)
            for ch in layer.children():
                all_layers.append(ch)
        else:
            all_layers.append(layer)
    return all_layers


class MobileNetWLR(nn.Module):


    def __init__(self, pretrained=True, latent_layer_num=20, random_memory_size=1500, replace_bn=True,
                 init_update_rate=0.01, inc_update_rate=0.00005, max_r_max=1.25, max_d_max=0.5,
                 inc_step=4.1e-05):
        super().__init__()

        self.trained = False
        self.saved_weights = {}
        self.RM = random_memory.RandomMemory(rmsize=random_memory_size)


        model = get_model("mobilenet_w1", pretrained=pretrained)
        model.features.final_pool = nn.AvgPool2d(4)

        all_layers = []
        remove_sequential(model, all_layers)
        all_layers = remove_DwsConvBlock(all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers[:-1]):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        self.output = nn.Linear(1024, 50, bias=False)

        if replace_bn:
            replace_bn_with_brn(
                self, momentum=init_update_rate, r_d_max_inc_step=inc_step,
                max_r_max=max_r_max, max_d_max=max_d_max
            )

        #self.randomMemory = random_memory.RandomMemory(lat_list[-1].si)

    def forward(self, x, latent_input=None, return_lat_acts=False):

        if not self.trained:
            assert "Net is not trained"

        orig_acts = self.lat_features(x)
        if latent_input is not None:
            lat_acts = torch.cat((orig_acts, latent_input), 0)
        else:
            lat_acts = orig_acts

        x = self.end_features(lat_acts)
        x = x.view(x.size(0), -1)
        logits = self.output(x)

        if return_lat_acts:
            return logits, orig_acts
        else:
            return logits


        #self.randomMemory = random_memory.RandomMemory(lat_list[-1].si)


def do_initial_training(model: MobileNetWLR, dataset, freeze_below_layer="lat_features.19.bn.beta",
                init_lr=0.0005, momentum=0.9, l2=0.0005,
                batch_size=128, use_cuda=True):

    # need to explicitly move model to cuda out of the function

    tot_it_step = 0

    freeze_up_to(model, freeze_below_layer, only_conv=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2)
    criterion = torch.nn.CrossEntropyLoss()

    correct_cnt, ave_loss = 0, 0
    for i, train_batch in tqdm(enumerate(dataset)):

        train_x, train_y = train_batch
        train_x = preprocess_imgs(train_x)

        cur_class = [int(o) for o in set(train_y)]

        print("----------- batch {0} -------------".format(i))
        print("train_x shape: {}, train_y shape: {}"
              .format(train_x.shape, train_y.shape))

        model.train()
        model.lat_features.eval()

        # What is the purpose of function below???
        # reset_weights(model, cur_class)

        (train_x, train_y), it_x_epoch = pad_data([train_x, train_y], batch_size)
        shuffle_in_unison([train_x, train_y], in_place=True)

        #model = maybe_cuda(model, use_cuda=use_cuda)

        acc = None
        ave_loss = 0

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor)

        activations_to_save = torch.Tensor()

        for it in tqdm(range(it_x_epoch // 8)):
            start = it * (batch_size)
            end = (it + 1) * (batch_size)

            optimizer.zero_grad()

            x_batch = maybe_cuda(train_x[start:end], use_cuda=use_cuda)

            lat_mb_x = None
            y_batch = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

            logits, lat_acts = model(
                x_batch, latent_input=lat_mb_x, return_lat_acts=True)

            lat_acts = lat_acts.cpu().detach()

            if not model.trained and i == 0:
                print(f'Random memory shape: {lat_acts.shape}')
                model.RM = random_memory.RandomMemory(patterns_shape=lat_acts.shape)

            if it == 0:
                cur_acts = copy.deepcopy(lat_acts)
            else:
                cur_acts = torch.cat((cur_acts, lat_acts), 0)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_batch).sum()

            loss = criterion(logits, y_batch)

            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            acc = correct_cnt.item() / \
                  ((it + 1) * y_batch.size(0))
            ave_loss /= ((it + 1) * y_batch.size(0))

            if it % 10 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'.format(it, ave_loss, acc)
                )

            tot_it_step += 1

        consolidate_weights(model, cur_class)

        set_consolidate_weights(model)

        #replace patterns in random memory
        model.RM.addPatterns(cur_acts, train_y)

        test_x, test_y = dataset.get_test_set()
        ave_loss, acc, accs = get_accuracy(
            model, criterion, batch_size, test_x, test_y, preproc=preprocess_imgs)
        print("---------------------------------")
        print("Accuracy: ", acc)
        print("---------------------------------")

        model.trained = True

if __name__ == "__main__":

    model = MobileNetWLR(pretrained=True)

    dataset = CORE50(root='G:\projects\core50\core50_128x128', scenario="nicv2_391")
    device = torch.device("cuda:0")
    #model.to(device)
    do_initial_training(model, dataset)



