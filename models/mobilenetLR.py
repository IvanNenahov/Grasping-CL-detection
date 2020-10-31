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
from sklearn.model_selection import train_test_split

try:
    from pytorchcv.models.mobilenet import DwsConvBlock
except:
    from pytorchcv.models.common import DwsConvBlock
from pytorchcv.model_provider import get_model


def remove_sequential(network, all_layers):
    for layer in network.children():
        if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
            # print(layer)
            remove_sequential(layer, all_layers)
        else:  # if leaf node, add it to list
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

        # set 50 neurons to output layer
        self.past_j = {i: 0 for i in range(50)}
        self.cur_j = {i: 0 for i in range(50)}

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

        # the regularization is based on Synaptic Intelligence as described in the
        # paper. ewcData is a list of two elements (best parametes, importance)
        # while synData is a dictionary with all the trajectory data needed by SI
        self.ewcData, self.synData = create_syn_data(model)

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

        # self.randomMemory = random_memory.RandomMemory(lat_list[-1].si)

    def train_on_data(self, dataset, freeze_below_layer="lat_features.19.bn.beta",
                            init_lr=0.0005, momentum=0.9, l2=0.0005,
                            batch_size=128, use_cuda=True, epochs=1,
                            reg_lambda=0):

        tot_it_step = 0

        print('model trained:', self.trained)
        if not self.trained:
            freeze_up_to(self, freeze_below_layer, only_conv=False)

        if reg_lambda != 0:
            init_batch(self, self.ewcData, self.synData)

        optimizer = torch.optim.SGD(self.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2)
        criterion = torch.nn.CrossEntropyLoss()

        (train_x, train_y) = dataset

        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)

        train_x = preprocess_imgs(train_x)

        if not self.trained:
            cur_class = [int(o) for o in set(train_y)]
            self.cur_j = examples_per_class(train_y)
        else:
            cur_class = [int(o) for o in set(train_y).union(set(self.RM.getLabels()))]
            self.cur_j = examples_per_class(list(train_y) + list(self.RM.getLabels()))

        # print("----------- batch {0} -------------".format(i))
        print("train_x shape: {}, train_y shape: {}"
              .format(train_x.shape, train_y.shape))

        self.train()
        self.lat_features.eval()

        # zero init weights
        reset_weights(self, cur_class)
        if not self.trained:
            (train_x, train_y), it_x_epoch = pad_data([train_x, train_y], batch_size)
        shuffle_in_unison([train_x, train_y], in_place=True)

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor)

        if self.trained:
            cur_sz = train_x.size(0) // ((train_x.size(0) + self.RM.getsize()) // batch_size)
            it_x_epoch = train_x.size(0) // cur_sz
            n2inject = max(0, batch_size - cur_sz)
        else:
            n2inject = 0

        print("total sz:", train_x.size(0) + self.RM.getsize())
        print("n2inject", n2inject)
        print("it x ep: ", it_x_epoch)

        for epoch in range(epochs):
            print("----------- epoch {0} -------------".format(epoch))

            correct_cnt, ave_loss = 0, 0

            for it in range(it_x_epoch):

                # do EWC strategy
                if reg_lambda != 0:
                    pre_update(self, self.synData)

                start = it * (batch_size - n2inject)
                end = (it + 1) * (batch_size - n2inject)

                optimizer.zero_grad()

                x_batch = maybe_cuda(train_x[start:end])

                if not self.trained:
                    lat_mb_x = None
                    y_batch = maybe_cuda(train_y[start:end])
                else:
                    lat_mb_x = self.RM.getActivations()[it * n2inject: (it + 1) * n2inject]
                    lat_mb_y = self.RM.getLabels()[it * n2inject: (it + 1) * n2inject]
                    y_batch = maybe_cuda(
                        torch.cat((train_y[start:end], lat_mb_y), 0),
                        use_cuda=use_cuda)
                    lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)

                logits, lat_acts = self(
                    x_batch, latent_input=lat_mb_x, return_lat_acts=True)

                lat_acts = lat_acts.cpu().detach()

                if not self.trained:
                    # print(f'Random memory shape: {lat_acts.shape}')
                    self.RM = random_memory.RandomMemory(patterns_shape=lat_acts.shape)

                # collect latent volumes only for the first ep
                # we need to store them to eventually add them into the external
                # replay memory
                if epoch == 0:
                    lat_acts = lat_acts.cpu().detach()
                    if it == 0:
                        cur_acts = copy.deepcopy(lat_acts)
                    else:
                        cur_acts = torch.cat((cur_acts, lat_acts), 0)

                _, pred_label = torch.max(logits, 1)
                correct_cnt = (pred_label == y_batch).sum()
                # print(correct_cnt)

                loss = criterion(logits, y_batch)

                if reg_lambda != 0:
                    loss += compute_ewc_loss(self, self.ewcData, lambd=reg_lambda)

                ave_loss += loss.item()

                loss.backward()
                optimizer.step()

                if reg_lambda != 0:
                    post_update(self, self.synData)

                acc = correct_cnt.item() / batch_size
                ave_loss /= ((it + 1) * y_batch.size(0))

                if it % 8 == 0:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'.format(it, ave_loss, acc)
                    )

                tot_it_step += 1

            ave_loss, acc, accs = get_accuracy(
                self, criterion, batch_size, val_x, val_y, preproc=preprocess_imgs)
            print("---------------------------------")
            print("Accuracy: ", acc)
            print("---------------------------------")

            # update number examples encountered over time
            for c, n in self.cur_j.items():
                self.past_j[c] += n

        consolidate_weights(self, cur_class)
        if reg_lambda != 0:
            update_ewc_data(self, self.ewcData, self.synData, 0.001, 1)

        set_consolidate_weights(self)

        # replace patterns in random memory
        self.RM.addPatterns(cur_acts, train_y)

        # update number examples encountered over time
        for c, n in self.cur_j.items():
            self.past_j[c] += n

        self.trained = True


def main(root='../core50_128x128'):
    model = MobileNetWLR(pretrained=True)

    dataset = CORE50(root=root, scenario="nicv2_391")
    device = torch.device("cuda:0")
    model.to(device)

    model.train_on_data(dataset.next())


if __name__ == "__main__":
    main()
