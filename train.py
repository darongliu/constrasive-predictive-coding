import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils import *

def train(model, train_data_loader, saver, total_epoch, lr, log_path, start_epoch=0):
    f_log = open(log_path, 'w')

    optimizer = optim.Adam(model.parameters(), lr=lr)

    min_loss = np.inf

    for epoch in range(start_epoch, start_epoch+total_epoch):
        model.train()
        total_loss = 0.
        total_count = 0

        for i, data in enumerate(train_data_loader):
            feat, length, neg_shift, _ = data
            feat, length, neg_shift = put_to_cuda([feat, length, neg_shift])

            optimizer.zero_grad()
            loss, _ = model(feat, length, neg_shift)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.detach().cpu().numpy()
            total_count += feat.size()[0]

            average_loss = total_loss/total_count

            log = f'epoch{epoch}, {i}/{len(train_data_loader)}, train average loss: {average_loss}'
            print_and_logging(f_log, log)

        if average_loss < min_loss:
            min_loss = average_loss
            log = 'save model'
            print_and_logging(f_log, log)
            #save model
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_average_loss': average_loss,
                'min_loss': min_loss
            }
            name = f'epoch_{epoch}_loss_{average_loss}'
            saver.save(state, name)
        else:
            log = f'higher loss!!!!!!'
            print_and_logging(f_log, log)
    log = 'training end, min loss: ' + str(average_loss)
    print_and_logging(f_log, log)
