import warnings
warnings.filterwarnings('ignore')
import wandb
import sys
from rt_module import configs
import os
from rt_module.ts_model import Timestamp_Feature_Encoder
from rt_module.dataset import Timestamp_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(model, train_loader, optimizer, criterion_v, criterion_a, epoch):
    model.train()
    losses_v = 0.0
    losses_a = 0.0
    steps = 0
    for batch_idx, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        audio = sample['audio'].to('cuda')
        visual = sample['visual'].to('cuda')
        label = sample['label'].to('cuda')

        optimizer.zero_grad()
        v_outs, a_outs = model(audio, visual)
        loss_v = criterion_v(v_outs, label)
        loss_a = criterion_a(a_outs, label)
        loss = loss_a * 0.5 + loss_v * 0.5
        loss.backward()
        optimizer.step()
        losses_v += loss_v.item()
        losses_a += loss_a.item()
        steps += 1

    print('Train Epoch: {} \tLoss V: {:.6f} \tLoss A: {:.6f}'.format(epoch, losses_v/steps, losses_a/steps))
    return losses_v/steps, losses_a/steps

def main():
    # Training settings
    if configs.wandb:
        wandb.init(project="iwla", name=configs.model_name)

    torch.manual_seed(configs.seed)

    model = Timestamp_Feature_Encoder(config=configs).cuda()

    train_dataset = Timestamp_dataset(config=configs)
    train_loader = DataLoader(train_dataset,
                              batch_size=configs.batch_size,
                              shuffle=True,
                              num_workers=configs.num_workers,
                              prefetch_factor=configs.num_workers,
                              persistent_workers=configs.num_workers,
                              pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion_v = nn.MSELoss()
    criterion_a = nn.MSELoss()
    for epoch in range(1, configs.epochs + 1):
        loss_v, loss_a = train(model, train_loader, optimizer, criterion_v, criterion_a, epoch=epoch)
        scheduler.step(epoch)
        if configs.wandb:
                wandb.log({"loss v": loss_v, "loss a": loss_a})
        torch.save(model.state_dict(), configs.model_save_dir)
        # if count == config.early_stop:
        #     exit()


if __name__ == '__main__':
    main()