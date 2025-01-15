import warnings
warnings.filterwarnings("ignore")
from iwla import configs as config
import os
import torch
import torch.nn as nn
import torch.optim as optim
from iwla.dataset import IWLA_dataset
from iwla.models import IWLA_Model
from torch.utils.data import DataLoader
import infobatch
import ast
import json
import wandb
import tqdm



def train(args, model, train_dataset, train_loader, optimizer, criterion, epoch):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        audio, visual, question, yolo, s_f_a, s_f_v, d_f_a, d_f_v,  target = sample['audio'].to('cuda'), sample['visual'].to(
            'cuda'), sample['question'].to('cuda'), sample["yolo"].to('cuda'), sample["sep_f_a"].to('cuda'), sample["sep_f_v"].to(
            'cuda'), sample["diff_f_a"].to('cuda'), sample["diff_f_v"].to('cuda'), sample['label'].to('cuda')

        optimizer.zero_grad()
        out_qa = model(audio, visual, question, yolo, d_f_a, d_f_v, s_f_a, s_f_v, stage='train')
        loss = criterion(out_qa.float(), target.float())
        loss = train_dataset.update(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def model_test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open("./datasets/json_update/avqa-test.json", 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
            audio, visual, question, yolo, s_f_a, s_f_v, d_f_a, d_f_v, target = sample['audio'].to('cuda'), sample[
                'visual'].to(
                'cuda'), sample['question'].to('cuda'), sample["yolo"].to('cuda'), sample["sep_f_a"].to('cuda'), sample[
                "sep_f_v"].to(
                'cuda'), sample["diff_f_a"].to('cuda'), sample["diff_f_v"].to('cuda'), sample['label'].to('cuda')

            preds_qa = model(audio, visual, question, yolo, d_f_a, d_f_v, s_f_a, s_f_v)
            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]
            type = ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())

    with open("test_results.log", "w", encoding="utf-8") as f:
        f.write('Audio Counting Accuracy: %.2f %%\n' % (
                100 * sum(A_count) / len(A_count)))
        f.write('Audio Cmp Accuracy: %.2f %%\n' % (
                100 * sum(A_cmp) / len(A_cmp)))
        f.write('Audio Accuracy: %.2f %%\n' % (
                100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
        f.write('Visual Counting Accuracy: %.2f %%\n' % (
                100 * sum(V_count) / len(V_count)))
        f.write('Visual Loc Accuracy: %.2f %%\n' % (
                100 * sum(V_loc) / len(V_loc)))
        f.write('Visual Accuracy: %.2f %%\n' % (
                100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
        f.write('AV Ext Accuracy: %.2f %%\n' % (
                100 * sum(AV_ext) / len(AV_ext)))
        f.write('AV counting Accuracy: %.2f %%\n' % (
                100 * sum(AV_count) / len(AV_count)))
        f.write('AV Loc Accuracy: %.2f %%\n' % (
                100 * sum(AV_loc) / len(AV_loc)))
        f.write('AV Cmp Accuracy: %.2f %%\n' % (
                100 * sum(AV_cmp) / len(AV_cmp)))
        f.write('AV Temporal Accuracy: %.2f %%\n' % (
                100 * sum(AV_temp) / len(AV_temp)))

        f.write('AV Accuracy: %.2f %%\n' % (
                100 * (sum(AV_count) + sum(AV_loc) + sum(AV_ext) + sum(AV_temp)
                       + sum(AV_cmp)) / (len(AV_count) + len(AV_loc) + len(AV_ext) + len(AV_temp) + len(AV_cmp))))

        f.write('Overall Accuracy: %.2f %%\n' % (
                100 * correct / total))


    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count) / len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc) + sum(AV_ext) + sum(AV_temp)
                   + sum(AV_cmp)) / (len(AV_count) + len(AV_loc) + len(AV_ext) + len(AV_temp) + len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total


def main():
    print("\n--------------- IWLA Model --------------- \n")
    # Training settings
    if config.wandb:
        wandb.init(project="IWLA", name=config.model_name)

    torch.manual_seed(config.seed)

    model = IWLA_Model(config)
    model = model.to('cuda')

    if config.mode == 'train':
        train_dataset = infobatch.InfoBatch(IWLA_dataset(config, dataset_json_path=config.label_train), config.epochs)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.num_workers,
            persistent_workers=config.num_workers,
            pin_memory=True,
            sampler=train_dataset.sampler
        )
        val_dataset = IWLA_dataset(config, dataset_json_path=config.label_test)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True)


        param_group = []
        train_params = 0
        total_params = 0
        additional_params = 0
        if os.path.exists(config.model_load_dir):
            model.load_state_dict(torch.load(config.model_load_dir))
        for name, param in model.named_parameters():
            param.requires_grad = True
            ### ---> compute params
            tmp = 1
            for num in param.shape:
                tmp *= num

            if 'ViT' in name or 'swin' in name or 'Resnet' in name:
                if 'norm' in name:
                    param.requires_grad = bool(config.is_vit_ln)
                    total_params += tmp
                    train_params += tmp
                else:
                    param.requires_grad = False
                    total_params += tmp

            # ### <----

            elif 'cal' in name:
                param.requires_grad = True
                train_params += tmp
                additional_params += tmp
                total_params += tmp
                print('########### train layer:', name)
            else:
                param.requires_grad = True
                train_params += tmp
                total_params += tmp

            if 'caf' in name:
                param_group.append({"params": param, "lr": config.lr_caf})
            else:
                param_group.append({"params": param, "lr": config.lr})
        print('####### Trainable paramsin M: %0.1f M  #######' % (train_params / 1000000))
        print(
            '####### CAL params in M: %0.1f M  ######' % (additional_params / 1000000))
        print('####### Total params in M: %0.1f M  #######' % (total_params / 1000000))

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss(reduction="none")
        best_F = 0
        count = 0
        for epoch in range(1, config.epochs + 1):
            train(config, model, train_dataset, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            # F = eval(model, val_loader, epoch)
            F = model_test(model, val_loader)
            count += 1
            if F >= best_F:
                count = 0
                best_F = F
                if config.wandb:
                    wandb.log({"val-best": best_F})
                if not os.path.exists(config.model_save_dir):
                    os.makedirs(config.model_save_dir)
                torch.save(model.state_dict(), os.path.join(config.model_save_dir, f"best_model_{best_F}.pt"))

    else:
        test_dataset = IWLA_dataset(config, dataset_json_path=config.label_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        print(test_dataset.__len__())
        # model.load_state_dict(torch.load(config.model_load_dir))
        model_test(model, test_loader)


if __name__ == '__main__':
    main()