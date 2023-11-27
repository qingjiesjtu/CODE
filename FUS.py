import tqdm, torch, os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from opts import get_opts
from utils.utils import get_name
from utils.settings import DATASETTINGS
from models import build_model
from datasets import build_transform, build_data
from attacks import build_trigger
import wandb
from torchvision import datasets
import json, fcntl
from PIL import Image

def FUS(opts):
    name = get_name(opts, 'FUS')
    print(name)
    #enter your wandb account to visualize the training process
    wandb.login(anonymous="allow",key="XXX") 
    config=wandb.config
    config={
        'score': opts.score,
        'dataset':opts.data_name,
        'model':opts.model_name,
        'attack name':opts.attack_name,
        'trigger':opts.trigger,
        'target':opts.target,
        'ratio':opts.ratio,
        'alpha':opts.alpha,
        'Searching iteration':opts.n_iter,
        'Training epochs':opts.n_epoch,
        'Early score epochs': opts.early_epoch,
        'device': opts.device,
    }
    experiment = wandb.init(project='Data selection', resume='never',name=name,config=config)
    
    # get data settings
    DSET = DATASETTINGS[opts.data_name]
    
    # set transforms
    train_transform = build_transform(True, DSET['img_size'], DSET['crop'], DSET['flip'])
    val_transform = build_transform(False, DSET['img_size'], DSET['crop'], DSET['flip'])
    
    # set the trigger to poison data
    trigger = build_trigger(opts.attack_name, DSET['img_size'], DSET['num_data'], mode=0, target=opts.target, trigger=opts.trigger)
    
    # load data and set preprocess method
    train_data = build_data(opts.data_name, True, trigger, train_transform)
    val_data = build_data(opts.data_name, False, trigger, val_transform)
    
    # set poison ratio and randomly poison data through shuffle
    poison_num = int(len(train_data.targets) * opts.ratio)
    shuffle = np.arange(len(train_data.targets))[np.array(train_data.targets) != opts.target]  # select poisoned samples from data of non-target classes only
    np.random.shuffle(shuffle)
    samples_idx = shuffle[:poison_num]  # create random poison samples idx

    # FUS begins
    for n in range(opts.n_iter):
        print('searching with {:2d} iteration'.format(n))
        model = build_model(opts.model_name, DSET['num_classes']).to(opts.device)
        # model = torch.compile(model) # for pytorch 2.0
        train_data = build_data(opts.data_name, True, trigger, train_transform)
        # append selected poisoned samples to the clean train dataset
        train_data.data = np.concatenate((train_data.data, train_data.data[samples_idx]), axis=0) \
            if opts.data_name in ('cifar10','cifar100') else train_data.data + [train_data.data[i] for i in samples_idx]
        train_data.targets = train_data.targets + [train_data.targets[i] for i in samples_idx]
        train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=2) 
        val_loader = DataLoader(dataset=val_data, batch_size=256, shuffle=False, num_workers=4)

        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55], 0.1)
        criterion = nn.CrossEntropyLoss().to(opts.device)

        correctness = []
        train_loss, train_acc, val_acc, back_acc = -1, -1, -1, -1        
        
        # model training begins        
        for epoch in range(opts.n_epoch):
            if opts.score in ('BAS') and epoch == opts.early_epoch and n<opts.n_iter-1: break
            trigger.set_mode(0), model.train() # poisoned training
            correct, total, ps, ds = 0, 0, [], []
            train_loss = 0
            desc = 'train - epoch: {:3d}, acc: {:.3f}'
            run_tqdm = tqdm.tqdm(train_loader, desc=desc.format(epoch, 0, 0, 0), disable=opts.disable)
            for x, y, b, s, d in run_tqdm:
                x, y, b, s, d = x.to(opts.device), y.to(opts.device), b.to(opts.device), s.to(opts.device), d.to(opts.device)
                optimizer.zero_grad()
                p = model(x)
                loss_cls = criterion(p, y)
                loss_cls.backward()
                train_loss += loss_cls.item()
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]

                ps.append((p == y)[d >= DSET['num_data']].long().detach().cpu().numpy())  # record correctness of poisoned samples
                ds.append(d[d >= DSET['num_data']].detach().cpu().numpy())

                optimizer.step()
                run_tqdm.set_description(desc.format(epoch, correct / (total + 1e-12)))
            scheduler.step()
            train_acc = correct / (total + 1e-8)
            train_loss /= len(train_loader)

            ps, ds = np.concatenate(ps, axis=0), np.concatenate(ds, axis=0)
            ps = ps[np.argsort(ds)]  # from small to large
            correctness.append(ps[:, np.newaxis])  # record correctness per epoch

                
            trigger.set_mode(1), model.eval() # benign evaluation
            correct, total = 0, 0
            desc = 'val   - epoch: {:3d}, acc: {:.3f}'
            run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0, 0), disable=opts.disable)
            for x, y, _, _, _ in run_tqdm:
                x, y = x.to(opts.device), y.to(opts.device)
                with torch.no_grad():
                    p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
                run_tqdm.set_description(desc.format(epoch, correct / total))
            val_acc = correct / (total + 1e-8)

            trigger.set_mode(2), model.eval() # asr evaluation
            correct, total = 0, 0
            desc = 'back  - epoch: {:3d}, acc: {:.3f}'
            run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0, 0), disable=opts.disable)
            for x, y, b, _, _ in run_tqdm:
                x, y, b = x.to(opts.device), y.to(opts.device), b.to(opts.device)
                idx = b == 1
                x, y, b = x[idx, :, :, :], y[idx], b[idx]
                if x.shape[0] == 0: continue
                with torch.no_grad():
                    p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
                run_tqdm.set_description(desc.format(epoch, correct / total))
            back_acc = correct / (total + 1e-8)

            if opts.disable:
                print('epoch: {:3d}, train_acc: {:.3f}, val_acc: {:.3f}, back_acc: {:.3f}'.format(epoch, train_acc, val_acc, back_acc))

            experiment.log({
                f'Search Iteration {n} Tr.Loss': train_loss,
                f'Search Iteration {n} Tr.ACC': train_acc,
                f'Search Iteration {n} Val.ACC': val_acc,
                f'Search Iteration {n} ASR': back_acc,
                f'Search Iteration {n} Epoch': epoch                       
            })

        experiment.log({
                'Search Iteration': n,
                'Tr.Loss': train_loss,
                'Tr.ACC': train_acc,
                'Val.ACC': val_acc,
                'ASR': back_acc                      
            })
        
        # save asr in json file
        if n == opts.n_iter-1 or (opts.score == 'forgettingscore' and n == 0):
            if n == opts.n_iter-1: # save the selected poisoned samples
                np.save(os.path.join(opts.sample_path, '{}.npy'.format(name)), samples_idx)  
            else:
                np.save(os.path.join(opts.sample_path, '{}.npy'.format(name+'random')), samples_idx) 
            resFileName = 'res.json'
            with open(resFileName, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(0)
                try:
                    res = json.load(f)
                except json.JSONDecodeError:
                    res = {}  

                data = res
                if opts.attack_name not in data:
                    data[opts.attack_name]={}
                data = data[opts.attack_name]
                if opts.model_name not in data:
                    data[opts.model_name]={}
                data = data[opts.model_name]
                if opts.data_name not in data:
                    data[opts.data_name]={}
                data = data[opts.data_name]
                method='_'.join([opts.score,'FUS']) if n == opts.n_iter-1 else 'random'
                if method not in data:
                    data[method]={}
                data = data[method]
                ratio = str(opts.ratio)
                if ratio not in data:
                    data[ratio]={'ASR':[],'ValAcc':[]}
                data = data[ratio]

                data['ASR'].append(back_acc)
                data['ValAcc'].append(val_acc)

                f.seek(0)
                json.dump(res, f, indent=4)
                f.truncate()
                
                fcntl.flock(f, fcntl.LOCK_UN)
                
            if(n == opts.n_iter-1):
                return 
        
        # compute score and sort data
        if opts.score == 'forgettingscore':
            correctness = np.concatenate(correctness, axis=1)
            diff = correctness[:, 1:] - correctness[:, :-1]
            forget_events = np.sum(diff == -1, axis=1)

            forget_events_idx = np.argsort(forget_events)
            samples_idx = samples_idx[forget_events_idx][::-1]  # sort the selected poisoned samples in order of FEs from large to small
        elif opts.score == 'BAS':
            trigger.set_mode(0), model.train() # poisoned training
            BAS, ds = [], []
            train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=2) # batchsize=1 for compute score
            run_tqdm = tqdm.tqdm(train_loader, desc="Compute BAS")
            for x, y, b, s, d in run_tqdm:
                x, y, b, s, d = x.to(opts.device), y.to(opts.device), b.to(opts.device), s.to(opts.device), d.to(opts.device)
                if d<DSET['num_data']: continue
                optimizer.zero_grad()
                p = model(x)
                loss_cls = criterion(p, y)
                loss_cls.backward()
                loss_grads = []
                for param in model.parameters():
                    loss_grads.append(param.grad.clone().view(-1)) 
                loss_grads = torch.cat(loss_grads)
                
                score = torch.linalg.norm(loss_grads)

                BAS.append(score.detach().cpu().numpy())  # record score of poisoned samples
                ds.append(d.detach().cpu().numpy())

                optimizer.zero_grad()

            BAS, ds = np.array(BAS), np.concatenate(ds, axis=0)
            BAS = BAS[np.argsort(ds)]  # from small to large
            BAS_idx = np.argsort(BAS)
            samples_idx = samples_idx[BAS_idx][::-1]  

        samples_idx = samples_idx[:int(len(samples_idx) * opts.alpha)]  # retain a certain number of poisoned samples

        np.random.shuffle(shuffle)
        samples_idx = np.concatenate((samples_idx, shuffle[:(poison_num - len(samples_idx))]), axis=0)  # random add new poisoned samples from the pool



if __name__ == '__main__':
    opts = get_opts() 
    FUS(opts)
