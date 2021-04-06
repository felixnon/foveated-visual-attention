import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import time
import datetime
import shutil
import pickle
import numpy as np

from tqdm import tqdm
from utils import AverageMeter
from model import RecurrentAttention
from tensorboard_logger import configure, log_value


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10 #1000 #365
        self.num_channels = 3

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_file = config.ckpt_file
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = 'ram_{}_{}x{}_{}_{}'.format(
            config.num_glimpses, config.patch_size,
            config.patch_size, config.glimpse_scale, datetime.date.today().strftime("%y-%m-%d")
        )

        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.alternating_learning = False
        self.train_loc_flag = False

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size, self.num_patches, self.glimpse_scale,
            self.num_channels, self.loc_hidden, self.glimpse_hidden,
            self.std, self.hidden_size, self.num_classes,
        )
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # # initialize optimizer and scheduler
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=self.momentum,
        # )
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer, 'min', patience=self.lr_patience
        # )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=3e-4,
        )

    def reset(self):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        h_t = torch.zeros(self.batch_size, self.hidden_size)
        h_t = Variable(h_t).type(dtype)

        l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)
        l_t = Variable(l_t).type(dtype)

        return h_t, l_t

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )
        
        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            # self.scheduler.step(valid_loss)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))


            # check for improvement
            if not is_best:
                self.counter += 1
            if self.alternating_learning:
                if self.counter >= 5:
                    self.train_loc_flag = not self.train_loc_flag
                    print("[!] No improvement in a while. Switch loss. Now training:", ["ActionNet", "LocationNet"][self.train_loc_flag])
                    self.counter = 0
                    # if not self.train_loc_flag:
                    #     self.lr /= 5
                    #     print("[!] No improvement in a while. Decrease learning rate:", self.lr)

            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
                
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 }, is_best
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
            # for i, (x, y), f in enumerate(self.train_loader): # uncomment when using dataset with fixation proposals
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                log_pi = []
                log_probs = []
                baselines = []

                for t in range(self.num_glimpses):
                    

                    locs.append(l_t)

                    h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t)
                    
                    #l_t = f[:,t].float() # uncomment when using dataset with fixation proposals

                    # store                    
                    baselines.append(b_t)
                    log_pi.append(p)
                    log_probs.append(log_probas[0:9])


                # convert list to tensors and reshape
                baselines = torch.stack(baselines)
                baselines = baselines.transpose(1, 0)
                #log_pi = torch.stack(log_pi).transpose(1, 0)  # only when using RL
                log_probs = torch.stack(log_probs).transpose(1, 0)

                # calculate reward
                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)
                #R = R - self.get_loc_reward(locs)

                # compute losses for differentiable modules
                loss_action = F.nll_loss(log_probas, y)
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                #loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)  # only when using RL
                #loss_reinforce = torch.mean(loss_reinforce, dim=0)  # only when using RL

                loss = loss_action
                #loss = loss_action + loss_baseline + loss_reinforce * 0.01  # only when using RL
                

                # sum up into a hybrid loss
                # if self.alternating_learning:
                #     if self.train_loc_flag:
                #         loss = loss_reinforce
                #     else:
                #         loss = loss_action
                # else:
                #     loss = loss_action + loss_baseline + loss_reinforce
                #loss = loss_action

                # compute accuracy
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc.data.item(), x.size()[0])

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc-tic), loss.data.item(), acc.data.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    if self.use_gpu:
                        imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                        locs = [l.cpu().data.numpy()[:9] for l in locs]
                        log_probs = [p.cpu().data.numpy() for p in log_probs]
                        ys = [g.cpu().data.numpy() for g in y[:9]]
                    else:
                        imgs = [g.data.numpy().squeeze() for g in imgs]
                        locs = [l.data.numpy()[:9] for l in locs]
                        log_probs = [p.data.numpy() for p in log_probs]
                        ys = [g.data.numpy() for g in y[:9]]
                    pickle.dump(
                        imgs, open(
                            self.plot_dir + "g_{}.p".format(epoch+1),
                            "wb"
                        )
                    )
                    pickle.dump(
                        locs, open(
                            self.plot_dir + "l_{}.p".format(epoch+1),
                            "wb"
                        )
                    )
                    pickle.dump(
                        log_probs, open(
                            self.plot_dir + "p_{}.p".format(epoch+1),
                            "wb"
                        )
                    )
                    pickle.dump(
                        ys, open(
                            self.plot_dir + "y_{}.p".format(epoch+1),
                            "wb"
                        )
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch*len(self.train_loader) + i
                    log_value('train_loss', losses.avg, iteration)
                    log_value('train_acc', accs.avg, iteration)

            return losses.avg, accs.avg

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
        # for i, (x, y), f in enumerate(self.valid_loader): # uncomment when using dataset with fixation proposals
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y  = Variable(x), Variable(y)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses):
                
                # forward pass through model
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t)
                #l_t = f[:,t].float()  # uncomment when using dataset with fixation proposals

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            #log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            log_probas = log_probas.view(
                self.M, -1, log_probas.shape[-1]
            )
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(
                self.M, -1, baselines.shape[-1]
            )
            baselines = torch.mean(baselines, dim=0)

            #log_pi = log_pi.contiguous().view(
            #   self.M, -1, log_pi.shape[-1]
            #)
            #log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            #loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)  # only when using RL
            #loss_reinforce = torch.mean(loss_reinforce, dim=0)  # only when using RL

            # sum up into a hybrid loss
            #loss = loss_action + loss_baseline + loss_reinforce * 0.01  # only when using RL
            loss = loss_action

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch*len(self.valid_loader) + i
                log_value('valid_loss', losses.avg, iteration)
                log_value('valid_acc', accs.avg, iteration)

        return losses.avg, accs.avg

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        torch.manual_seed(15)
        import pandas as pd

        correct = 0
        #corrects = np.zeros((self.num_glimpses))

        offset_x = []
        offset_y = []
        l_ts1 = []
        l_ts2 = []
        l_ts3 = []
        probas1 = []
        probas2 = []
        probas3 = []
        ys = []
        corrs = []

        # load the best checkpoint
        self.load_checkpoint(best=self.best, ckpt_file=self.ckpt_file)

        # for i, (x, y, offset) in enumerate(self.test_loader):
        for i, (x, y) in enumerate(self.test_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            #x, y = Variable(x, volatile=True), Variable(y)
            x, y = Variable(x, requires_grad=False), Variable(y)
            # duplicate 10 times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            l_ts_temp = []
            probas_temp = []
            for t in range(self.num_glimpses):

                l_ts_temp.append((l_t + 1) / 2.0)

                # forward pass through model
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t)
                
                # get acc after each glimpse
                log_probas = log_probas.view( 
                    self.M, -1, log_probas.shape[-1]
                )
                log_probas = torch.mean(log_probas, dim=0)

                pred = log_probas.data.max(1, keepdim=True)[1]
                #corrects[t] += pred.eq(y.data.view_as(pred)).cpu().sum()
                
                
                probas_temp.append(log_probas)

            

            # offset_x.append(offset[0].cpu().detach())
            # offset_y.append(offset[1].cpu().detach())
            l_ts1.append(l_ts_temp[0].cpu().detach())
            l_ts2.append(l_ts_temp[1].cpu().detach())
            l_ts3.append(l_ts_temp[2].cpu().detach())
            probas1.append(probas_temp[0].cpu().detach())
            probas2.append(probas_temp[1].cpu().detach())
            probas3.append(probas_temp[2].cpu().detach())
            ys.append(y.cpu().detach())
            corrs.append(pred.eq(y.data.view_as(pred)).cpu().detach())

            # # last iteration
            # h_t, l_t, b_t, log_probas, p = self.model(
            #     x, l_t, h_t, last=True
            # )
            # # get acc after each glimpse
            # log_probas = log_probas.view(
            #     self.M, -1, log_probas.shape[-1]
            # )
            # log_probas = torch.mean(log_probas, dim=0)

            # pred = log_probas.data.max(1, keepdim=True)[1]
            # #corrects[t+1] += pred.eq(y.data.view_as(pred)).cpu().sum()
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

            #print(i+1,":",corrects/(i+1))

            # log_probas = log_probas.view(
            #     self.M, -1, log_probas.shape[-1]
            # )
            # log_probas = torch.mean(log_probas, dim=0)

            # pred = log_probas.data.max(1, keepdim=True)[1]
            # correct += pred.eq(y.data.view_as(pred)).cpu().sum()

            # if i >= 3:
            #     break

        # offset_x = torch.unsqueeze(torch.cat(offset_x), dim=1).float()
        # offset_y = torch.unsqueeze(torch.cat(offset_y), dim=1).float()
        l_ts1 = torch.cat(l_ts1)
        l_ts2 = torch.cat(l_ts2)
        l_ts3 = torch.cat(l_ts3)
        probas1 = torch.cat(probas1)
        probas2 = torch.cat(probas2)
        probas3 = torch.cat(probas3)
        ys = torch.unsqueeze(torch.cat(ys), dim=1).float()
        corrs = torch.cat(corrs).float()

        offset_x = torch.zeros(ys.shape)
        offset_y = torch.zeros(ys.shape)

        data = torch.cat([offset_x, offset_y,
                         l_ts1, l_ts2, l_ts3,
                         probas1, probas2, probas3,
                         ys, corrs], dim=1)

        data = pd.DataFrame(data.numpy())
        data.to_csv('temp.csv')

        # perc = (100. * corrects[t+1]) / (self.num_test)
        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        print(
            '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
                correct, self.num_test, perc, error)
        )
        #print((100. * correct) / (self.num_test))

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False, ckpt_file=None):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """

        if ckpt_file == None:
            if best:
                filename = self.model_name + '_model_best.pth.tar'
            else:
                filename = self.model_name + '_ckpt.pth.tar'
        else:
            filename = ckpt_file

        
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )

    def get_loc_reward(self, locs):
        """
        Calculates a negative reward if subsequent glimpses are very close to a previuos glimpse

        Args:
        ----
        locs: List of locations

        Returns:
        ----
        reward: A negativ reward signal

        """
        pdist = torch.nn.PairwiseDistance(p=2, )

        min_dists = []

        # calc distance from glimpse to all glimpses before
        for i, l_t in enumerate(locs):
            dists = []

            # use max possible distance for first glimpse as no glimpses before
            if i == 0:
                if self.use_gpu:
                    dists.append(torch.ones(l_t.shape[0]).unsqueeze(1).cuda() * float("inf"))
                else:
                    dists.append(torch.ones(l_t.shape[0]).unsqueeze(1) * float("inf"))
                
            # get distance to all previous glimpses
            for l in locs[:i]:
                dists.append(pdist(l, l_t).unsqueeze(1))
            dists = torch.cat(dists, dim=1)
            dists = torch.min(dists, dim=1).values.unsqueeze(1)
            min_dists.append(dists)

        min_dists = torch.cat(min_dists, dim=1)
        reward = torch.exp(-10*min_dists)
        
        return reward

            

            

