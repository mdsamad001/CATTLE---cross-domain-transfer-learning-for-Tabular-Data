import os
import pdb
import math
import time
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
# from transformers.optimization import get_scheduler
from tqdm.autonotebook import trange
from loguru import logger
import matplotlib.pyplot as plt
from . import constants
from .evaluator import predict, get_eval_metric_fn, EarlyStopping
from .modeling_cattle import cattleFeatureExtractor
from .trainer_utils import SupervisedTrainCollator, TrainDataset
from .trainer_utils import get_parameter_names
from .trainer_utils import get_scheduler
from .modeling_cattle import cattleTransformerLayer
from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(self,
        model,
        train_set_list,
        test_set_list=None,
        collate_fn=None,
        output_dir='./ckpt',
        num_epoch=10,
        batch_size=64,
        lr=1e-4,
        num_class=2,
        weight_decay=0,
        patience=5,
        eval_batch_size=256,
        warmup_ratio=None,
        warmup_steps=None,
        balance_sample=False,
        load_best_at_last=True,
        ignore_duplicate_cols=False,
        eval_metric='val_loss',
        eval_less_is_better=True,
        num_workers=0,
        exp='',
        save_best=True,
        **kwargs,
        ):
        '''args:
        train_set_list: a list of training sets [(x_1,y_1),(x_2,y_2),...]
        test_set_list: a list of tuples of test set (x, y), same as train_set_list. if set None, do not do evaluation and early stopping
        patience: the max number of early stop patience
        num_workers: how many workers used to process dataloader. recommend to be 0 if training data smaller than 10000.
        eval_less_is_better: if the set eval_metric is the less the better. For val_loss, it should be set True.
        '''
        self.model = model
        if isinstance(train_set_list, tuple): train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple): test_set_list = [test_set_list]

        self.train_set_list = train_set_list
        self.test_set_list = test_set_list
        self.collate_fn = collate_fn
        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )
        self.trainloader_list = [
            self._build_dataloader(trainset, batch_size, collator=self.collate_fn, num_workers=num_workers) for trainset in train_set_list
        ]
        if test_set_list is not None:
            self.testloader_list = [
                self._build_dataloader(testset, eval_batch_size, collator=self.collate_fn, num_workers=num_workers, shuffle=False) for testset in test_set_list
            ]
        else:
            self.testloader_list = None
            
        self.test_set_list = test_set_list
        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(output_dir=output_dir, patience=patience, verbose=False, less_is_better=eval_less_is_better)
        self.args = {
            'lr':lr,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'num_epoch':num_epoch,
            'eval_batch_size':eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'num_training_steps': self.get_num_train_steps(train_set_list, num_epoch, batch_size),
            'eval_metric': get_eval_metric_fn(eval_metric),
            'eval_metric_name': eval_metric,
            }
        self.args['steps_per_epoch'] = int(self.args['num_training_steps'] / (num_epoch*len(self.train_set_list)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.optimizer = None
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last
        
        self.num_class = num_class
        
        self.exp=exp
        
        self.save_best=save_best

    def train(self):
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            num_train_steps = args['num_training_steps']
            logger.info(f'set warmup training in initial {num_train_steps} steps')
            self.create_scheduler(num_train_steps, self.optimizer)

        start_time = time.time()
        
        train_losses = []
        test_losses = []
        epochs = []
        
        val_aucs = []
        
        best_eval_res = float('-inf')  # Initialize with worst possible performance
        best_model_path = os.path.join(self.output_dir, f"{self.exp}_best_model.pth")  # Path to save best model

        for epoch in trange(args['num_epoch'], desc='Epoch'):
            ite = 0
            train_loss = 0

            for dataindex in range(len(self.trainloader_list)):
                for data in self.trainloader_list[dataindex]:
                    self.optimizer.zero_grad()
                    logits, loss = self.model(data[0], data[1])

                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    ite += 1

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            if self.test_set_list is not None:
                val_loss = 0.0
                val_ite = 0
                # Disable grad for validation
                with torch.no_grad():
                    for data in self.test_set_list:
                        _, vloss = self.model(data[0], data[1])
                        val_loss += vloss.item()
                        val_ite += 1

                avg_val_loss = val_loss / val_ite if val_ite > 0 else 0.0
                test_losses.append(avg_val_loss)
                
                eval_res_list = self.evaluate(self.num_class)
                eval_res = np.mean(eval_res_list)
                #test_losses.append(eval_res)
                val_aucs.append(eval_res)

                # Save the best model if the evaluation metric improves
                if eval_res > best_eval_res:
                    best_eval_res = eval_res
                    if self.save_best:
                        torch.save(self.model.state_dict(), best_model_path)
                        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                        # self.model.save(best_model_path)
                        if epoch%10==0:
                            print(f"Best model updated at epoch {epoch} with eval result: {eval_res:.6f}")
                    
            avg_train_loss = train_loss / ite
            train_losses.append(avg_train_loss)
            epochs.append(epoch)

        if self.save_best:
            # After training, reload the best mode
            
            print("Loading the best model after training...")
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Best model loaded with evaluation result: {best_eval_res:.6f}")
        
        
#         for epoch in trange(args['num_epoch'], desc='Epoch'):
#             ite = 0
#             train_loss = 0

#             for dataindex in range(len(self.trainloader_list)):
#                 #print('len',len(self.trainloader_list[dataindex]))
#                 for data in self.trainloader_list[dataindex]:
#                     #if self.args['eval_metric_name'] == 'auc':
#                         #self.model.apply_ca()
#                     self.optimizer.zero_grad()
                    
#                     logits, loss = self.model(data[0], data[1])

#                     loss.backward()
                    
#                     #for name, param in self.model.named_parameters():
#                     #    if "in_proj_weight" in name and param.grad is not None:
#                     #        d_model = param.shape[1]  # Extract embedding dimension
#                     #        grad_q, grad_k, grad_v = param.grad[:d_model], param.grad[d_model:2*d_model], param.grad[2*d_model:]

#                     #        print(f"{name} - Q: {grad_q.norm().item():.6f}, K: {grad_k.norm().item():.6f}, V: {grad_v.norm().item():.6f}")
#                     self.optimizer.step()
                    
#                     train_loss += loss.item()
#                     ite += 1
#                     #print('ite: {} loss: {}'.format(ite, loss.item()))
#                     if self.lr_scheduler is not None:
#                         self.lr_scheduler.step()
                    
#             if self.test_set_list is not None:
#                 eval_res_list = self.evaluate(self.num_class)
#                 eval_res = np.mean(eval_res_list)
                
#                 # Record test loss for the current epoch
#                 test_losses.append(eval_res)
                
#                 #print('epoch: {}, val {}: {:.6f}'.format(epoch, self.args['eval_metric_name'], eval_res))
#                 self.early_stopping(-eval_res, self.model)
#                 if self.early_stopping.early_stop:
#                     print('early stopped')
#                     break
            

            
            #print('epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch, train_loss, self.optimizer.param_groups[0]['lr'], time.time()-start_time))
            # Record train loss for the current epoch
            # avg_train_loss = train_loss / ite
            # train_losses.append(avg_train_loss)
            # epochs.append(epoch)
            
        # if os.path.exists(self.output_dir):
        #     if self.test_set_list is not None:
        #         # load checkpoints
        #         logger.info(f'load best at last from {self.output_dir}')
        #         state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
        #         self.model.load_state_dict(state_dict)
        #     self.save_model(self.output_dir)
            
        
        self.plot_losses(train_losses, test_losses, epochs, plot_save_dir=self.output_dir) 
        logger.info('training complete, cost {:.1f} secs.'.format(time.time()-start_time))
        
    def plot_losses(self, train_losses, test_losses, epochs, plot_save_dir='./plots'):
        #print('train loss : ', train_losses)
        #print('val loss: ', test_losses)
        
        # Ensure the directory exists
        os.makedirs(plot_save_dir, exist_ok=True)

        # Generate a unique name using the current timestamp
        #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args = self.args
        epoch = args['num_epoch']
        
        plot_name = f'{epoch}_{self.exp}_loss_plot.png'

        # Plot the losses
        plt.figure(figsize=(10, 6))
        min_size = min(len(epochs), len(train_losses))

        plt.plot(epochs[:min_size], train_losses[:min_size], label="Training Loss")

        if test_losses:
            min_size = min(len(epochs), len(test_losses))
            plt.plot(epochs[:min_size], test_losses[:min_size], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        full_dir = plot_save_dir+f'/plots/'
        # Check if the directory exists, if not, create it
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        
        # Save the plot to the specified directory
        plot_path = os.path.join(full_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free memory
        logger.info(f'Plot saved at: {plot_path}')
        
    def evaluate(self, num_class):
        # evaluate in each epoch
        self.model.eval()
        eval_res_list = []
        for dataindex in range(len(self.testloader_list)):
            y_test, pred_list, loss_list = [], [], []
            for data in self.testloader_list[dataindex]:
                if data[1] is not None:
                    label = data[1]
                    if isinstance(label, pd.Series):
                        label = label.values
                    y_test.append(label)
                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1])
                if loss is not None:
                    loss_list.append(loss.item())
                if logits is not None:
                    if logits.shape[-1] == 1: # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else: # multi-class classification
                        pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())

            if len(pred_list)>0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()

            if self.args['eval_metric_name'] == 'val_loss':
                eval_res = np.mean(loss_list)
            else:
                y_test = np.concatenate(y_test, 0)
                if num_class==2:
                    eval_res = self.args['eval_metric'](y_test, pred_all)
                else:
                    eval_res = roc_auc_score(y_test, pred_all, multi_class="ovr", average="macro")

            eval_res_list.append(eval_res)

        return eval_res_list

    def train_no_dataloader(self,
        resume_from_checkpoint = None,
        ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            print('set warmup training.')
            self.create_scheduler(args['num_training_steps'], self.optimizer)

        for epoch in range(args['num_epoch']):
            ite = 0
            # go through all train sets
            for train_set in self.train_set_list:
                x_train, y_train = train_set
                train_loss_all = 0
                for i in range(0, len(x_train), args['batch_size']):
                    self.model.train()
                    if self.balance_sample:
                        bs_x_train_pos = x_train.loc[y_train==1].sample(int(args['batch_size']/2))
                        bs_y_train_pos = y_train.loc[bs_x_train_pos.index]
                        bs_x_train_neg = x_train.loc[y_train==0].sample(int(args['batch_size']/2))
                        bs_y_train_neg = y_train.loc[bs_x_train_neg.index]
                        bs_x_train = pd.concat([bs_x_train_pos, bs_x_train_neg], axis=0)
                        bs_y_train = pd.concat([bs_y_train_pos, bs_y_train_neg], axis=0)
                    else:
                        bs_x_train = x_train.iloc[i:i+args['batch_size']]
                        bs_y_train = y_train.loc[bs_x_train.index]

                    self.optimizer.zero_grad()
                    logits, loss = self.model(bs_x_train, bs_y_train)
                    loss.backward()

                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            if self.test_set is not None:
                # evaluate in each epoch
                self.model.eval()
                x_test, y_test = self.test_set
                pred_all = predict(self.model, x_test, self.args['eval_batch_size'])
                eval_res = self.args['eval_metric'](y_test, pred_all)
                print('epoch: {}, test {}: {}'.format(epoch, self.args['eval_metric_name'], eval_res))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print('early stopped')
                    break

            print('epoch: {}, train loss: {}, lr: {:.6f}'.format(epoch, train_loss_all, self.optimizer.param_groups[0]['lr']))

        if os.path.exists(self.output_dir):
            if self.test_set is not None:
                # load checkpoints
                print('load best at last from', self.output_dir)
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

    def save_model(self, output_dir=None):
        if output_dir is None:
            print('No path assigned for saving model, default saved to ./ckpt/model.pt !')
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f'Saving model checkpoint to {output_dir}')
        
        # ✅ Ensure `state_dict` is always defined
        state_dict = self.model.state_dict()  # Get model's full state dictionary

        # ✅ Extract Q, K, V from the last transformer layer
        last_layer_index = len(self.model.encoder.transformer_encoder) - 1  # Get last transformer layer index
        print(last_layer_index)
        last_layer = self.model.encoder.transformer_encoder[last_layer_index]

        if isinstance(last_layer, cattleTransformerLayer):  # Ensure it's a transformer layer
            attn_layer = last_layer.self_attn  # Get MultiheadAttention layer

            # Extract Q, K, V projection weights
            qkv_weight = attn_layer.in_proj_weight  # Shape: (3*d_model, d_model)
            d_model = attn_layer.embed_dim  # Typically 128 in your model

            # Split into separate Q, K, V matrices
            q_weight = qkv_weight[:d_model, :].cpu().detach()
            k_weight = qkv_weight[d_model:2*d_model, :].cpu().detach()
            v_weight = qkv_weight[2*d_model:, :].cpu().detach()

            # ✅ Store Q, K, V inside a separate dictionary
            state_dict = self.model.state_dict()  # Get model's full state dictionary
            state_dict['last_layer.self_attn.q_weight'] = q_weight
            state_dict['last_layer.self_attn.k_weight'] = k_weight
            state_dict['last_layer.self_attn.v_weight'] = v_weight

        # ✅ Save the full model state_dict (including Q, K, V)
        torch.save(state_dict, os.path.join(output_dir, constants.WEIGHTS_NAME))
        print(f"Model and final Q, K, V weights saved inside state_dict to {output_dir}")

        # ✅ Save optimizer, scheduler, and training arguments if available
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, constants.OPTIMIZER_NAME))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, constants.SCHEDULER_NAME))
        if self.args is not None:
            train_args = {k: v for k, v in self.args.items() if isinstance(v, (int, str, float))}
            with open(os.path.join(output_dir, constants.TRAINING_ARGS_NAME), 'w', encoding='utf-8') as f:
                json.dump(train_args, f)


    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args['lr'])

    def create_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
            'cosine',
            optimizer = optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        total_step *= num_epoch
        return total_step

    def get_warmup_steps(self, num_training_steps):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.args['warmup_steps'] if self.args['warmup_steps'] is not None else math.ceil(num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=True):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            )
        return trainloader
