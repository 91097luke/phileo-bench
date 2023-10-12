# Standard Library
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
# import PyQt5
# matplotlib.use('QtAgg')

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import json

# utils
from utils import visualize
from utils import config_lc


class TrainBase():

    def __init__(self, model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, epochs:int = 50, early_stop:int=25, lr: float = 0.001, lr_scheduler: str = None, warmup:bool=True,
                 metrics: list = None, name: str="model", out_folder :str ="trained_models/", visualise_validation:bool=True, ):

        self.test_loss = None
        self.last_epoch = None
        self.best_sd = None
        self.epochs = epochs
        self.early_stop = early_stop
        self.learning_rate = lr
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.warmup = warmup
        self.name = name
        self.out_folder = out_folder
        self.visualise_validation = visualise_validation
        if visualise_validation:
            os.makedirs(f'{self.out_folder}/val_images', exist_ok=True)

        self.scaler, self.optimizer = self.set_optimizer()
        self.criterion = self.set_criterion()
        self.scheduler = self.set_scheduler()

        if self.warmup:
            self.scheduler_warmup = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[1, 2, 3, 4, 5], gamma=(10))

        # initialize torch device
        torch.set_default_device(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            print("No CUDA device available.")

        # init useful variables
        self.best_epoch = 0
        self.best_loss = None
        self.best_model_state = model.state_dict().copy()
        self.epochs_no_improve = 0

        # used for plots
        self.tl = []
        self.vl = []
        self.e = []
        self.lr = []



    def set_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate, eps=1e-06)

        scaler = GradScaler()

        # Save the initial learning rate in optimizer's param_groups
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = self.learning_rate

        return scaler, optimizer

    def set_criterion(self):
        return nn.MSELoss()

    def set_scheduler(self):
        if self.lr_scheduler == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                20,
                2,
                eta_min=0.000001,
                last_epoch=self.epochs - 1,
            )
        elif self.lr_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=6, min_lr=1e-6)
        else:
            scheduler = None
        return scheduler

    def get_loss(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss
    
    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['mse','mae','mave','acc','precision','recall','baseline_mse']
            # intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']

            final_metrics = {'mse':running_metric[0] / (k + 1), 'mae':running_metric[1] / (k + 1), 'mave':running_metric[2] / (k + 1), 'acc':running_metric[3]/ (k + 1), 'precision':running_metric[4]/(running_metric[4]+running_metric[5]), 'recall':running_metric[4]/(running_metric[4]+running_metric[6]), 'baseline_mse':running_metric[7] / (k + 1)}
            final_metrics['f1'] = 2 * final_metrics['precision'] * final_metrics['recall'] / (final_metrics['precision'] + final_metrics['recall'])

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']
            metric_init = np.zeros(len(intermediary_values)) # 
            return  metric_init
        
        
        else:
            
            outputs = self.model(images)
            # regression metrics
            error = outputs - labels
            squared_error = error**2
            test_mse = squared_error.mean().item()
            test_mae = error.abs().mean().item()
            test_mave = torch.mean(torch.abs(outputs.mean(dim=(1,2)) - labels.mean(dim=(1,2)) ) ).item()

            # regression metrics disguised as classification
            threshold = 0.5
            label_classification = (labels > threshold).type(torch.int8)
            output_classification = (outputs > threshold).type(torch.int8)

            diff = output_classification - label_classification
            fp = torch.count_nonzero(diff==1).item()
            fn = torch.count_nonzero(diff==-1).item()
            tp = label_classification.sum().item() - fn

            test_accuracy = (label_classification==output_classification).type(torch.float).mean().item()
            test_zero_model_mse = (labels**2).mean().item()

            return np.array([test_mse,test_mae,test_mave,test_accuracy,tp,fp,fn,test_zero_model_mse])



    def t_loop(self, epoch, s):
        # Initialize the running loss
        train_loss = 0.0
        # Initialize the progress bar for training
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        # loop training through batches
        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()
            # get loss
            with autocast(dtype=torch.float16):
                loss = self.get_loss(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            train_loss += loss.item()

            # display progress on console
            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                f"lr": self.optimizer.param_groups[0]['lr']})

            # # Update the scheduler
            if self.lr_scheduler == 'cosine_annealing':
                s.step()

        return i, train_loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize(x=images, y=labels, y_pred=outputs, images=5,
                            channel_first=True, vmin=0, vmax=1, save_path=f"{self.out_folder}/{name}.png")

    def v_loop(self, epoch):

        # Initialize the progress bar for training
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss = self.get_loss(images, labels)
                val_loss += loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                outputs = self.model(images)
                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss

    def save_ckpt(self, epoch, val_loss):
        model_sd = self.model.state_dict().copy()
        if self.best_loss is None:
            self.best_epoch = epoch
            self.best_loss = val_loss
            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        elif self.best_loss > val_loss:
            self.best_epoch = epoch
            self.best_loss = val_loss
            self.epochs_no_improve = 0

            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        else:
            self.epochs_no_improve += 1

        torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_last.pt"))

    def plot_curves(self, epoch):
        # visualize loss & lr curves
        self.e.append(epoch)

        fig = plt.figure()
        plt.plot(self.e, self.tl, label='Training Loss', )
        plt.plot(self.e, self.vl, label='Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"loss.png"))
        plt.close('all')
        fig = plt.figure()
        plt.plot(self.e, self.lr, label='Learning Rate')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"lr.png"))
        plt.close('all')

    def train(self):
        print("Starting training...")
        print("")

        # init model
        self.model.to(self.device)
        self.model.train()

        # create dst folder for generated files/artifacts
        os.makedirs(self.out_folder, exist_ok=True)
        s = self.scheduler

        # Training loop
        for epoch in range(self.epochs):
            if epoch == 0 and self.warmup == True:
                s = self.scheduler_warmup
                print('Starting linear warmup phase')
            elif epoch == 5 and self.warmup == True:
                s = self.scheduler
                self.warmup = False
                print('Warmup finished')

            i, train_loss = self.t_loop(epoch, s)
            j, val_loss = self.v_loop(epoch)

            self.tl.append(train_loss / (i + 1))
            self.vl.append(val_loss / (j + 1))
            self.lr.append(self.optimizer.param_groups[0]['lr'])

            # Update the scheduler
            if self.warmup:
                s.step()
            elif self.lr_scheduler == 'reduce_on_plateau':
                s.step(self.vl[-1])

            #save check point
            self.save_ckpt(epoch, val_loss)

            # visualize loss & lr curves
            self.plot_curves(epoch)
            self.model.train()

            # Early stopping
            if self.epochs_no_improve == self.early_stop:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                self.last_epoch = epoch + 1
                break

    def test(self):
        # Load the best weights
        self.model.load_state_dict(self.best_sd)

        print("Finished Training. Best epoch: ", self.best_epoch + 1)
        print("")
        print("Starting Testing...")
        self.model.eval()
        test_pbar = tqdm(self.test_loader, total=len(self.test_loader),
                          desc=f"Test Set")
        with torch.no_grad():

            running_metric = self.get_metrics()

            for k, (images, labels) in enumerate(test_pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                running_metric += self.get_metrics(images,labels)


            self.test_metrics = self.get_metrics(running_metric=running_metric, k=k)
        

            print(f"Test Loss: {self.test_metrics}")
            outputs = self.model(images)
            self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                               outputs.detach().cpu().numpy(), name='test')

    def save_info(self, model_summary=None, n_shot=None, p_split=None, warmup=None, lr=None):
        artifacts = {'training_parameters': {'model': self.name,
                                             'lr': lr,
                                             'scheduler': self.lr_scheduler,
                                             'warm_up': warmup,
                                             'optimizer': str(self.optimizer).split(' (')[0],
                                             'device': str(self.device),
                                             'training_epochs': self.epochs,
                                             'early_stop': self.early_stop,
                                             'train_samples': len(self.train_loader) * model_summary.input_size[0][0],
                                             'val_samples': len(self.val_loader) * model_summary.input_size[0][0],
                                             'test_samples': len(self.test_loader) * model_summary.input_size[0][0],
                                             'n_shot': n_shot,
                                             'p_split': p_split
                                             },

                     'training_info': {'best_val_loss': self.best_loss,
                                       'best_epoch': self.best_epoch,
                                       'last_epoch': self.last_epoch},

                     'test_metrics': self.test_metrics,

                     'plot_info': {'epochs': self.e,
                                   'val_losses': self.vl,
                                   'train_losses': self.tl,
                                   'lr': self.lr},

                     'model_summary': {'batch_size': model_summary.input_size[0],
                                       'input_size': model_summary.total_input,
                                       'total_mult_adds': model_summary.total_mult_adds,
                                       'back_forward_pass_size': model_summary.total_output_bytes,
                                       'param_bytes': model_summary.total_param_bytes,
                                       'trainable_params': model_summary.trainable_params,
                                       'non-trainable_params': model_summary.total_params - model_summary.trainable_params,
                                       'total_params': model_summary.total_params}
                     }

        with open(f"{self.out_folder}/artifacts.json", "w") as outfile:
            json.dump(artifacts, outfile)


class TrainLandCover(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def get_loss(self, images, labels):
        outputs = self.model(images)
        outputs = outputs.flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['acc','precision','recall','baseline_mse']
            # intermediary_values = ['confusion_matrix']

            confmat = running_metric

            total_pixels = np.sum(confmat)
            
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class
            

            precision_per_class = tp_per_class/(fp_per_class+tp_per_class)
            recall_per_class = tp_per_class/(fn_per_class+tp_per_class)

            precision_micro = total_tp/(fp_per_class.sum() + total_tp)
            recall_micro = total_tp/(fn_per_class.sum() + total_tp)
            precision_macro = np.mean(precision_per_class)
            recall_macro = np.mean(recall_per_class)

            acc_total = total_tp/total_pixels

            final_metrics = {'acc':acc_total, 'precision_per_class':precision_per_class.tolist(),'recall_per_class':recall_per_class.tolist() ,'precision_micro':precision_micro, 'precision_macro':precision_macro, 'recall_micro':recall_micro, 'recall_macro':recall_macro, 'conf_mat':confmat.tolist()}

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['confusion_matrix']
            num_classes = len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes,num_classes)) # 
            return  metric_init
        
        
        else:
            outputs = self.model(images)
            outputs = outputs.argmax(axis=1).flatten()
            labels = labels.squeeze().flatten()
            
            # stolen from pytorch confusion matrix
            num_classes = len(config_lc.lc_raw_classes.keys())
            unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
            bins = torch.bincount(unique_mapping, minlength=num_classes**2) 
            cfm = bins.reshape(num_classes, num_classes)

            return cfm.cpu().numpy()

class TrainViT(TrainBase):
    def get_loss(self, images, labels):
        outputs = self.model(images)
        labels = self.model.patchify(labels)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=labels.shape[1])
        visualize.visualize(x=images, y=labels, y_pred=outputs.detach().cpu().numpy(), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")


class TrainViTLandCover(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def get_loss(self, images, labels):
        outputs = self.model.unpatchify(self.model(images), c=11).flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=11)
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.detach().cpu().numpy().argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['acc','precision','recall','baseline_mse']
            # intermediary_values = ['confusion_matrix']

            confmat = running_metric

            total_pixels = np.sum(confmat)
            
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class
            

            precision_per_class = tp_per_class/(fp_per_class+tp_per_class)
            recall_per_class = tp_per_class/(fn_per_class+tp_per_class)

            precision_micro = total_tp/(fp_per_class.sum() + total_tp)
            recall_micro = total_tp/(fn_per_class.sum() + total_tp)
            precision_macro = np.mean(precision_per_class)
            recall_macro = np.mean(recall_per_class)

            acc_total = total_tp/total_pixels

            final_metrics = {'acc':acc_total, 'precision_per_class':precision_per_class.tolist(),'recall_per_class':recall_per_class.tolist() ,'precision_micro':precision_micro, 'precision_macro':precision_macro, 'recall_micro':recall_micro, 'recall_macro':recall_macro, 'conf_mat':confmat.tolist()}

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['confusion_matrix']
            num_classes = len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes,num_classes)) # 
            return  metric_init
        
        
        else:
            outputs = self.model.unpatchify(self.model(images), c=11)
            outputs = outputs.argmax(axis=1).flatten()
            labels = labels.squeeze().flatten()
            
            # stolen from pytorch confusion matrix
            num_classes = len(config_lc.lc_raw_classes.keys())
            unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
            bins = torch.bincount(unique_mapping, minlength=num_classes**2) 
            cfm = bins.reshape(num_classes, num_classes)

            return cfm.cpu().numpy()