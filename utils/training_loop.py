# Standard Library
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
import PyQt5
matplotlib.use('QtAgg')

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np


# utils
import utils


class TrainBase():

    def __init__(self, model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, epochs:int = 50, early_stop:int=25, lr: float = 0.001, lr_scheduler: str = None, warmup:bool=True,
                 metrics: list = None, name: str="model", out_folder :str ="trained_models/", visualise_validation:bool=True, ):

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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10, min_lr=1e-6)
        else:
            scheduler = None
        return scheduler

    def get_loss(self, images, labels):
        # Cast to bfloat16
        with autocast(dtype=torch.float16):
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss

    def t_loop(self, train_pbar, s):
        # Initialize the running loss
        train_loss = 0.0

        # loop training through batches
        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()
            # get loss
            loss = self.get_loss(images, labels)
            train_loss += loss.item()

            # display progress on console
            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                f"lr": self.optimizer.param_groups[0]['lr']})

            # # Update the scheduler
            if self.lr_scheduler == 'cosine_annealing':
                self.s.step()

        return i, train_loss

    def val_visualize(self, images, labels, outputs):
        utils.visualize(x=images, y=labels, y_pred=outputs, images=5,
                        channel_first=True, vmin=0, vmax=1, save_path=f"{self.out_folder}/val_images.png")

    def v_loop(self):
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for j, (images, labels) in enumerate(self.val_loader):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss = self.get_loss(images, labels)
                val_loss += loss.item()

            if self.visualise_validation:
                outputs = self.model(images)
                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())

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

        # Training loop
        for epoch in range(self.epochs):
            if epoch == 0 and self.warmup == True:
                s = self.scheduler_warmup
                print('Starting linear warmup phase')
            elif epoch == 5:
                s = self.scheduler
                self.warmup = False
                print('Warmup finished')

            # Initialize the progress bar for training
            train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                              desc=f"Epoch {epoch + 1}/{self.epochs}")

            i, train_loss = self.t_loop(train_pbar, s)
            j, val_loss = self.v_loop()

            # display progress on console
            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                "val_loss": f"{val_loss / (j + 1):.4f}",
                f"lr": self.optimizer.param_groups[0]['lr']})

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
                break

    def test(self):
        # Load the best weights
        self.model.load_state_dict(self.best_sd)

        print("Finished Training. Best epoch: ", self.best_epoch + 1)
        print("")
        print("Starting Testing...")
        self.model.eval()
        with torch.no_grad():

            test_loss = 0
            for k, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

            print(f"Test Loss: {test_loss / (k + 1):.4f}")
            self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                               outputs.detach().cpu().numpy())


