# Standard Library
import os
import random
import time

from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
from utils import config_kg
import PyQt5
matplotlib.use('QtAgg')

# PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import json
from pytorch_metric_learning import losses
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


# utils
from utils import visualize



class TrainBase():

    def __init__(self, model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, epochs:int = 50, early_stop:int=25, lr: float = 0.001, lr_scheduler: str = None, warmup:bool=True,
                 metrics: list = None, name: str="model", out_folder :str ="trained_models/", visualise_validation:bool=True, ):

        self.loss_optimizer = None
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10, min_lr=1e-6)
        else:
            scheduler = None
        return scheduler

    def get_loss(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss, outputs

    def t_loop(self, epoch, s):
        # Initialize the running loss
        train_loss = 0.0
        self.model.train()
        # Initialize the progress bar for training
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        t_images = []
        t_outputs = []
        t_labels = []

        # loop training through batches
        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()
            # get loss
            with autocast(dtype=torch.float16):
                loss, outputs = self.get_loss(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.loss_optimizer:
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

            if self.visualise_validation:
                if len(t_images) < images.shape[0]:
                    j = random.randint(0, images.shape[0] - 1)
                    t_images.append(images[j].detach().cpu())
                    t_outputs.append(outputs[j].detach().cpu())
                    t_labels.append(labels[j].detach().cpu())

            if self.visualise_validation and (i % 10000) == 0 and i != 0: # 10000
                t_images = torch.stack(t_images)
                t_outputs = torch.stack(t_outputs)
                t_labels = torch.stack(t_labels)
                self.val_visualize(t_images.detach().cpu().numpy(), t_labels.detach().cpu().numpy(),
                                   t_outputs.detach().cpu().numpy(), name=f'/val_images/train_{i}')

                t_images = []
                t_outputs = []
                t_labels = []

                torch.cuda.empty_cache()
                torch.save(self.model.state_dict(), os.path.join(self.out_folder, f"{self.name}_ckpt.pt"))

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
            val_images = []
            val_outputs = []
            val_labels = []
            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss, outputs = self.get_loss(images, labels)
                val_loss += loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

                if self.visualise_validation:
                    val_images.append(images[0].detach().cpu())
                    val_outputs.append(outputs[0].detach().cpu())
                    val_labels.append(labels[0].detach().cpu())

            if self.visualise_validation:
                val_images = torch.stack(val_images)
                val_outputs = torch.stack(val_outputs)
                val_labels = torch.stack(val_labels)
                self.val_visualize(val_images.detach().cpu().numpy(), val_labels.detach().cpu().numpy(),
                                   val_outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

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
        s = None

        # Training loop
        for epoch in range(self.epochs):
            if epoch == 0 and self.warmup == True:
                s = self.scheduler_warmup
                print('Starting linear warmup phase')
            elif epoch == 0 and self.warmup == False:
                s = self.scheduler
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

            test_loss = 0
            for k, (images, labels) in enumerate(test_pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # get loss
                loss, outputs = self.get_loss(images, labels)
                test_loss += loss.item()

            self.test_loss = test_loss / (k + 1)

            print(f"Test Loss: {self.test_loss:.4f}")
            # outputs = self.model(images)
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
                                       'last_epoch': self.last_epoch,
                                       'test_loss': self.test_loss},

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
        return loss, outputs

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")


class TrainViT(TrainBase):
    def get_loss(self, images, labels):
        outputs = self.model(images)
        labels = self.model.patchify(labels)
        loss = self.criterion(outputs, labels)
        return loss, outputs

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
        return loss, outputs

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=11)
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.detach().cpu().numpy().argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")


class TrainKG(TrainBase):
    def __init__(self, *args, num_classes=31, embedding_size=2048, **kwargs): #2048 512
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        super(TrainKG, self).__init__(*args, **kwargs)
        self.loss_optimizer = self.set_optimizer()

    def set_criterion(self):
        return losses.ArcFaceLoss(num_classes=self.num_classes, embedding_size=self.embedding_size, margin=28.6, scale=64).to(self.device)

    def get_encoded_labels(self, labels):
        climate_zone_labels, _ = torch.mode(labels.flatten(start_dim=1), -1)
        return climate_zone_labels.long()

    def get_loss(self, images, labels):
        outputs = self.model(images)
        labels = self.get_encoded_labels(labels)
        loss = self.criterion(outputs, labels)
        return loss, outputs

    def val_visualize(self, images, labels, outputs, name):
        labels = self.get_encoded_labels(torch.from_numpy(labels))
        visualize.visualize_arcface(x=outputs, y=labels.detach().cpu().numpy(), save_path=f"{self.out_folder}/{name}.png")


class TrainContrastive(TrainBase):
    def __init__(self, *args, num_classes=31, embedding_size=2048, **kwargs): #2048 512
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        super(TrainContrastive, self).__init__(*args, **kwargs)
        self.loss_optimizer = self.set_optimizer()

    def kg_similarity(self, kg):

        climate_codes = torch.zeros((kg.shape[0], 6))
        precipitation_codes = torch.zeros((kg.shape[0], 6))
        temperature_codes = torch.zeros((kg.shape[0], 8))
        codes = torch.zeros((kg.shape[0], 20))

        climate_sim = torch.zeros((kg.shape[0], kg.shape[0]))
        precipitation_sim = torch.zeros((kg.shape[0], kg.shape[0]))
        temperature_sim = torch.zeros((kg.shape[0], kg.shape[0]))

        for i, kg_label in enumerate(kg):
            # climate_codes[i] = torch.FloatTensor(config_kg.kg_map[int(torch.argmax(kg_label))]['climate_code'])
            # precipitation_codes[i] = torch.FloatTensor(config_kg.kg_map[int(torch.argmax(kg_label))]['precipitation_code'])
            # temperature_codes[i] = torch.FloatTensor(config_kg.kg_map[int(torch.argmax(kg_label))]['temperature_code'])
            codes[i] = torch.FloatTensor(config_kg.kg_map[int(torch.argmax(kg_label))]['climate_code'] +
                                         config_kg.kg_map[int(torch.argmax(kg_label))]['precipitation_code'] +
                                         config_kg.kg_map[int(torch.argmax(kg_label))]['temperature_code'])

        # for i in range(kg.shape[0]):
        #     for j in range(kg.shape[0]):
        #         climate_sim[i, j] = binary_jaccard_index(preds=climate_codes[i], target=climate_codes[j])
        #         precipitation_sim[i, j] = binary_jaccard_index(preds=precipitation_codes[i], target=precipitation_codes[j])
        #         temperature_sim[i, j] = binary_jaccard_index(preds=temperature_codes[i], target=temperature_codes[j])


        similarity = pairwise_cosine_similarity(codes, zero_diagonal=False)

        # similarity = (climate_sim + precipitation_sim + temperature_sim)/3
        # similarity = torch.nan_to_num(similarity, nan=1) # self similarity calculation can return nan if vector is all zeros
        return similarity

    def coords_similarity(self, coords):
        lat_enc, long_sin, long_cos = coords[:, 0], (2 * coords[:, 1] - 1), (2 * coords[:, 2] - 1)
        lat_r = lat_enc * np.pi

        cos_dif = torch.outer(long_cos, long_cos) + torch.outer(long_sin, long_sin)

        return torch.outer(torch.sin(lat_r), torch.sin(lat_r)) * cos_dif + torch.outer(torch.cos(lat_r),
                                                                                       torch.cos(lat_r))

    def date_similarity(self, date):
        sin, cos = (2 * date[:, 0] - 1), (2 * date[:, 1] - 1)
        return torch.outer(cos, cos) + torch.outer(sin, sin)

    def get_similarity_labels(self, labels):

        kg_label = labels[:, :31]
        co_ordinate_labels = labels[:, 31:34]
        date_lables = labels[:, 34:]

        kg_similarity = self.kg_similarity(kg_label)
        coords_similarity = self.coords_similarity(co_ordinate_labels)
        date_similarity = self.date_similarity(date_lables)

        return ((0.4*kg_similarity + 0.4*coords_similarity + 0.2*date_similarity) + 1) / 2 # similarity range: 0 (dissimialr) - 1 (similar)

    def contrastive_criterion(self, embeddings, y, margin = 0.05):
        similarity = pairwise_cosine_similarity(embeddings, zero_diagonal=False)

        loss = torch.sqrt(torch.maximum(torch.zeros(y.shape), torch.square(y - similarity) - (margin ** 2)))
        loss = torch.triu(loss, diagonal=1)

        return loss.sum() / embeddings.shape[0]

    def get_loss(self, images, labels):
        outputs = self.model(images)
        outputs = F.normalize(outputs, p=2, dim=1)
        y = self.get_similarity_labels(labels)
        loss = self.contrastive_criterion(outputs, y)
        return loss, outputs

    def val_visualize(self, images, labels, outputs, name):
        outputs = torch.from_numpy(outputs)
        visualize.visualise_contrastive(y=labels, images=images,
                                        est_sim=self.get_similarity_labels(torch.from_numpy(labels).to(self.device)).detach().cpu().numpy(),
                                        pred_sim=pairwise_cosine_similarity(outputs, zero_diagonal=False).detach().cpu().numpy(),
                                        save_path=f"{self.out_folder}/{name}.png")

