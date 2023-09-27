# phileo-testbed
Repo for testing foundation models

## Installation
conda env create -f environment.yml

## Usage

```python
python training_script.py [--experiment_name EXPERIMENT_NAME] --model_name
                          {baseline_cnn,core_unet_base,core_unet_large,core_unet_huge,mixer_base,mixer_large,mixer_huge,linear_vit_base,linear_vit_larger,linear_vit_huge,autoencoder_vit_base,autoencoder_vit_large,autoencoder_vit_huge}
                          [--lr LR] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--early_stop EARLY_STOP] [--lr_scheduler {None,reduce_on_plateau,cosine_annealing}] [--warmup] [--device DEVICE] [--num_workers NUM_WORKERS]
                          [--vis_val] --downstream_task {roads,building,lc} [--input_channels INPUT_CHANNELS] --input_size INPUT_SIZE --output_channels OUTPUT_CHANNELS
                          [--regions {None,denmark-1,denmark-2,east-africa,egypt-1,eq-guinea,europe,ghana-1,isreal-1,isreal-2,japan,nigeria,north-america,senegal,south-america,tanzania-1,tanzania-2,tanzania-3,tanzania-4,tanzania-5,uganda-1}]
                          [--n_shot N_SHOT] [--split_ratio SPLIT_RATIO] [--augmentations]

```

## Experiment Script Usage
Trains a series of models (for a specific downstream task) on variouse training dataset sizes and plots the test loss vs number of training samples. 
```python
python p_split_experiment.py [--experiment_name EXPERIMENT_NAME] --model_name
                          {baseline_cnn,core_unet_base,core_unet_large,core_unet_huge,mixer_base,mixer_large,mixer_huge,linear_vit_base,linear_vit_larger,linear_vit_huge,autoencoder_vit_base,autoencoder_vit_large,autoencoder_vit_huge}
                          [--lr LR] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--early_stop EARLY_STOP] [--lr_scheduler {None,reduce_on_plateau,cosine_annealing}] [--warmup] [--device DEVICE] [--num_workers NUM_WORKERS]
                          [--vis_val] --downstream_task {roads,building,lc} [--input_channels INPUT_CHANNELS] --input_size INPUT_SIZE --output_channels OUTPUT_CHANNELS
                          [--regions {None,denmark-1,denmark-2,east-africa,egypt-1,eq-guinea,europe,ghana-1,isreal-1,isreal-2,japan,nigeria,north-america,senegal,south-america,tanzania-1,tanzania-2,tanzania-3,tanzania-4,tanzania-5,uganda-1}]
                          [--n_shot N_SHOT] [--split_ratio SPLIT_RATIO] [--augmentations]

python n_shot_experiment.py [--experiment_name EXPERIMENT_NAME] --model_name
                          {baseline_cnn,core_unet_base,core_unet_large,core_unet_huge,mixer_base,mixer_large,mixer_huge,linear_vit_base,linear_vit_larger,linear_vit_huge,autoencoder_vit_base,autoencoder_vit_large,autoencoder_vit_huge}
                          [--lr LR] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--early_stop EARLY_STOP] [--lr_scheduler {None,reduce_on_plateau,cosine_annealing}] [--warmup] [--device DEVICE] [--num_workers NUM_WORKERS]
                          [--vis_val] --downstream_task {roads,building,lc} [--input_channels INPUT_CHANNELS] --input_size INPUT_SIZE --output_channels OUTPUT_CHANNELS
                          [--regions {None,denmark-1,denmark-2,east-africa,egypt-1,eq-guinea,europe,ghana-1,isreal-1,isreal-2,japan,nigeria,north-america,senegal,south-america,tanzania-1,tanzania-2,tanzania-3,tanzania-4,tanzania-5,uganda-1}]
                          [--n_shot N_SHOT] [--split_ratio SPLIT_RATIO] [--augmentations]
```

### Parameters
```bash
  --experiment_name EXPERIMENT_NAME                         Experiment folder name
  --model                                                   Select appropriate model 
  --lr LR                                                   Set learning rate
  --batch_size BATCH_SIZE                                   Set batch size
  --epochs EPOCHS                                           Set training epochs
  --early_stop EARLY_STOP                                   Set training loop patience for early stopping
  --lr_scheduler                                            Select learning rate scheduler
  --warmup WARMUP                                           Enables epoch linear warmup 
  --device DEVICE                                           Select training device
  --num_workers NUM_WORKERS                                 Set number of workers
  --vis_val VIS_VAL                                         Enable saving of intermediate visualization plots
  --downstream_task {roads,building,lc}                     Select downstream task
  --input_channels                                          Define number of input channels
  --input_size                                              Define hw of input array
  --output_channels                                         Define number of input channels
  --regions REGIONS                                         Select regions to be included if None all regions will be included
  --n_shot N_SHOT                                           Loads n-samples of the training data from specified geographic regions
  --split_ratio SPLIT_RATIO                                 Loads a percentage of the training data from specified geographic regions
  --augmentations AUGMENTATIONS                             Enables augmentations
```

