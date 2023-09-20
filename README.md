# phileo-testbed
Repo for testing foundation models

## Installation
conda env create -f environment.yml

## Usage

```python
python training_script.py 
```

### Parameters
```bash
  --experiment_name EXPERIMENT_NAME                         Experiment folder name
  --model {CoreUNET,LinearViT,AutoEncoderViT}               Select appropriate model
  --lr LR                                                   Set learning rate
  --batch_size BATCH_SIZE                                   Set batch size
  --epochs EPOCHS                                           Set training epochs
  --early_stop EARLY_STOP                                   Set training loop patience for early stopping
  --lr_scheduler {None,reduce_on_plateau,cosine_annealing}  Select learning rate scheduler
  --warmup WARMUP                                           Enables linear 5 epoch warmup scheduler
  --device DEVICE                                           Select training device
  --num_workers NUM_WORKERS                                 Set number of workers
  --vis_val VIS_VAL                                         Enable saving of intermediate visualization plots
  --downstream_task {roads,building,lc}                     Select downstream task
  --regions REGIONS                                         Select regions to be included
  --n_shot N_SHOT                                           Loads n-samples of data from specified geographic regions
  --split_ratio SPLIT_RATIO                                 Loads a percentage of the data from specified geographic regions.
  --augmentations AUGMENTATIONS                             Enables augmentations
```
