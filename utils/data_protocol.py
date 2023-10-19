# Standard Library
import os
from glob import glob
import pandas as pd

# External Libraries
import buteo as beo
import numpy as np
import random

random.seed(97)


REGIONS_DOWNSTREAM_DATA = ['denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                           'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                           'tanzania-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1']

REGIONS = REGIONS_DOWNSTREAM_DATA
LABELS = ['label_roads','label_kg','label_building','label_lc', 'label_coords']


def sanity_check_labels_exist(x_files, y_files):
    """
    checks that s2 and label numpy files are consistent
    :param x_files:
    :param y_files:
    :return:
    """
    existing_x = []
    existing_y = []
    counter_missing = 0

    assert len(x_files) == len(y_files)
    for x_path, y_path in zip(x_files, y_files):

        exists = os.path.exists(y_path)
        if exists:
            existing_x.append(x_path)
            existing_y.append(y_path)
        else:
            counter_missing += 1

    if counter_missing > 0:
        print(f'WARNING: {counter_missing} label(s) not found')
        missing = [y_f for y_f in y_files if y_f not in existing_y]
        print(f'Showing up to 5 missing files: {missing[:5]}')

    return existing_x, existing_y


def get_testset(folder: str,
                regions: list = None,
                y: str = 'building'):

    """
    Loads a pre-defined test set data from specified geographic regions.
    :param folder: dataset source folder
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: test MultiArrays
    """
    x_test_files = []

    if regions is None:
        regions = REGIONS
    else:
        for r in regions:
            assert r in REGIONS, f"region {r} not found"

    for region in regions:
        # get test samples of region
        x_test_files = x_test_files + sorted(glob(os.path.join(folder, f"{region}*test_s2.npy")))
    y_test_files = [f_name.replace('s2', f'label_{y}') for f_name in x_test_files]
    x_test_files, y_test_files = sanity_check_labels_exist(x_test_files, y_test_files)

    x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])
    y_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_test_files])

    assert len(x_test) == len(y_test), "Lengths of x and y do not match."

    return x_test, y_test


def protocol_minifoundation(folder: str, y:str):
    """
    Loads all the data from the data folder.
    """

    x_train = sorted(glob(os.path.join(folder, f"*/*train_s2.npy")))
    y_train = [f_name.replace('s2', f'label_{y}') for f_name in x_train]

    x_val = []
    y_val = []
    for i in range(int(len(x_train)*0.05)):
        j = random.randint(0, len(x_train)-1)
        x_val.append(x_train[j])
        y_val.append(y_train[j])
        del x_train[j]; del y_train[j]

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train], shuffle=True)
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train], shuffle=True)
    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val], shuffle=True)
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val], shuffle=True)

    return x_train, y_train, x_val, y_val


def protocol_split(folder: str,
                   split_percentage: float = 0.1,
                   regions: list = None,
                   y: str = 'building'):
    """
    Loads a percentage of the data from specified geographic regions.
    :param folder: dataset source folder
    :param split_percentage: percentage of data to sample from each region
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: train, val MultiArrays
    """

    if regions is None:
        regions = REGIONS
    else:
        for r in regions:
            assert r in REGIONS, f"region {r} not found"

    assert 0 < split_percentage <= 1, "split percentage out of range (0 - 1)"

    df = pd.read_csv(glob(os.path.join(folder, f"*.csv"))[0])
    df = df.sort_values(by=['samples'])

    x_train_files = []

    for region in regions:
        mask = [region in f for f in df.iloc[:, 0]]
        df_temp = df[mask].copy().reset_index(drop=True)
        # skip iteration if Region does not belong to current dataset
        if df_temp.shape[0] == 0:
            continue

        df_temp['cumsum'] = df_temp['samples'].cumsum()

        # find row with closest value to the required number of samples
        idx_closest = df_temp.iloc[
            (df_temp['cumsum'] - int(df_temp['samples'].sum() * split_percentage)).abs().argsort()[:1]].index.values[0]
        x_train_files = x_train_files + list(df_temp.iloc[:idx_closest, 0])

    x_train_files = [os.path.join(folder, f_name) for f_name in x_train_files]
    y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
    x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
    y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]


    # checks that s2 and label numpy files are consistent
    x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
    x_val_files, y_val_files = sanity_check_labels_exist(x_val_files, y_val_files)


    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])

    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

    assert len(x_train) == len(y_train)  and len(x_val) == len(
        y_val), "Lengths of x and y do not match."

    return x_train, y_train, x_val, y_val


def check_region_validity(folder, regions, y):
    l = []
    for i, region in enumerate(regions):
        # generate multi array for region
        x_train_files = sorted(glob(os.path.join(folder, f"{region}*train_s2.npy")))
        y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]

        # checks that s2 and label numpy files are consistent
        x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
        if x_train_files:
            l.append(region)

    return l


def protocol_fewshot(folder: str,
                     dst: str,
                     n: int = 10,
                     val_ratio: float = 0.2,
                     regions: list = None,
                     y: str = 'building',
                     resample: bool = False,
                     ):

    """
    Loads n-samples data from specified geographic regions.
    :param folder: dataset source folder
    :param dst: save folder
    :param n: number of samples
    :param val_ratio: ratio of validation set
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: train, val MultiArrays
    """
    if os. path. exists(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy'):
        train_X_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy', mmap_mode='r')
        train_y_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_train_label_{y}.npy', mmap_mode='r')
        val_X_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_val_s2.npy', mmap_mode='r')
        val_y_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_val_label_{y}.npy', mmap_mode='r')
    else:

        if regions is None:
            regions = REGIONS
        else:
            for r in regions:
                assert r in REGIONS, f"region {r} not found"

        regions = check_region_validity(folder, regions, y)

        f_x = glob(os.path.join(folder, f"{regions[0]}*test_s2.npy"))[0]
        ref_x = np.load(f_x, mmap_mode='r')
        f_y = glob(os.path.join(folder, f"{regions[0]}*test_label_{y}.npy"))[0]
        ref_y = np.load(f_y, mmap_mode='r')

        d_size = n*len(regions)
        d_size_val = int(np.ceil(n*val_ratio)*len(regions))

        train_X_temp = np.zeros_like(a=ref_x, shape=(d_size, ref_x.shape[1], ref_x.shape[2], ref_x.shape[3]))
        val_X_temp = np.zeros_like(a=ref_x, shape=(d_size_val, ref_x.shape[1], ref_x.shape[2], ref_x.shape[3]))
        train_y_temp = np.zeros_like(a=ref_y, shape=(d_size, ref_y.shape[1], ref_y.shape[2], ref_y.shape[3]))
        val_y_temp = np.zeros_like(a=ref_y, shape=(d_size_val, ref_y.shape[1], ref_y.shape[2], ref_y.shape[3]))
        del ref_x ; del ref_y

        for i, region in enumerate(regions):
            # generate multi array for region
            x_train_files = sorted(glob(os.path.join(folder, f"{region}*train_s2.npy")))
            y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
            x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
            y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]

            # checks that s2 and label numpy files are consistent
            x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
            x_val_files, y_val_files = sanity_check_labels_exist(x_val_files, y_val_files)


            x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
            y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])
            x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
            y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

            if n < len(x_train):
                train_indexes = random.sample(range(0, len(x_train)), n)

                for j, idx in enumerate(train_indexes):
                    train_X_temp[(n*i)+j] = x_train[idx]
                    train_y_temp[(n * i) + j] = y_train[idx]

            else:
                # resample if n > than regions number of samples
                for j in range(0, len(x_train)):
                    train_X_temp[(n * i)+j] = x_train[j]
                    train_y_temp[(n * i)+j] = y_train[j]

                if resample:
                    train_indexes = random.choices(range(0, len(x_train)), k=(n - len(x_train)))
                    for j, idx in enumerate(train_indexes):
                        train_X_temp[(n * i)+len(x_train)+j] = x_train[idx]
                        train_y_temp[(n * i)+len(x_train) + j] = y_train[idx]

            if int(np.ceil(n * val_ratio)) < len(x_val):

                val_indexes = random.sample(range(0, len(x_val)), int(np.ceil(n * val_ratio)))

                for j, idx in enumerate(val_indexes):
                    val_X_temp[(int(np.ceil(n * val_ratio)) * i) + j] = x_val[idx]
                    val_y_temp[(int(np.ceil(n * val_ratio)) * i) + j] = y_val[idx]

            else:
                # resample if n > than regions number of samples
                for j in range(0, len(x_val)):
                    val_X_temp[(int(np.ceil(n * val_ratio)))+j] = x_val[j]
                    val_y_temp[(int(np.ceil(n * val_ratio)))+j] = y_val[j]

                if resample:
                    val_indexes = random.choices(range(0, len(x_val)), k=((int(np.ceil(n * val_ratio))) - len(x_val)))
                    for j, idx in enumerate(val_indexes):
                        val_X_temp[(int(np.ceil(n * val_ratio)))+len(x_val)+j] = x_val[idx]
                        val_y_temp[(int(np.ceil(n * val_ratio)))+len(x_val) + j] = y_val[idx]

            del x_train; del y_train; del x_val; del y_val

        os.makedirs(f'{dst}/{n}_shot_{y}', exist_ok=True)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy', train_X_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_train_label_{y}.npy', train_y_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_val_s2.npy', val_X_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_val_label_{y}.npy', val_y_temp)
    return train_X_temp, train_y_temp, val_X_temp, val_y_temp


if __name__ == '__main__':
    label =['roads', 'building', 'lc']
    n_shots = [1, 2, 5, 10, 50, 100, 150, 200, 500, 750, 1000]
    for l in label:
        for n in n_shots:
            x_train, y_train, x_val, y_val = protocol_fewshot('/phileo_data/downstream/downstream_dataset_patches_np/',
                                                              dst='/phileo_data/downstream/downstream_datasets_nshot/',
                                                              n=n,
                                                              y=l)