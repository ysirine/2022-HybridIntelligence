""" compileLabels.py

this file is to organize images according to their labels in order to use them later for training

"""
import os, sys
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def load_original_dataset(labels_dataset_addr, relative_label_addr, dataset_addr):
    if dataset_addr is None:
        # Parse original dataset address from labels dataset address
        relative_label_addr_len = len(relative_label_addr)+1
        dataset_addr = labels_dataset_addr[relative_label_addr_len:]
        # print(f'Loading data from original dataset address: {dataset_addr}')

    # Load rendered.npz data
    data_dict = dict(np.load(os.path.join(dataset_addr, 'rendered.npz')))

    # Load metadata
    with open(os.path.join(dataset_addr, 'metadata.json')) as f:
        metadata_dict = json.load(f)

    return data_dict, metadata_dict    


def load_actions(labels_dataset_addr, relative_label_addr, dataset_addr=None):
    # Load original dataset
    data_dict, metadata_dict = load_original_dataset(labels_dataset_addr, relative_label_addr, dataset_addr)

    actions = np.vstack((
        data_dict['action$attack'], data_dict['action$back'], data_dict['action$equip'],
        data_dict['action$forward'], data_dict['action$jump'], data_dict['action$left'],
        data_dict['action$right'], data_dict['action$sneak'], data_dict['action$sprint'],
        data_dict['action$use'], data_dict['action$camera'][:,0], data_dict['action$camera'][:,1]
    ))
    actions = actions.transpose()

    # There are more video frames than actions
    # ASSUMPTION: the initial video frames are when the minecraft is still loading,
    # so all actions are zero
    diff_frame_count = metadata_dict['true_video_frame_count']-metadata_dict['duration_steps']
    action_padding_template = ['0', '0', 'none', '0', '0', '0', '0', '0', '0', '0', '0.0', '0.0']
    action_padding = np.full((diff_frame_count,12), action_padding_template)
    actions = np.vstack((action_padding, actions))

    return actions


def print_summary(all_labels_np):
    '''Prints summary of labeled data.'''
    num_labels = all_labels_np.shape[0]
    print(f'Images labeled: {num_labels} images')
    for i in range(all_labels_np.shape[1]):
        labels_per_class = all_labels_np[all_labels_np[:,i] == 1]
        num_labels_per_class = labels_per_class.shape[0]
        print(f'  Labels for class {i}: {num_labels_per_class} ({100*num_labels_per_class/num_labels:.3f} %)')


def main():
    MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
    relative_label_addr = 'labels'

    # Find all labelled tasks
    label_tasks = glob.glob(os.path.join('labels', 'data', '*'))

    all_images_list = []
    all_labels_list = []
    all_actions_list = []

    all_x_trains = []
    all_y_trains = []
    all_x_vals = []
    all_y_vals = []
    all_x_tests = []
    all_y_tests = []

    # For each task, find all labeled datasets
    for label_task in label_tasks:
        print(f'Compiling {label_task} files...')
        dataset_addrs = glob.glob(os.path.join(label_task, '*'))

        # Delete previous labels
        os.system(f"rm -rf {os.path.join(label_task, 'images.npy')}")
        os.system(f"rm -rf {os.path.join(label_task, 'labels.npy')}")
        os.system(f"rm -rf {os.path.join(label_task, 'actions.npy')}")

        # For each dataset, compile all images and labels
        for dataset_addr in dataset_addrs:
            # print(f'Compiling images and labels from {dataset_addr}')
            dataset_files = sorted(glob.glob(os.path.join(dataset_addr, '*')))

            # Load all demonstrated actions (not compiled files)
            if dataset_addr[-3:] != 'npy':
                actions = load_actions(
                    labels_dataset_addr=dataset_addr,
                    relative_label_addr=relative_label_addr)

            # Loop for all files in the labeled dataset folder
            for dataset_file in dataset_files:
                file_extension = dataset_file[-4:]

                # Use only files/images with json labels
                if file_extension == 'json':
                    file_number = dataset_file[-12:-5]

                    # Load labels in numpy format
                    with open(dataset_file) as f:
                        label_json = json.load(f)
                    label_np = np.array(list(label_json.values())[1:])

                    # check if all labels are zeros before appendding (skip images with no labels)
                    if label_np.sum() != 0:
                        all_labels_list.append(label_np)

                        # Only use the actions for the frames we have label
                        all_actions_list.append(actions[int(file_number)])

                        # Load image in numpy format
                        img_addr = dataset_addr + '/' + file_number + '.png'
                        img_np = plt.imread(img_addr)
                        all_images_list.append(img_np)

        # Save all images with their labels to disk
        all_images_np = np.array(all_images_list)
        with open(f'{label_task}/images.npy', 'wb') as f:
            np.save(f, all_images_np)

        all_labels_np = np.array(all_labels_list)
        with open(f'{label_task}/labels.npy', 'wb') as f:
            np.save(f, all_labels_np)

        all_actions_np = np.array(all_actions_list)
        with open(f'{label_task}/actions.npy', 'wb') as f:
            np.save(f, all_actions_np)


        print_summary(all_labels_np)
        print(f'Done. Saves images.npy and labels.npy in {label_task} folder.')


if __name__ == "__main__":
    main()
