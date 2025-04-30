import random
import subprocess
import glob
import torch
import os
import sys
import numpy as np
import pickle

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from helper import CNN
from helper import helper

def main():
    raw_data_path = r"D:\3D joints"
    categories = {
        "emblems": [0, 1, 2],
        "illustrators": [3, 4],
        "regulators": [5, 6, 7],
        "adaptors": [8],
        "affect display": [9, 10, 11]
    }

    num_subsets = 30
    num_actions_low = 5
    num_actions_high = 12
    action_numbers = []

    result_dict = {}

    for idx in range(num_subsets):
	    action_numbers.append(random.randint(num_actions_low, num_actions_high))

    test_list = ['CC01', 'CM10']

    for idx in range(num_subsets):
        actions = helper.select_samples(categories, action_numbers[idx])
        helper.extract_keypoints(actions, raw_data_path, test_list, idx)
        print("------------------------------")
        print(f"Experiment {idx}:")
        print(f"Activities: {actions}")
        print(f"Finished extracting keypoint data for experiment_{idx}...")
        config_file_path = os.path.join(f".\experiment_{idx}", f"DUET_experiment_{idx}_config.py")
        result = subprocess.run([sys.executable, r".\mmaction2\tools\train.py", config_file_path, '--seed', '42'])
        print(f"Finished training ST-GCN for experiment_{idx}...")

        checkpoint_file_path = glob.glob(os.path.join(".\work_dirs", f"DUET_experiment_{idx}_config", "best_acc_top1_epoch_*.pth"))
        device = "cuda"
        pickle_path = os.path.join(f".\experiment_{idx}", f"experiment_{idx}.pkl")
        helper.extract_features(config_file_path, checkpoint_file_path[0], pickle_path, device, idx, test_list)
        print(f"Finished extracting hidden features for experiment_{idx}...")

        features_train = np.load(os.path.join(f".\experiment_{idx}", "train.npy"))
        activity_train = np.load(os.path.join(f".\experiment_{idx}", "train_label.npy"))
        features_test = np.load(os.path.join(f".\experiment_{idx}", "gtest.npy"))
        activity_test = np.load(os.path.join(f".\experiment_{idx}", "g_label.npy"))
        features_train = np.reshape(features_train, (features_train.shape[0], 1, features_train.shape[1]))
        features_test = np.reshape(features_test, (features_test.shape[0], 1, features_test.shape[1]))

        taxonomy = {0: 4, 1: 4, 2: 4, 3: 3, 4: 3, 5: 0, 6: 0, 7: 0, 8: 2, 9: 1, 10: 1, 11: 1}
        labels_train = np.copy(activity_train)
        labels_test = np.copy(activity_test)
        for key, value in iter(taxonomy.items()):
            labels_train[activity_train==key] = value
            labels_test[activity_test==key] = value

        train_set = TensorDataset(torch.tensor(features_train), torch.tensor(labels_train))
        test_set = TensorDataset(torch.tensor(features_test), torch.tensor(labels_test))
        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)

        model = CNN.NeuralNetwork(features_train.shape[2], num_classes=5)
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            CNN.train(train_dataloader, model, loss_fn, optimizer)
        test_accuracy = CNN.test(test_dataloader, model, loss_fn)

        result_dict[idx] = {"experiment_num": idx, "num_activities": len(actions), "actions": actions, "accuracy": test_accuracy}
        print(f"Finished training CNN for experiment_{idx}...")
    
    with open("experiment_results.pkl", "wb") as file: 
        pickle.dump(result_dict, file)


if __name__ == "__main__":
    main()

        


    
