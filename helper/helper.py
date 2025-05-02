import os
import pickle
import numpy as np
import random

from transformers import CLIPProcessor, CLIPModel
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.engine.hooks import OutputHook

def extract_keypoints(actions, path, test_list, experiment_num):
    # 0: Base of the spine -> 0
    # 1: Middle of the spine -> 1
    # 2: Neck -> 3
    # 3: Head -> 26
    # 4: Left shoulder -> 5
    # 5: Left elbow -> 6
    # 6: Left wrist -> 7
    # 7: Left hand -> 8
    # 8: Right shoulder -> 12
    # 9: Right elbow -> 13
    # 10: Right wrist -> 14
    # 11: Right hand -> 15
    # 12: Left hip -> 18
    # 13: Left knee -> 19
    # 14: Left ankle -> 20
    # 15: Left foot -> 21
    # 16: Right hip -> 22
    # 17: Right knee -> 23
    # 18: Right ankle -> 24
    # 19: Right foot -> 25
    # 20: Spine -> 2
    # 21: Left hand tip -> 9
    # 22: Left thumb -> 10
    # 23: Right hand tip -> 16
    # 24: Right thumb -> 17
    keypoint_pair = {0:0, 1:1, 2:3, 3:26, 4:5, 5:6, 6:7, 7:8, 8:12, 9:13, 10:14, 11:15, 12:18, 13:19,
                     14:20, 15:21, 16:22, 17:23, 18:24, 19:25, 20:2, 21:9, 22:10, 23:16, 24:17}
    folder_list = os.listdir(path)
    dataset = {"split": {"xsub_train":[], "xsub_test":[]}, "annotations": []}
    for folder in folder_list:
        location = folder[0:2]
        label = int(folder[2:4]) - 1
        subject = folder[4:6]
        data_file_list = os.listdir(os.path.join(path, folder))
        if label in actions:
            for data_file in data_file_list:
                joints = np.loadtxt(os.path.join(path, folder, data_file), delimiter=",", dtype=float, usecols=range(1,193))
                if joints.ndim < 2:
                    joints = joints.reshape((-1, joints.shape[0]))
                time_start, time_end = data_file.split("_")
                num_frames = joints.shape[0]
                num_subjects = 2
                num_keypoints = 25
                num_coords = 3

                if num_frames < 30:
                    continue

                print(f"Processing {location}{label:02d}{subject} between {time_start} and {time_end}...")

                keypoints = np.zeros((num_subjects, num_frames, num_keypoints, num_coords))
                for sub in range(num_subjects):
                    for frame in range(num_frames):
                        for keypoint_key, keypoint_value in keypoint_pair.items():
                            keypoints[sub, frame, keypoint_key, 0] = joints[frame, sub*32*num_coords+keypoint_value*num_coords]
                            keypoints[sub, frame, keypoint_key, 1] = joints[frame, sub*32*num_coords+keypoint_value*num_coords+1]
                            keypoints[sub, frame, keypoint_key, 2] = joints[frame, sub*32*num_coords+keypoint_value*num_coords+2]
                annotation = {"frame_dir": f"{location}{label:02d}{subject}/{time_start}_{time_end}", "label": label, "total_frames": num_frames,
                            "keypoint": keypoints}
                dataset["annotations"].append(annotation)
                
                if f"{location}{subject}" in test_list:
                # if location in train_location:
                    dataset["split"]["xsub_test"].append(f"{location}{label:02d}{subject}/{time_start}_{time_end}")
                else:
                    dataset["split"]["xsub_train"].append(f"{location}{label:02d}{subject}/{time_start}_{time_end}")

    
    os.makedirs(f".\experiment\experiment_{experiment_num}", exist_ok=True)
    pickle_filename = os.path.join(f".\experiment\experiment_{experiment_num}", f"experiment_{experiment_num}.pkl")

    print("Writing to pickle file...")
    with open(pickle_filename, "wb") as file: 
        pickle.dump(dataset, file)


def select_samples(categories, n):
    if n < len(categories):
        raise ValueError("n must be at least the number of categories to include each category.")
    
    selected = {cat: random.choice(samples) for cat, samples in categories.items()}
    chosen_samples = set(selected.values()) 

    all_samples = [s for cat_samples in categories.values() for s in cat_samples]
    remaining_samples = list(set(all_samples) - chosen_samples) 

    extra_samples = random.sample(remaining_samples, n - len(categories))

    return list(chosen_samples) + extra_samples


def extract_features(config_file_path, checkpoint_file_path, data_path, device, experiment_num, test_list_subject):
    
    model = init_recognizer(config_file_path, checkpoint_file_path, device=device)

    with open(data_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    
    num_samples = len(data['annotations'])
    num_train_sample = 300
    num_test_sample = 200
    labels_train = np.zeros(num_train_sample, dtype=np.int8)
    labels_test = np.zeros(num_test_sample, dtype=np.int8)
    features_train = np.zeros((num_train_sample, 512), dtype=np.float32)
    features_test = np.zeros((num_test_sample, 512), dtype=np.float32)
    train_tracker = 0
    test_tracker = 0

    with OutputHook(model.cls_head, outputs=['loss_cls', 'pool', 'fc']) as hook:
        for index in range(num_samples):
            result = inference_recognizer(model, data['annotations'][index])
        
            frame_dir = data['annotations'][index]['frame_dir'].split('/')
            loc = frame_dir[0][0:2]
            subject = frame_dir[0][-2:]
            label = data['annotations'][index]['label']
            
            if f'{loc}{subject}' in test_list_subject:
                if test_tracker < num_test_sample:
                    num_rows, num_cols, depth_1, depth_2 = hook.layer_outputs['pool'].shape
                    features_test[test_tracker, :] = np.reshape(hook.layer_outputs['pool'], (-1, num_rows*num_cols))
                    labels_test[test_tracker] = data['annotations'][index]['label']
                else:
                    num_rows, num_cols, depth_1, depth_2 = hook.layer_outputs['pool'].shape
                    features_test = np.vstack((features_test, np.reshape(hook.layer_outputs['pool'], (-1, num_rows*num_cols))))
                    labels_test = np.concatenate((labels_test, np.array([data['annotations'][index]['label']])))
                test_tracker = test_tracker + 1

            else:
                if train_tracker < num_train_sample:
                    num_rows, num_cols, depth_1, depth_2 = hook.layer_outputs['pool'].shape
                    features_train[train_tracker, :] = np.reshape(hook.layer_outputs['pool'], (-1, num_rows*num_cols))
                    labels_train[train_tracker] = data['annotations'][index]['label']
                else:
                    num_rows, num_cols, depth_1, depth_2 = hook.layer_outputs['pool'].shape
                    features_train = np.vstack((features_train, np.reshape(hook.layer_outputs['pool'], (-1, num_rows*num_cols))))
                    labels_train = np.concatenate((labels_train, np.array([data['annotations'][index]['label']])))
                train_tracker = train_tracker + 1

    np.save(os.path.join(f".\experiment\experiment_{experiment_num}", "train.npy"), features_train)
    np.save(os.path.join(f".\experiment\experiment_{experiment_num}", "train_label.npy"), labels_train)
    np.save(os.path.join(f".\experiment\experiment_{experiment_num}", "gtest.npy"), features_test)
    np.save(os.path.join(f".\experiment\experiment_{experiment_num}", "g_label.npy"), labels_test)

if __name__ == "__main__":
    pass