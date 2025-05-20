#!/usr/bin/env python3
import os
from os import path as osp
import argparse
import mmengine
import numpy as np

# Define your class names and mapping.
pcdClass_names = ['Cone_Yellow', 'Cone_Blue', 'Cone_Orange', 'Cone_Big']
pcdCategories = {name: idx for idx, name in enumerate(pcdClass_names)}

def _fill_infos(root_path):
    """Generate info dicts for training, validation, and test splits."""
    train_infos = []
    val_infos = []
    test_infos = []

    # Define paths for the split files.
    trainSet = osp.join(root_path, 'ImageSets', 'train.txt')
    valSet = osp.join(root_path, 'ImageSets', 'val.txt')
    testSet = osp.join(root_path, 'ImageSets', 'test.txt')

    # Read split file IDs.
    train_ids = set()
    val_ids = set()
    test_ids = set()
    if osp.exists(trainSet):
        with open(trainSet, 'r', encoding='utf-8') as f:
            for line in f:
                train_ids.add(line.strip())
    if osp.exists(valSet):
        with open(valSet, 'r', encoding='utf-8') as f:
            for line in f:
                val_ids.add(line.strip())
    if osp.exists(testSet):
        with open(testSet, 'r', encoding='utf-8') as f:
            for line in f:
                test_ids.add(line.strip())

    # List all .bin files in the 'points' folder.
    points_dir = osp.join(root_path, 'points')
    all_files = sorted(os.listdir(points_dir))
    
    # Define dummy calibration matrices for KITTI evaluation.
    dummy_cam2img = np.array([
        [707.0493, 0, 604.0814, 0],
        [0, 707.0493, 180.5066, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    dummy_lidar2cam = np.array([
        [0.00692796, -0.9999722, -0.00275783, -0.02457729],
        [-0.00116298, 0.00274984, -0.9999955, -0.06127237],
        [0.9999753, 0.00693114, -0.0011439, -0.3321029]
    ], dtype=np.float32)

    for i, point_file in enumerate(all_files):
        file_name, ext = osp.splitext(point_file)
        if ext.lower() != '.bin':
            continue  # Skip non-bin files

        try:
            file_id = str(int(file_name))
        except ValueError:
            print(f"Warning: Cannot convert {file_name} to integer, skipping.")
            continue

        # Store the relative path instead of the full absolute path.
        lidar_path = osp.join('points', file_name + '.bin')
        label_path = osp.join(root_path, 'labels', file_name + '.txt')
        
        # Check that the required files exist.
        mmengine.check_file_exist(osp.join(root_path, lidar_path))
        mmengine.check_file_exist(label_path)
        
        # Create the info dict for this sample.
        info = {
            'sample_idx': i,
            'timestamp': i,
            'lidar_points': {},
            'instances': [],
        }
        info['lidar_points']['lidar_path'] = lidar_path
        info['lidar_points']['num_pts_feats'] = 4
        info['lidar_points']['Tr_velo_cam'] = np.array([
            [0.79807554, 0.60254895, 0.00319398, 0.1853],
            [0.2647308, -0.34586413, -0.90016421, 0.12779],
            [-0.54128832, 0.71924458, -0.43553896, -0.12141],
            [0, 0, 0, 1]
        ])
        info['lidar_points']['Tr_imu_to_velo'] = None

        # Process the label file.
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                ann = line.strip().split()
                if not ann:
                    continue
                if len(ann) != 8:
                    print(f"Warning: Skipping line in {label_path} due to unexpected format: {line.strip()}")
                    continue
                instance = {}
                instance['bbox'] = [0, 0, 0, 0]
                label_str = ann[7].strip()
                if label_str not in pcdCategories:
                    print(f"Warning: Label '{label_str}' not found in mapping. Skipping line: {line.strip()}")
                    continue
                label_int = pcdCategories[label_str]
                instance['bbox_label'] = label_int
                instance['bbox_label_3d'] = label_int
                instance['truncated'] = 0.0
                instance['occluded'] = 0
                instance['alpha'] = 0.0
                instance['score'] = 1.0
                
                try:
                    instance['bbox_3d'] = [
                        float(ann[0]),
                        float(ann[1]),
                        float(ann[2]),
                        float(ann[5]),
                        float(ann[4]),
                        float(ann[3]),
                        float(ann[6])
                    ]
                except Exception as e:
                    print(f"Error parsing line in {label_path}: {line.strip()}")
                    continue
                
                instance['num_lidar_pts'] = None
                
                # Add a default difficulty level (0 for easy, adjust if needed)
                instance['difficulty'] = 0
                
                info['instances'].append(instance)
        
        # Inject a dummy "images" field (for KITTI evaluation)
        info['images'] = {
            'CAM2': {
                'height': 375,
                'width': 1242,
                'cam2img': dummy_cam2img,
                'lidar2cam': dummy_lidar2cam
            }
        }
        
        print(f"Sample {file_name}: {len(info['instances'])} instances")
        
        if file_id in train_ids:
            train_infos.append(info)
        elif file_id in val_ids:
            val_infos.append(info)
        elif file_id in test_ids:
            test_infos.append(info)
        else:
            print(f"Warning: {file_id} not found in any split file, skipping.")
    
    return train_infos, val_infos, test_infos

def create_custom_dataset_infos(root_path, info_prefix):
    train_infos, val_infos, test_infos = _fill_infos(root_path)
    metainfo = {
        'classes': pcdClass_names,
        'categories': {name: idx for idx, name in enumerate(pcdClass_names)},
        'dataset': 'custom_dataset',
        'info_version': 1.0,
    }
    
    if train_infos:
        data = dict(data_list=train_infos, metainfo=metainfo)
        train_info_path = osp.join(root_path, f'{info_prefix}_infos_train_lidar.pkl')
        mmengine.dump(data, train_info_path)
        print(f"Train info saved to {train_info_path}")
    else:
        print("No training infos generated!")
    
    if val_infos:
        data = dict(data_list=val_infos, metainfo=metainfo)
        val_info_path = osp.join(root_path, f'{info_prefix}_infos_val_lidar.pkl')
        mmengine.dump(data, val_info_path)
        print(f"Validation info saved to {val_info_path}")
    else:
        print("No validation infos generated!")
    
    if test_infos:
        data = dict(data_list=test_infos, metainfo=metainfo)
        test_info_path = osp.join(root_path, f'{info_prefix}_infos_test_lidar.pkl')
        mmengine.dump(data, test_info_path)
        print(f"Test info saved to {test_info_path}")
    else:
        print("No test infos generated!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Custom dataset info generator for LiDAR-only data')
    parser.add_argument('--root-path', type=str, default='./data/vargarda',
                        help='Root path of the dataset (should contain ImageSets, points, and labels folders)')
    parser.add_argument('--extra-tag', type=str, default='custom',
                        help='Extra tag used for naming the output info files')
    args = parser.parse_args()
    create_custom_dataset_infos(args.root_path, args.extra_tag)
