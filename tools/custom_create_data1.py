#!/usr/bin/env python3
import os
from os import path as osp
import argparse
import mmengine
import numpy as np
import pickle

# Define your class names and mapping.
pcdClass_names = ['Cone_Yellow', 'Cone_Blue', 'Cone_Orange', 'Cone_Big']
pcdCategories = {name: idx for idx, name in enumerate(pcdClass_names)}

def _fill_infos(root_path):
    """Generate info dicts for training, validation, and test splits.
    
    The output format is adapted to the KITTI format (excluding detailed image info and real calibration).
    Each sample will contain:
      - "image": a dict with dummy image info,
      - "point_cloud": a dict with point cloud info,
      - "calib": a dict with dummy calibration matrices,
      - "annos": a dict with annotation fields.
    """
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
    
    for i, point_file in enumerate(all_files):
        file_name, ext = osp.splitext(point_file)
        if ext.lower() != '.bin':
            continue  # Skip non-bin files

        try:
            file_id = str(int(file_name))
        except ValueError:
            print(f"Warning: Cannot convert {file_name} to integer, skipping.")
            continue

        # Construct relative paths.
        lidar_path = osp.join('points', file_name + '.bin')
        label_path = osp.join(root_path, 'labels', file_name + '.txt')
        
        # Check that the required files exist.
        mmengine.check_file_exist(osp.join(root_path, lidar_path))
        mmengine.check_file_exist(label_path)
        
        # --- Build basic info dictionary ---
        # We first build a temporary dict with our original fields.
        info_orig = {
            'sample_idx': i,
            'timestamp': i,
            'lidar_points': {},
            'instances': [],
        }
        info_orig['lidar_points']['lidar_path'] = lidar_path
        info_orig['lidar_points']['num_pts_feats'] = 4
        info_orig['lidar_points']['Tr_velo_cam'] = np.array([
            [0.79807554, 0.60254895, 0.00319398, 0.1853],
            [0.2647308, -0.34586413, -0.90016421, 0.12779],
            [-0.54128832, 0.71924458, -0.43553896, -0.12141],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        info_orig['lidar_points']['Tr_imu_to_velo'] = None

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
                # For 2D bbox, we use dummy zeros since we have no image.
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
                    # Original order in file: ann[0],ann[1],ann[2],ann[3],ann[4],ann[5],ann[6]
                    # We re-order as: location (x,y,z), dimensions (l, w, h), rotation_y
                    instance['bbox_3d'] = [
                        float(ann[0]),  # x
                        float(ann[1]),  # y
                        float(ann[2]),  # z
                        float(ann[5]),  # l (we ignore the value and later override for cone size)
                        float(ann[4]),  # w
                        float(ann[3]),  # h
                        float(ann[6])   # rotation_y
                    ]
                except Exception as e:
                    print(f"Error parsing line in {label_path}: {line.strip()}")
                    continue
                instance['num_lidar_pts'] = None
                info_orig['instances'].append(instance)
        
        print(f"Sample {file_name}: {len(info_orig['instances'])} instances")
        
        # --- Convert to KITTI format ---
        # Create dummy image info.
        image_dict = {
            'image_idx': i,
            'image_path': "",  # No image available
            'image_shape': np.array([375, 1242], dtype=np.int32)
        }
        # Create point_cloud info.
        point_cloud_dict = {
            'num_features': info_orig['lidar_points']['num_pts_feats'],
            'velodyne_path': lidar_path
        }
        # Create a dummy calibration dict.
        calib_dict = {
            'P0': np.eye(4, dtype=np.float32),
            'P1': np.eye(4, dtype=np.float32),
            'P2': np.eye(4, dtype=np.float32),
            'P3': np.eye(4, dtype=np.float32),
            'R0_rect': np.eye(4, dtype=np.float32),
            'Tr_velo_to_cam': info_orig['lidar_points']['Tr_velo_cam'][:3, :],
            'Tr_imu_to_velo': (info_orig['lidar_points']['Tr_imu_to_velo'] 
                               if info_orig['lidar_points']['Tr_imu_to_velo'] is not None 
                               else np.eye(4, dtype=np.float32))
        }
        # Process annotations (convert "instances" to KITTI "annos" format).
        annos = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': []
        }
        # For each instance, we fill KITTI fields.
        for inst in info_orig['instances']:
            annos['name'].append(pcdClass_names[inst['bbox_label']])
            annos['truncated'].append(inst['truncated'])
            annos['occluded'].append(inst['occluded'])
            annos['alpha'].append(inst['alpha'])
            # Since there is no image, use dummy 2D bbox (zeros).
            annos['bbox'].append(inst['bbox'])
            # For dimensions, we use a fixed cone size.
            annos['dimensions'].append([0.23, 0.23, 0.23])
            # Location is taken from the first three elements of bbox_3d.
            annos['location'].append(inst['bbox_3d'][:3])
            # Rotation_y is the last element.
            annos['rotation_y'].append(inst['bbox_3d'][6])
            annos['score'].append(inst['score'])
        # Convert lists to numpy arrays.
        for k, v in annos.items():
            annos[k] = np.array(v, dtype=np.float32) if k != 'name' else np.array(v)

        # Build the final KITTI-style info.
        info_kitti = {
            'image': image_dict,
            'point_cloud': point_cloud_dict,
            'calib': calib_dict,
            'annos': annos
        }
        
        # Append to corresponding split based on file_id.
        if file_id in train_ids:
            train_infos.append(info_kitti)
        elif file_id in val_ids:
            val_infos.append(info_kitti)
        elif file_id in test_ids:
            test_infos.append(info_kitti)
        else:
            print(f"Warning: {file_id} not found in any split file, skipping.")
    
    return train_infos, val_infos, test_infos

def create_custom_dataset_infos(root_path, info_prefix):
    train_infos, val_infos, test_infos = _fill_infos(root_path)
    metainfo = {
        'classes': pcdClass_names,
        'categories': {name: idx for idx, name in enumerate(pcdClass_names)},
        'dataset': 'vargarda',
        'info_version': 1.0,
    }
    
    # Save as a dict with keys: data_list and metainfo.
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
        description='Custom dataset info generator for LiDAR-based 3D detection in vargarda (KITTI-like format)')
    parser.add_argument('--root-path', type=str, default='./data/vargarda',
                        help='Root path of the dataset (should contain ImageSets, points, and labels folders)')
    parser.add_argument('--extra-tag', type=str, default='custom',
                        help='Extra tag used for naming the output info files')
    args = parser.parse_args()
    create_custom_dataset_infos(args.root_path, args.extra_tag)
