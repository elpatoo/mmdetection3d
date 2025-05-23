# custom_kitti_dataset.py
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np

from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes

@DATASETS.register_module()
class CustomKittiDataset(KittiDataset):
    """
    Custom KITTI Dataset for LiDAR-based 3D object detection with custom classes.

    This dataset overrides the default METAINFO and re-implements annotation parsing
    so that string labels are converted to integer indices using a custom mapping.
    It also bypasses the image-related calibration that is not available in LiDAR-only data.
    """
    # Define the meta information. Note: categories is now a dict.
    METAINFO = {
        'classes': ('Cone_Yellow', 'Cone_Blue', 'Cone_Orange', 'Cone_Big'),
        'categories': {'Cone_Yellow': 0, 'Cone_Blue': 1, 'Cone_Orange': 2, 'Cone_Big': 3},
        'palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    }

    def __init__(self, *args, **kwargs):
        # Remove 'filter_empty_gt' if present (if not supported)
        kwargs.pop('filter_empty_gt', None)
        self.METAINFO = CustomKittiDataset.METAINFO
        super().__init__(*args, **kwargs)
        # Override label_mapping: map each class string to its integer index.
        self.label_mapping = {cls: idx for idx, cls in enumerate(self.METAINFO['classes'])}
        self.num_ins_per_cat = [0] * len(self.METAINFO['classes'])
        # Pre-parse annotations for each data sample.
        for info in self.data_list:
            self.parse_ann_info(info)

    def parse_ann_info(self, info: dict) -> dict:
        """Parse annotation information from a single data sample.

        This override does not rely on any image information.
        """
        instances = info.get('instances', [])
        if len(instances) == 0:
            return None

        gt_bboxes_3d = []
        gt_labels_3d = []
        for inst in instances:
            # We assume each instance has a 'bbox_3d' and a label already as an integer.
            if 'bbox_3d' in inst and 'bbox_label_3d' in inst:
                gt_bboxes_3d.append(inst['bbox_3d'])
                label = inst['bbox_label_3d']
                # If label is a string, convert it; if already int, leave it.
                if isinstance(label, str):
                    label = label.strip()
                    label = self.label_mapping.get(label, -1)
                gt_labels_3d.append(label)
        if len(gt_bboxes_3d) == 0:
            ann_info = {
                'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                'gt_labels_3d': np.zeros(0, dtype=np.int64)
            }
        else:
            ann_info = {
                'gt_bboxes_3d': LiDARInstance3DBoxes(np.array(gt_bboxes_3d, dtype=np.float32)),
                'gt_labels_3d': np.array(gt_labels_3d, dtype=np.int64)
            }
            for label in gt_labels_3d:
                if label != -1:
                    self.num_ins_per_cat[label] += 1

        # Save the raw instance annotations as well.
        ann_info['instances'] = copy.deepcopy(instances)
        return ann_info

def parse_data_info(self, info: dict) -> dict:
    # Parse annotation info using our custom method
    ann_info = self.parse_ann_info(info)
    info['ann_info'] = ann_info

    # Inject a dummy "images" field so that evaluation (e.g., KITTI metric) doesn't crash.
    if 'images' not in info:
        info['images'] = {
            'CAM2': {
                'height': 375,
                'width': 1242,
                'cam2img': None,
                'lidar2cam': None
            }
        }

    # Add a top-level "lidar_path" key (for visualization)
    #if 'lidar_points' in info:
      #  info['lidar_path'] = info['lidar_points']  # Make sure this is correct for your data

    return info

