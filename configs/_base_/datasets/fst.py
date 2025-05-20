custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet3d.datasets.custom_kitti_dataset',
    ])

# dataset settings
dataset_type = 'CustomKittiDataset'
data_root = 'data/fst/'
class_names = ['Cone_Yellow', 'Cone_Blue', 'Cone_Orange', 'Cone_Big']  # replace with your dataset class
#point_cloud_range = [-50, -50, -5, 50, 50, 30] # adjust according to your dataset
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        dataset=dict(
            ann_file='custom_infos_train_lidar.pkl',
            box_type_3d='LiDAR',
            data_prefix=dict(pts=''),
            data_root='data/fst/',
            metainfo=dict(
                categories=dict(
                    Cone_Big=3, Cone_Blue=1, Cone_Orange=2, Cone_Yellow=0),
                classes=[
                    'Cone_Yellow',
                    'Cone_Blue',
                    'Cone_Orange',
                    'Cone_Big',
                ]),
            modality=dict(use_camera=False, use_lidar=True),
            pipeline=[
                dict(
                    coord_type='LIDAR',
                    load_dim=4,
                    type='LoadPointsFromFile',
                    use_dim=4),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    global_rot_range=[
                        0.0,
                        0.0,
                    ],
                    num_try=100,
                    rot_range=[
                        -0.78539816,
                        0.78539816,
                    ],
                    translation_std=[
                        1.0,
                        1.0,
                        0.5,
                    ],
                    type='ObjectNoise'),
                dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
                dict(
                    rot_range=[
                        -0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        0.95,
                        1.05,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(type='PointShuffle'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type='CustomKittiDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        global_rot_range=[
            0.0,
            0.0,
        ],
        num_try=100,
        rot_range=[
            -0.78539816,
            0.78539816,
        ],
        translation_std=[
            1.0,
            1.0,
            0.5,
        ],
        type='ObjectNoise'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.78539816,
            0.78539816,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        type='GlobalRotScaleTrans'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]


test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='custom_infos_test_lidar.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(pts=''),
        data_root='data/fst/',
        metainfo=dict(
            categories=dict(
                Cone_Big=3, Cone_Blue=1, Cone_Orange=2, Cone_Yellow=0),
            classes=[
                'Cone_Yellow',
                'Cone_Blue',
                'Cone_Orange',
                'Cone_Big',
            ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='CustomKittiDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/fst/custom_infos_test_lidar.pkl',
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='custom_infos_val_lidar.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(pts=''),
        data_root='data/fst/',
        metainfo=dict(
            categories=dict(
                Cone_Big=3, Cone_Blue=1, Cone_Orange=2, Cone_Yellow=0),
            classes=[
                'Cone_Yellow',
                'Cone_Blue',
                'Cone_Orange',
                'Cone_Big',
            ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='CustomKittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/fst/custom_infos_val_lidar.pkl',
    metric='bbox',
    type='KittiMetric')
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(save_dir='viz', type='LocalVisBackend'),
    ])