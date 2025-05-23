auto_scale_lr = dict(base_batch_size=48, enable=False)
class_names = [
    'Cone_Yellow',
    'Cone_Blue',
    'Cone_Orange',
    'Cone_Big',
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet3d.datasets.custom_kitti_dataset',
    ])
data_root = 'data/vargarda/'
dataset_type = 'CustomKittiDataset'
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.0018
metainfo = dict(classes=[
    'Cone_Yellow',
    'Cone_Blue',
    'Cone_Orange',
    'Cone_Big',
])
model = dict(
    backbone=dict(
        in_channels=64,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            2,
            2,
            2,
        ],
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    bbox_head=dict(
        anchor_generator=dict(
            ranges=[
                [
                    0,
                    -39.68,
                    -0.6,
                    69.12,
                    39.68,
                    -0.6,
                ],
                [
                    0,
                    -39.68,
                    -0.6,
                    69.12,
                    39.68,
                    -0.6,
                ],
                [
                    0,
                    -39.68,
                    -1.78,
                    69.12,
                    39.68,
                    -1.78,
                ],
                [
                    0,
                    -39.68,
                    -1.78,
                    69.12,
                    39.68,
                    -1.78,
                ],
            ],
            reshape_out=False,
            rotations=[
                0,
                1.57,
            ],
            sizes=[
                [
                    0.23,
                    0.23,
                    0.23,
                ],
                [
                    0.23,
                    0.23,
                    0.23,
                ],
                [
                    0.23,
                    0.23,
                    0.23,
                ],
                [
                    0.23,
                    0.23,
                    0.23,
                ],
            ],
            type='AlignedAnchor3DRangeGenerator'),
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        diff_rad_by_sin=True,
        feat_channels=384,
        in_channels=384,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=2.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_dir=dict(
            loss_weight=0.2, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_classes=4,
        type='Anchor3DHead',
        use_direction_classifier=True),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            max_voxels=(
                16000,
                40000,
            ),
            point_cloud_range=[
                0,
                -39.68,
                -3,
                69.12,
                39.68,
                1,
            ],
            voxel_size=[
                0.16,
                0.16,
                4,
            ])),
    middle_encoder=dict(
        in_channels=64, output_shape=[
            496,
            432,
        ], type='PointPillarsScatter'),
    neck=dict(
        in_channels=[
            64,
            128,
            256,
        ],
        out_channels=[
            128,
            128,
            128,
        ],
        type='SECONDFPN',
        upsample_strides=[
            1,
            2,
            4,
        ]),
    test_cfg=dict(
        max_num=50,
        min_bbox_size=0,
        nms_across_levels=False,
        nms_pre=100,
        nms_thr=0.1,
        score_thr=0.1,
        use_rotate_nms=True),
    train_cfg=dict(
        allowed_border=0,
        assigner=[
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                min_pos_iou=0.1,
                neg_iou_thr=0.1,
                pos_iou_thr=0.1,
                type='Max3DIoUAssigner'),
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                min_pos_iou=0.1,
                neg_iou_thr=0.1,
                pos_iou_thr=0.1,
                type='Max3DIoUAssigner'),
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                min_pos_iou=0.1,
                neg_iou_thr=0.1,
                pos_iou_thr=0.5,
                type='Max3DIoUAssigner'),
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                min_pos_iou=0.1,
                neg_iou_thr=0.1,
                pos_iou_thr=0.5,
                type='Max3DIoUAssigner'),
        ],
        debug=False,
        pos_weight=-1),
    type='VoxelNet',
    voxel_encoder=dict(
        feat_channels=[
            64,
        ],
        in_channels=4,
        point_cloud_range=[
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ],
        type='PillarFeatureNet',
        voxel_size=[
            0.16,
            0.16,
            4,
        ],
        with_distance=False))
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        betas=(
            0.95,
            0.99,
        ), lr=0.0018, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=16,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=16,
        eta_min=0.018,
        type='CosineAnnealingLR'),
    dict(
        T_max=24,
        begin=16,
        by_epoch=True,
        convert_to_iter_based=True,
        end=40,
        eta_min=1.8e-07,
        type='CosineAnnealingLR'),
    dict(
        T_max=16,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=16,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=24,
        begin=16,
        by_epoch=True,
        convert_to_iter_based=True,
        end=40,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]
point_cloud_range = [
    0,
    -39.68,
    -3,
    69.12,
    39.68,
    1,
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='custom_infos_test_lidar.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(pts=''),
        data_root='data/vargarda/',
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
    ann_file='data/vargarda/custom_infos_test_lidar.pkl',
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(coord_type='LIDAR', load_dim=4, type='LoadPointsFromFile', use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        dataset=dict(
            ann_file='custom_infos_train_lidar.pkl',
            box_type_3d='LiDAR',
            data_prefix=dict(pts=''),
            data_root='data/vargarda/',
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
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='custom_infos_val_lidar.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(pts=''),
        data_root='data/vargarda/',
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
    ann_file='data/vargarda/custom_infos_val_lidar.pkl',
    metric='bbox',
    type='KittiMetric')
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(save_dir='viz', type='LocalVisBackend'),
    ])
voxel_size = [
    0.16,
    0.16,
    4,
]
work_dir = './work_dirs/pointpillars_custom'
