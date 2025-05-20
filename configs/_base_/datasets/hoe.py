custom_imports = dict(imports=['mmdet3d.datasets.custom_kitti_dataset'], allow_failed_imports=False)

# Dataset settings.
dataset_type = 'CustomKittiDataset'
data_root = 'data/vargarda/'  # Adjust as needed.
class_names = ['Cone_Yellow', 'Cone_Blue', 'Cone_Orange', 'Cone_Big']
metainfo = dict(
    classes=class_names,
    categories={name: idx for idx, name in enumerate(class_names)}
)
input_modality = dict(use_lidar=True, use_camera=False)

# IMPORTANT:
# Our info file now stores the full relative path (e.g., "points/0000000.bin").
# In this case, set data_prefix to an empty string so that the loader does:
# osp.join(data_root, "", "points/0000000.bin") → "data/vargarda/points/0000000.bin"
data_prefix = dict(pts='')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='custom_infos_train_lidar.pkl',
            data_prefix=data_prefix,
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='custom_infos_val_lidar.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val_lidar.pkl',
    metric='bbox')
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='custom_infos_test_lidar.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'
    )
)
test_evaluator = dict(
    type='KittiMetric',
    ann_file='data/vargarda/custom_infos_test_lidar.pkl',
    metric='bbox'
)
#test_dataloader = val_dataloader
#test_evaluator = val_evaluator
visualizer = dict(
    type='Det3DLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend', save_dir='viz')]
)

