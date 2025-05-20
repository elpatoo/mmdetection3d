_base_ = [
    '../_base_/schedules/cyclic-20e.py', 
    '../_base_/default_runtime.py'
]

# -----------------------------------------------------------------------------
# Dataset & Pipeline Settings (Ultra-light)
# -----------------------------------------------------------------------------
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
dataset_type = 'CustomKittiDataset'
data_root = 'data/vargarda/'
class_names = ['Cone_Yellow', 'Cone_Blue', 'Cone_Orange', 'Cone_Big']
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# Use minimal sampling: one instance per class.
db_sampler = dict(
    classes=class_names,
    data_root=data_root,
    info_path=data_root + 'custom_infos_train_lidar.pkl',
    rate=1.0,
    prepare=dict(),
    sample_groups=dict(
        Cone_Yellow=1, Cone_Blue=1, Cone_Orange=0, Cone_Big=0
    ),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args
    ),
    backend_args=backend_args
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile', 
        coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=1,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile', 
        coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]
            ),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', 
                point_cloud_range=point_cloud_range
            )
        ]
    ),
    dict(type='Pack3DDetInputs', keys=['points'])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile', 
        coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args
    ),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='custom_infos_train_lidar.pkl',
            data_prefix=dict(pts=''),
            pipeline=train_pipeline,
            modality=input_modality,
            metainfo=dict(classes=class_names),
            box_type_3d='LiDAR',
            test_mode=False,
            backend_args=backend_args
        )
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='custom_infos_val_lidar.pkl',
        data_prefix=dict(pts='points'),
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR',
        test_mode=True,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='custom_infos_val_lidar.pkl',
        data_prefix=dict(pts='points'),
        pipeline=eval_pipeline,
        modality=input_modality,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR',
        test_mode=True,
        backend_args=backend_args
    )
)

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val_lidar.pkl',
    metric='bbox',
    backend_args=backend_args
)
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=0.001))
find_unused_parameters = True
auto_scale_lr = dict(enable=False, base_batch_size=1)

# -----------------------------------------------------------------------------
# Model Settings (Ultra-light)
# -----------------------------------------------------------------------------

# Use very coarse voxelization: fewer voxels, lower resolution.
# Voxel size [10, 10, 10] over the same point_cloud_range gives a grid:
#   x: ceil((70.4-0)/10)=8, y: ceil((40-(-40))/10)=8, z: ceil((1-(-3))/10)=1  --> sparse_shape=[1,8,8]
voxel_size = [10, 10, 10]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
sparse_shape = [1, 8, 8]

model = dict(
    type='PartA2',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(20, 20)  # drastically reduced maximum voxels
        )
    ),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseUNet',
        in_channels=4,
        sparse_shape=sparse_shape,  # [1, 8, 8] grid
        order=('conv', 'norm', 'act')
    ),
    backbone=dict(
        type='SECOND',
        in_channels=16,            # reduced input channels
        layer_nums=[2, 2],         # fewer layers
        layer_strides=[1, 2],
        out_channels=[16, 32],     # reduced feature channels
        norm_cfg=dict(type='GN', num_groups=4, requires_grad=True)
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[16, 32],
        upsample_strides=[1, 2],
        out_channels=[32, 32]
    ),
    rpn_head=dict(
        type='PartA2RPNHead',
        num_classes=4,
        in_channels=64,            # 32+32 from neck outputs
        feat_channels=64,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [point_cloud_range[0], point_cloud_range[1], -0.6, point_cloud_range[3], point_cloud_range[4], -0.6],
                [point_cloud_range[0], point_cloud_range[1], -0.6, point_cloud_range[3], point_cloud_range[4], -0.6],
                [point_cloud_range[0], point_cloud_range[1], -1.78, point_cloud_range[3], point_cloud_range[4], -1.78],
                [point_cloud_range[0], point_cloud_range[1], -1.78, point_cloud_range[3], point_cloud_range[4], -1.78]
            ],
            sizes=[[0.23, 0.23, 0.23]] * 4,
            rotations=[0, 1.57],
            reshape_out=False
        ),
        diff_rad_by_sin=True,
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', 
            beta=1.0 / 9.0, 
            loss_weight=2.0
        ),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.2
        )
    ),
    roi_head=dict(
        type='PartAggregationROIHead',
        num_classes=4,
        semantic_head=dict(
            type='PointwiseSemanticHead',
            in_channels=4,           # reduced input channels for semantic head
            extra_width=0.2,
            seg_score_thr=0.3,
            num_classes=4,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0
            ),
            loss_part=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0
            )
        ),
        seg_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=7,  # reduced output resolution (from 14)
                max_pts_per_voxel=64,
                mode='max'
            )
        ),
        bbox_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=7,  # reduced output resolution (from 14)
                max_pts_per_voxel=64,
                mode='avg'
            )
        ),
        bbox_head=dict(
            type='PartA2BboxHead',
            num_classes=4,
            seg_in_channels=4,       # reduced from 16
            part_in_channels=4,
            seg_conv_channels=[16, 16],
            part_conv_channels=[16, 16],
            merge_conv_channels=[32, 32],
            down_conv_channels=[32, 64],
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            shared_fc_channels=[64, 64],  # much smaller fully connected layers
            cls_channels=[32, 32],
            reg_channels=[32, 32],
            dropout_ratio=0.1,
            roi_feat_size=7,         # matches the ROI extractor output size
            with_corner_loss=True,
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0
            ),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0
            )
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=[
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                )
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=512,
            nms_post=50,
            max_num=50,
            nms_thr=0.8,
            score_thr=0,
            use_rotate_nms=False
        ),
        rcnn=dict(
            assigner=[
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1
                ),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1
                )
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=32,
                pos_fraction=0.55,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True
            ),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=256,
            nms_post=20,
            max_num=20,
            nms_thr=0.7,
            score_thr=0,
            use_rotate_nms=True
        ),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.01,
            score_thr=0.1
        )
    )
)
