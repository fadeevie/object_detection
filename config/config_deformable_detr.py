from mmengine.config import Config

cfg = Config({
    'dataset_type': 'CocoDataset',
    'data_root': '/content/drive/MyDrive/eff_dataset',
    'train_dataloader': {
        'dataset': {
            'type': 'CocoDataset',
            'data_root': '/content/drive/MyDrive/eff_dataset',
            'ann_file': 'annotations/instances_train_fixed.json',
            'data_prefix': {'img': 'images/train/'},
            'filter_cfg': {'filter_empty_gt': True},
            'metainfo': {'classes': ('window',), 'palette': [(220, 20, 60)]},
        },
        'batch_size': 4,
        'num_workers': 4,
    },
    'val_dataloader': {
        'dataset': {
            'type': 'CocoDataset',
            'data_root': '/content/drive/MyDrive/eff_dataset',
            'ann_file': 'annotations/instances_val_fixed.json',
            'data_prefix': {'img': 'images/val/'},
            'metainfo': {'classes': ('window',), 'palette': [(220, 20, 60)]},
        },
        'batch_size': 2,
        'num_workers': 2,
    },
    'test_dataloader': {
        'dataset': {
            'type': 'CocoDataset',
            'data_root': '/content/drive/MyDrive/eff_dataset',
            'ann_file': 'annotations/instances_val_fixed.json',
            'data_prefix': {'img': 'images/val/'},
            'metainfo': {'classes': ('window',), 'palette': [(220, 20, 60)]},
        },
        'batch_size': 2,
        'num_workers': 2,
    },
    'model': {
        'type': 'DeformableDETR',
        'bbox_head': {'num_classes': 1},
    },
    'val_evaluator': {
        'type': 'CocoMetric',
        'ann_file': '/content/drive/MyDrive/eff_dataset/annotations/instances_val_fixed.json',
        'metric': 'bbox',
    },
    'test_evaluator': {
        'type': 'CocoMetric',
        'ann_file': '/content/drive/MyDrive/eff_dataset/annotations/instances_val_fixed.json',
        'metric': 'bbox',
    },
    'optim_wrapper': {
        'type': 'OptimWrapper',
        'optimizer': {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0001},
        'clip_grad': {'max_norm': 0.1, 'norm_type': 2},
    },
    'param_scheduler': [
        {
            'type': 'MultiStepLR',
            'begin': 0,
            'end': 50,
            'by_epoch': True,
            'milestones': [8, 10],
            'gamma': 0.1,
        }
    ],
    'train_cfg': {'type': 'EpochBasedTrainLoop', 'max_epochs': 50},
    'val_cfg': {'type': 'ValLoop'},
    'test_cfg': {'type': 'TestLoop'},
    'default_hooks': {
        'logger': {'type': 'LoggerHook', interval=50},
        'checkpoint': {'type': 'CheckpointHook', interval=1},
    },
    'gpu_ids': range(1),
    'device': 'cuda',
})