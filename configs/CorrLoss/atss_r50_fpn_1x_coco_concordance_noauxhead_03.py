_base_ =  '../atss/atss_r50_fpn_1x_coco.py'

# model settings
model = dict(
    bbox_head=dict(type='CorrATSSHead',
                   corr_w=0.30,
                   use_loss_single=0,
                   corr_type='concordance',
                   loss_centerness=dict(
                       type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0)
                   ))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
