_base_ = './paa_r50_fpn_1x_coco.py'

# model settings
model = dict(
    bbox_head=dict(type='CorrPAAHead',
                   corr_w=0.3,
                   corr_type='concordance',
                   loss_centerness=dict(
                       type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0)
                   ))  # concordance, pearson or spearman
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
