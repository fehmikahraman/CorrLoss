_base_ =  './atss_r50_fpn_1x_coco.py'

# model settings
model = dict(
    bbox_head=dict(type='CorrATSSHead',
                   corr_w=0.20,
                   use_loss_single=0,
                   corr_type='concordance'))  # concordance, pearson or spearman
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
