_base_ = '../foveabox/fovea_r50_fpn_4x4_1x_coco.py'

# model settings
model = dict(
    bbox_head=dict(
        type='CorrFoveaHead',
        corr_w=0.2,
        corr_type='concordance'
        ))

