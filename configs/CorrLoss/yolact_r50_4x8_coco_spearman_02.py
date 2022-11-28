_base_ = './yolact_r50_4x8_coco.py'

model = dict(
    bbox_head=dict(
        type='CorrYOLACTHead',
        corr_w=0.20,
        corr_type='spearman',
    ))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4
)
