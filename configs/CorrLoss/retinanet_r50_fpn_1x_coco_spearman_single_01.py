_base_ =  '../retinanet/retinanet_r50_fpn_1x_coco.py'

# model settings
model = dict(
    bbox_head=dict(type='CorrRetinaHead', 
                   corr_w=0.10,
                   use_loss_single = 1,
                   corr_type = 'spearman'))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
