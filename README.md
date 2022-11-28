
# Correlation Loss: Enforcing Correlation between Classification and Localization


The official implementation of Correlation Loss. Our implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

> [**Correlation Loss: Enforcing Correlation between Classification and Localization**](https://arxiv.org/abs/),            
> Fehmi Kahraman, [Kemal Oksuz](https://kemaloksuz.github.io/), [Sinan Kalkan](http://www.kovan.ceng.metu.edu.tr/~sinan/), [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/),
> *AAAI 2023.*

## How to Cite

Please cite the paper if you benefit from our paper or the repository:
```
@inproceedings{CorrLoss,
       title = {Correlation Loss: Enforcing Correlation between Classification and Localization},
       author = {Fehmi Kahraman and Kemal Oksuz and Sinan Kalkan and Emre Akbas},
       booktitle = {Association for the Advancement of Artificial Intelligence (AAAI)},
       year = {2023}
}
```


## Specification of Dependencies and Preparation

- Please see [get_started.md](docs/get_started.md) for requirements and installation of MMDetection.
- Please refer to [introduction.md](docs/1_exist_data_model.md) for dataset preparation and basic usage of MMDetection.



## Training Code
The configuration files of all models listed above can be found in the `configs/CorrLoss` folder. You can follow [introduction.md](docs/1_exist_data_model.md) for training code. As an example, to train Sparse R-CNN with our Correlation Loss on 4 GPUs as we did, use the following command:

```
./tools/dist_train.sh configs/CorrLoss/sparse_rcnn_r50_fpn_1x_coco_spearman_02.py 4
```

## Test Code
The configuration files of all models listed above can be found in the `configs/CorrLoss` folder. You can follow [introduction.md](docs/1_exist_data_model.md) for test code. As an example, first download a trained model using the links provided in the tables or you train a model, then run the following command to test an object detection model on multiple GPUs:

```
./tools/dist_test.sh configs/CorrLoss/sparse_rcnn_r50_fpn_1x_coco_spearman_02.py ${CHECKPOINT_FILE} 4 --eval bbox 
```
and use the following command to test an instance segmentation model on multiple GPUs:

```
./tools/dist_test.sh configs/CorrLoss/yolact_r50_4x8_coco_spearman_02.py ${CHECKPOINT_FILE} 4 --eval bbox segm 
```
You can also test a model on a single GPU with the following example command:
```
python tools/test.py configs/CorrLoss/sparse_rcnn_r50_fpn_1x_coco_spearman_02.py ${CHECKPOINT_FILE} --eval bbox 
```


## Details for Correlation Loss Implementation

Below is the links to the most relevant files that can be useful check out the details of the implementation:
- [CorrATSSHead and implementation of the Correlation Loss](mmdet/models/dense_heads/corr_atss_head.py)
- [CorrFoveaHead](mmdet/models/dense_heads/corr_fovea_head.py)
- [CorrPAAHead](mmdet/models/dense_heads/corr_paa_head.py)
- [DIIHead for Corr Sparse-RCNN](mmdet/models/roi_heads/bbox_heads/dii_head.py)
- [Config files folder of Correlation Loss](configs/CorrLoss)
