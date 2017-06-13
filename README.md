# Confident Multiple Choice Learning

This code is for the paper "Confident Multiple Choice Learning".

## Preliminaries

It is tested under Ubuntu Linux 16.04.1 and Python 2.7 environment, and requries following Python packages to be installed:

* [TensorFlow](https://github.com/tensorflow/tensorflow): version 1.0.0 or above. Only GPU version is available.
* [Torchfile](https://github.com/bshillingford/python-torchfile): version 0.0.2 or above.

*Simple torchfile installation:*

    pip install torchfile

## Dataset 

We provide the following datasets in torch format:

* CIFAR-10 whitened: [pre-processed data (1.37GB)](https://www.dropbox.com/s/l5wuml42r7opo4h/cifar10_whitened.t7?dl=0)
* SVHN (excluding the extra dataset): [pre-processed data (2.27GB)](https://www.dropbox.com/s/jibp9hiv5gj47v3/svhn_preprocessed.t7?dl=0)

## Example scripts

* [`run_CMCL.sh`](run_CMCL.sh): train the models using "Confident multiple choice learning".
* [`run_MCL.sh`](run_MCL.sh): train the models using "Multiple choice learning".
* [`run_IE.sh`](run_IE.sh): train the models using "Independent ensemble".

## All training options:

    python src/ensemble.py \
    --dataset=cifar \
    --model_type=resnet \
    --batch_size=128 \
    --num_model=5 \
    --loss_type=cmcl_v0 \
    --k=4 \
    --beta=0.75 \
    --feature_sharing=True \
    --test=False

* `dataset`         : supports `cifar` and `svhn`.
* `model_type`      : supports `vggnet`, `googlenet`, and `resnet`.
* `batch_size`      : we use batch size 128.
* `num_model`       : number of models to ensemble.
* `loss_type`       : supports `independent`, `mcl`, `cmcl_v0`, and `cmcl_v1`.
* `k`               : overlap parameter.
* `beta`            : penalty parameter.
* `feature_sharing` : use feature sharing if `True`.
* `test`            : if `True`, test the result of previous training, otherwise run a new training.
