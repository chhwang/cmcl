# Confident Multiple Choice Learning



## Preliminaries

It is tested under Ubuntu Linux 16.04.1 and Python 2.7 environment, and requries following Python packages to be installed:

* [TensorFlow](https://github.com/tensorflow/tensorflow): version 1.0.0 or above. Only GPU version is available.
* [Torchfile](https://github.com/bshillingford/python-torchfile): version 0.0.2 or above.

*Simple torchfile installation:*

    pip install torchfile


## How to run

The following is an example command also written in [`run_example.sh`](run_example.sh).

    python src/ensemble.py \
    --model_type=cnn \ 
    --loss_type=cmcl_v1 \
    --feature_sharing=True \
    --batch_size=64 \
    --num_model=5 \
    --k=4 \
    --beta=0.75 \
    --test=False

* `model_type`      : supports `cnn`, `vggnet`, `googlenet`, and `resnet`.
* `loss_type`       : supports `independent`, `mcl`, `cmcl_v0`, and `cmcl_v1`.
* `feature_sharing` : use feature sharing if `True`.
* `batch_size`      : use batch size 64 for `cnn`, otherwise use 128.
* `num_model`       : number of models to ensemble.
* `k`               : overlap parameter.
* `beta`            : penalty parameter.
* `test`            : run training if `False`, otherwise run test only without training.
