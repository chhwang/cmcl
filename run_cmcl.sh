###### Example script to run the code ######
#!/bin/sh

# dataset:    Supports cifar, svhn.
# model_type: Supports vggnet, googlenet, resnet.
# batch_size: We use batch size 128.
# num_model:  # of models to ensemble.
# loss_type:  Supports independent, mcl, cmcl_v0, cmcl_v1.
# k:          Overlap parameter.
# beta:       Penalty parameter.
# feature_sharing: Use feature sharing if True.
COMMAND="python src/ensemble.py \
--dataset=cifar \
--model_type=resnet \
--batch_size=128 \
--num_model=5 \
--loss_type=cmcl_v0 \
--k=4 \
--gpu=$1 \
--beta=$2 \
--feature_sharing=True"
COMMAND_TRAIN="$COMMAND --test=False"
COMMAND_TEST="$COMMAND --test=True"

MODEL="run-gpu-$1-beta-$2"

# run train
echo $COMMAND_TRAIN
$COMMAND_TRAIN > "log_$MODEL"

# run test
echo $COMMAND_TEST
$COMMAND_TEST >> "log_$MODEL"
