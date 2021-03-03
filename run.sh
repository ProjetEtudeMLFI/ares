# Sample run.sh
export PYTHONPATH=$(pwd)/:

TRAINED_MODELS_DIR="/ares/experiments/train/trained_models"
PRETRAINED_WEIGHTS="/ares/quantized_weights"
THEANO_FLAGS='device=gpu'
SEED=0
FRATE=0.05

mkdir -p cache
mkdir -p results
mkdir -p $PRETRAINED_WEIGHTS
mkdir -p $TRAINED_MODELS_DIR
mkdir -p logs/mnist_fc

# Training Model
python /ares/experiments/train/train.py -c /ares/conf -m mnist_fc -eps 10 -v --results /ares/results -to -sw_name  $TRAINED_MODELS_DIR/mnist_fc

# Quantize Model
python /ares/experiments/quantize/quantize_net.py -m mnist_fc -lw -ld_name $TRAINED_MODELS_DIR/mnist_fc -qi 2 -qf 6 --cache /ares/cache/ --conf /ares/conf --results /ares/results -sw /ares/quantized_weights/mnist_fc

# Testing pre-trained weights
python /ares/experiments/eval/eval.py -m mnist_fc -v --results /ares/results -c /ares/conf -lw -ld_name $PRETRAINED_WEIGHTS/mnist_fc_quantized_2_6

# Fault-Injection
fname="mnist_quantized_2_8_($FRATE)_$SEED"
python /ares/experiments/bits/bits.py -c /ares/conf -m mnist_fc --results /ares/results -lw -qi 2 -qf 6 -ld_name $PRETRAINED_WEIGHTS/mnist_fc_quantized_2_6 -frate $FRATE -seed $SEED | tee -a logs/mnist_fc/$fname

# Training MNIST
#python3 /ares/run_models.py -m mnist_fc -eps 10 -v  -to -c /ares/conf -sw

# Training CiFar
#python /ares/run_models.py -m cifar10_vgg -eps 10 -v  -to -c /ares/conf

#Training SVHN
#python /ares/run_models.py -m imagenet_resnet50 -eps 10 -v  -to -c /ares/conf
