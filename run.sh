#!/bin/sh
export ARESPATH=$(pwd)/ares
export PYTHONPATH="${PYTHONPATH}:${ARESPATH}"

TRAINED_MODELS_DIR="${ARESPATH}/experiments/train/trained_models"
PRETRAINED_WEIGHTS="${ARESPATH}/quantized_weights"
CACHE_PATH="cache"
RESULT_PATH="results"
LOG_MNIST_PATH="logs/mnist_fc"
CONF_PATH="conf"

THEANO_FLAGS='device=gpu'
SEED=0
FRATE=0.05

# Build PATHS
mkdir -p $CACHE_PATH
mkdir -p $RESULT_PATH
mkdir -p $PRETRAINED_WEIGHTS
mkdir -p $TRAINED_MODELS_DIR
mkdir -p $LOG_MNIST_PATH

# Training Model
python "${ARESPATH}/experiments/train/train.py" \
    -c "${CONF_PATH}" \
    -m mnist_fc \
    -eps 10 \
    -v \
    --cache "${CACHE_PATH}" \
    --results "${RESULT_PATH}" \
    -to -sw_name "${TRAINED_MODELS_DIR}/mnist_fc"

# Quantize Model
python ${ARESPATH}/experiments/quantize/quantize_net.py \
    -m mnist_fc \
    -lw \
    -ld_name "${TRAINED_MODELS_DIR}/mnist_fc" \
    -qi 2 \
    -qf 6 \
    --cache "${CACHE_PATH}" \
    --conf "${CONF_PATH}" \
    --results "${RESULT_PATH}" \
    -sw "${PRETRAINED_WEIGHTS}/mnist_fc"

# Testing pre-trained weights
python "${ARESPATH}/experiments/eval/eval.py" \
    -m mnist_fc  \
    -v \
    --results "${RESULT_PATH}" \
    -c "${CONF_PATH}" \
    -lw \
    -ld_name "${PRETRAINED_WEIGHTS}/mnist_fc_quantized_2_6"

# Fault-Injection
fname="mnist_fc_quantized_2_6_${FRATE}_${SEED}"
python "${ARESPATH}/experiments/bits/bits.py" \
    -c "${CONF_PATH}" \
    -m mnist_fc \
    --results "${RESULT_PATH}" \
    -lw \
    -qi 2 \
    -qf 6 \
    -ld_name "${PRETRAINED_WEIGHTS}/mnist_fc_quantized_2_6" \
    -frate "${FRATE}" \
    -seed "${SEED}" | tee -a "${LOG_MNIST_PATH}/$fnamemkdir" -p results
