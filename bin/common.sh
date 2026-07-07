#!/usr/bin/env bash

set -e

echo COMMON $1

source $1

export MODELS="${APP}/var/models"
export MODEL_SILERO="${MODELS}/silero"
export MODEL_SPEAKER="${MODELS}/speaker"
export MODEL_VOICE="${MODELS}/voice-sense"
export MODEL_VITS="${MODELS}/vits"
export MODEL_DENOISE="${MODELS}/denoise"
