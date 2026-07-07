#!/usr/bin/env bash

set -e

# idempotent from here until # Fetch MODELS below
SCRIPT_PATH="${BASH_SOURCE:-$0}"
ABS_SCRIPT_PATH="$(realpath "${SCRIPT_PATH}")"
SCRIPT_DIR="$(dirname "${ABS_SCRIPT_PATH}")"

echo ${SCRIPT_DIR}/common.sh ${SCRIPT_DIR}/$1
source ${SCRIPT_DIR}/common.sh ${SCRIPT_DIR}/$1

if [[ "${DRY_RUN}" -eq "" ]]; then
    DRY_RUN=0
fi
echo DRY_RUN=$DRY_RUN

WORKDIR=$(mktemp -d)

cleanup() {
    echo "Cleaning up..."
    rm -r $WORKDIR
}

trap cleanup EXIT
trap 'exit 130' INT     # Ctrl-C
trap 'exit 143' TERM    # kill (SIGTERM)

if [ ! $DRY_RUN -eq 1 ]; then
    mkdir -p $APP
    mkdir -p $MODEL_SILERO
    mkdir -p $MODEL_SPEAKER
    mkdir -p $MODEL_VOICE
    mkdir -p $MODEL_VITS
    mkdir -p $MODEL_DENOISE
else
    echo mkdir -p $APP
    echo mkdir -p $MODEL_SILERO
    echo mkdir -p $MODEL_SPEAKER
    echo mkdir -p $MODEL_VOICE
    echo mkdir -p $MODEL_VITS
    echo mkdir -p $MODEL_DENOISE
fi

if [[ ! -f `which uv` ]]; then
    if [[ ! $DRY_RUN -eq 1 ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        echo Dry run: install uv
    fi
fi

if [[ ! $DRY_RUN -eq 1 ]]; then
    cd $APP
fi

if [[ ! -d $APP/.venv ]]; then
    if [[ ! $DRY_RUN -eq 1 ]]; then
        uv venv -p 3.13
    else
        echo Dry run: Create venv
    fi
fi

if $( type -t deactivate ); then
    if [[ ! $DRY_RUN -eq 1 ]]; then
        deactivate
    else
        echo Dry run: deactivate current venv
    fi
fi

if [[ ! $DRY_RUN -eq 1 ]]; then
    . $APP/.venv/bin/activate
else
    echo Dry run: activate venv
fi


if [[ ! $DRY_RUN -eq 1 ]]; then
    uv pip install sherpa-onnx sounddevice soundfile numpy
else
    echo Dry run: uv pip install
fi

##############
# Fetch Models
##############

# 1. Fetch Silero VAD models
if [[ ! $DRY_RUN -eq 1 ]]; then
    wget -P ${MODEL_SILERO} \
        https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
else
    echo Dry run: Download MODEL_SILERO
fi

# 2. Fetch speaker embedding model: WeSpeaker English VoxCeleb ResNet-34
if [[ ! $DRY_RUN -eq 1 ]]; then
    wget -P ${MODEL_SPEAKER} \
         https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx
else
    echo Dry run: Download MODEL_SPEAKER
fi

# 3. ASR model: SenseVoice int8 (zh/en/ja/ko/yue)
if [[ ! $DRY_RUN -eq 1 ]]; then
    BASE="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
    wget -P $WORKDIR \
        https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${BASE}.tar.bz2
    tar xf ${WORKDIR}/${BASE}.tar.bz2 -C ${WORKDIR}
    mv ${WORKDIR}/${BASE}/*.onnx ${WORKDIR}/${BASE}/*.txt ${MODEL_VOICE}/
    rm -r ${WORKDIR}/${BASE}*
else
    echo Dry run: Download MODEL_VOICE
fi

# 4. TTS model: VITS-piper Amy low (English)
if [[ ! $DRY_RUN -eq 1 ]]; then
    BASE="vits-piper-en_US-amy-low"
    wget -P ${WORKDIR} https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${BASE}.tar.bz2
    tar xf ${WORKDIR}/${BASE}.tar.bz2 -C ${WORKDIR}
    mv ${WORKDIR}/${BASE}/*.onnx ${WORKDIR}/${BASE}/tokens.txt ${WORKDIR}/${BASE}/espeak-ng-data ${MODEL_VITS}
    rm -r ${WORKDIR}/${BASE}*
else
    echo Dry run: Download MODEL_VITS
fi

#5. GTCRN speech denoiser
# Only needed if you pass --gtcrn-model
if [[ ! $DRY_RUN -eq 1 ]]; then
    wget -P ${MODEL_DENOISE} \
        https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
else
    echo Dry run: Download MODEL_DENOISE
fi

############
# 6. Cleanup
############

unset MODELS
unset MODEL_SILERO
unset MODEL_SPEAKER
unset MODEL_VOICE
unset MODEL_VITS
unset MODEL_DENOISE
