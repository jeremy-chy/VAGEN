set -x


export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "data/alfred_vagen/train.parquet" \
    --test_path "data/alfred_vagen/test.parquet" \