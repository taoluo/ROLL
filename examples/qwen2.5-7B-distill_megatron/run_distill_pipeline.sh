
#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_distill_pipeline.py --config_path $CONFIG_PATH  --config_name distill_megatron
