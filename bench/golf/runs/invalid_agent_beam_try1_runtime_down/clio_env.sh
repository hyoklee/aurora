B=$HOME/clio-core.dataverse/build/bin
export LD_LIBRARY_PATH=$B:/opt/aurora/26.181.0/oneapi/compiler/latest/lib:/home/hyoklee/miniconda3/envs/iowarp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$B:$PATH
export PYTHONPATH=$B
export CLIO_SERVER_CONF=/home/hyoklee/aurora/bench/golf/clio_golf.yaml
export CAE_DATAVERSE_CACHE=/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/golf/dvcache
