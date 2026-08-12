#!/bin/bash --login
# Uncomment specific block of interest and run
# $ bash script.sh
#
# Specify CUDA devices
# $ CUDA_VISIBLE_DEVICES=1 bash script.sh
#
# Specify log directory
# $ LOGDIR=/localscrtach/scdiff bash script.sh
#
# To view the generated script without executing, pass the TEST_FLAG envar as 1
# $ TEST_FLAG=1 bash script.sh

trap "echo ERROR && exit 1" ERR


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
LOGDIR=${LOGDIR:-logs}
CONFIG_PREFIX="configs/eval"
NAME=scHiC_v1.1
MASK_MODE="v2"

TEST_FLAG=${TEST_FLAG:-0}

# OFFLINE_SETTINGS="--wandb f"
OFFLINE_SETTINGS="--wandb_offline t"
# --------------------

HOMEDIR=$(dirname $(dirname $(realpath $0)))
cd $HOMEDIR

# Definition of launch() function
launch () {
    full_settings=($@)   # $@ 是一个包含了所有传递给脚本的命令行参数的数组。
    echo ":::DXY TEST:::" "Received parameters: ${full_settings}" # 来打印所有接收到的参数。  Received parameters: denoising 10

    task=${full_settings[0]}
    save_dir=${full_settings[1]}
    seed=${full_settings[2]}
    save_path="${HOMEDIR}/results/${save_dir}"

    # check if the save_path is exists
    if [ ! -d "$save_path" ]; then
        mkdir "$save_path"
        echo "Directory created: $save_path"
    else
        echo -e "Directory already exists: $save_path \n"
    fi


    # CIAN HERE
    if [[ $task == denoising ]]; then
        dataset_name=Tan
        data_prefix="1Mb_chr1"
        data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${dataset_name}_${data_prefix}.h5ad"
        data_settings+=" data.params.validation.params.dataset=${dataset_name} data.params.validation.params.fname=${dataset_name}_${data_prefix}.h5ad"
        data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${dataset_name}_${data_prefix}.h5ad"
    else
        echo Unknown task $task && exit 1
    fi

    # 后台运行
    # script="nohup python -u main.py -b ${CONFIG_PREFIX}_${task}_${MASK_MODE}.yaml --name ${NAME} --seed ${seed} --wandb False --save_path ${save_path}"
    script="python main.py -b ${CONFIG_PREFIX}_${task}_${MASK_MODE}.yaml --name ${NAME} --seed ${seed} --wandb False --save_path ${save_path}"

    script+=" --logdir ${LOGDIR} --postfix ${task}_r.seed${seed} ${OFFLINE_SETTINGS} ${data_settings}"
    echo ${script}

    [[ $TEST_FLAG == 0 ]] && eval ${script}
    # 如果 $TEST_FLAG 的值等于 0，则执行 eval ${script}。eval 命令会执行它接收到的参数作为shell命令。
}


task=$1  # 这行代码将脚本的第一个参数赋值给变量 task。
save_dir=$2
SEED=${SEED:-10}

launch ${task} ${save_dir} ${SEED}
# 启动launch函数




