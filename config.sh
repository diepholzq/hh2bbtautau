# storage directories, all directories are created for your
export USER=diepholq
export STORE_DIR=/data/dust/user/${USER}/HH_DNN # ROOT of storage
export CACHE_DIR=${STORE_DIR}/cache             # directory where preprocessed data is stored as well as dataset paths
export PICTURE_DIR=${STORE_DIR}/pictures/features
export MODELS_DIR=${STORE_DIR}/models           # saved models
export TENSORBOARD_DIR=${STORE_DIR}/tensorboard # tensorboard storage

# debug level: DEBUG; INFO; WARNING
export LOG_LEVEL="DEBUG"
export FILE_LOG_LEVEL="DEBUG"

# location where the input data can be found.
export ERA=prod24 # possible eras: prod14, prod20, prod24 (20 only has 22pre)
export TRAINIG_ROOT="/data/dust/user/riegerma/hh2bbtautau/run3_training_data/"
export INPUT_DATA_DIR="/data/dust/user/riegerma/hh2bbtautau/run3_training_data/${ERA}"
export SIGNAL_DATA_DIR="/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/dev_likelihood_ratio/"
export BG_DATA_DIR="/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/tt_dl_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/dev_likelihood_ratio/"

# virtualenv handling
export VENV_MODE="venv"  # venv_switch - possible values: pyenv, venv or cf
export ML_ENV="ml_torch" # name of your virtualenv, so it can be activated by source setup.sh

# virtualenv directories - you only need to set 1 (depending on your VENV_MODE)
export VENV_ROOT="/data/dust/user/${USER}/pyenv_virtualenvs" # place to look for existing virtualenvs
export CF_SANDBOX="venv_hbt_dev"                             #  sandbox name within columnflow - ATTENTION: need to run source of columnflow before
export PYENV_ROOT="/afs/desy.de/user/w/${USER}/.pyenv"       # root of pyenv installation

# flags to stop unnecessar dir checks, can be undone to recreate dirs
export SETUP_DIRS_DONE=0 # 1 = done
