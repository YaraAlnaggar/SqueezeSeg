#!/bin/bash

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/npy_player.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help,"
  echo "-data_dir                 Where is the *.npy training set."
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/npy_player.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help,"
      echo "-data_dir                 Where is the *.npy training set."
      exit 0
      ;;
    -data_dir)
      export DATA_DIR="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

#bash ./scripts/killall.sh
bash ./scripts/quickstart.sh -rviz_cfg ./rviz/default.rviz

python ./src/ros/npy_node.py \
  --dataset_path=$DATA_DIR
