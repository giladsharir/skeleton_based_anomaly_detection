import os
import argparse
from utils.meaningful_splits import splits


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_folder', default='/mnt/datasets/Skeleton_Anomaly_Detection_data/')
parser.add_argument(
  '--split', default='')
parser.add_argument('--all_splits', action='store_true')
parser.add_argument(
  '--part', default='train', choices=['train', 'val'])
arg = parser.parse_args()

def run_train(split):
  t_path = os.path.join(arg.data_folder, split, 'train', 'trajectories')
  log_path = os.path.join('checkpoints', split)

  if not os.path.isdir(log_path):
    os.mkdir(log_path)

  cmd_train = 'python train.py --gpu_ids 0 --gpu_memory 0.1 combined_model ' + \
            t_path + ' --video_resolution 856x480 --message_passing ' \
                     '--multiple_outputs ' \
                     '--multiple_outputs_before_concatenation --input_length 12 ' \
                     '--rec_length 12 --pred_length 6 --reconstruct_reverse --cell gru ' \
                     '--global_hidden 8 --local_hidden 16 --output_activation linear ' \
                     '--optimiser adam --learning_rate 0.001 --loss mse --epochs 1 ' \
                     '--batch_size 256 --global_normalisation robust --local_normalisation robust ' \
                     '--out_normalisation robust --root_log_dir ' + log_path

  os.system(cmd_train)

def run_test(split):
  #  checkpoints/music/trajectories_2019_10_17_14_37_35_mp_Grobust_Lrobust_Orobust_down/
  #/mnt/datasets/Skeleton_Anomaly_Detection_data/music/val/trajectories
  #/mnt/datasets/Skeleton_Anomaly_Detection_data/music/val/frame_level_masks
  s_path = os.path.join('checkpoints', split)
  all_subdirs = [os.path.join(s_path,d) for d in os.listdir(s_path) if os.path.isdir(os.path.join(
    s_path,d))]
  latest_dir = max(all_subdirs, key=os.path.getmtime)

  model_path = latest_dir
  log_path = os.path.join(latest_dir, 'log_' + split +'.txt')
  tr_path = os.path.join(arg.data_folder, split, 'val', 'trajectories')
  anno_path = os.path.join(arg.data_folder, split, 'val', 'frame_level_masks')

  cmd_test = 'python evaluate.py --gpu_ids 0 --gpu_memory 0.1 combined_model ' + \
             model_path + '/ ' + tr_path + ' ' + anno_path + ' --video_resolution 856x480 ' \
             '--log_roc_pr ' + log_path
  print(cmd_test)
  os.system(cmd_test)

def main():

  if not arg.all_splits:
    split = arg.split
    if arg.part == 'train':
      run_train(split)
    else:
      run_test(split)
  else:
    for split in splits.keys():
      if arg.part == 'train':
        run_train(split)
      else:
        run_test(split)


if __name__ == '__main__':
    main()