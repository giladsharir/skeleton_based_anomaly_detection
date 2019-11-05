#0. split the dataset to normal/abnormal according to the chosen split
#1. convert the kinetics npy file to trajectory files for input
#2. convert the label.pkl file to npy label files

import os
import inspect
import sys
import argparse
import numpy as np
import pickle
import random
from tqdm import tqdm

from utils.meaningful_splits import splits

# from tbad.combined_model.data import write_reconstructed_trajectories

parser = argparse.ArgumentParser()
parser.add_argument(
        '--data_path', default='/mnt/datasets/Kinetics-skeleton/')
parser.add_argument(
    '--out_folder', default='/mnt/datasets/Skeleton_Anomaly_Detection_data/')
parser.add_argument(
  '--part', default='train', choices=['train', 'val'])
parser.add_argument(
  '--split', default='')
parser.add_argument('--all_splits', action='store_true')
parser.add_argument('--inverse', action='store_true')
parser.add_argument(
  '--debug', action='store_true')

arg = parser.parse_args()


def np_del_by_val_1d(arr, *del_arrs):
  ret = np.array(arr)
  for del_arr in del_arrs:
    ret = np.delete(ret, [i for i in range(ret.shape[0]) if ret[i] in del_arr])
  return ret


def get_exp_classes(split_name, m=250, ntu=False):
  # NTU only removes non-normal classes from the 60, Kinetics filters by sorted success threshold
  if isinstance(split_name, (list, tuple, np.ndarray, np.generic)):
    normal_classes = list(split_name)
  else:
    # if ntu:
    #   normal_classes = ntu_splits[split_name]
    # else:
    normal_classes = splits[split_name]
  # if ntu:
  #   abnormal_classes = np_del_by_val_1d(list(range(60)), normal_classes)
  #   return normal_classes, abnormal_classes
  currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  parentdir = os.path.dirname(currentdir)
  class_acc_path = os.path.join(parentdir, 'csv/class_accuracy_np.npy')
  class_acc_np = np.load(class_acc_path)
  class_num = class_acc_np.shape[0]
  unusable_classes = class_acc_np[m:, 0].astype(int)
  normal_classes = np_del_by_val_1d(normal_classes, unusable_classes)
  abnormal_classes = np_del_by_val_1d(np.arange(class_num), normal_classes, unusable_classes)
  return normal_classes, abnormal_classes



def denormalize_keypoints(kpts_n, image_size):
  kpts = kpts_n.copy()
  kpts[kpts_n != 0] = kpts[kpts_n != 0] + 0.5
  new_kpts = np.vstack((kpts[np.newaxis, 0] * image_size[0], kpts[np.newaxis, 1] * image_size[1]))
  return new_kpts

# def keypoints17_to_coco18(kps):
#   """
#   Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
#   New keypoint (neck) is the average of the shoulders, and points
#   are also reordered.
#   """
#   kp_np = np.array(kps)
#   neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
#   kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
#   opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
#   opp_order = np.array(opp_order, dtype=np.int)
#   kp_coco18 = kp_np[..., opp_order, :]
#   return kp_coco18

def convert_18_to_17(kpts):
  """
  convert 18 keypoint format given in kinetics npy files to 17 keypoints according to the
  function above
  """
  order = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
  order = np.array(order, dtype=np.int)
  kpts_17 = kpts[..., order, :]
  return kpts_17

class AugNpFeeder():
  """ Feeder that takes two numpy arrays as data instead of paths"""

  def __init__(self, data_np, label_np, specify_classes=None, transform_list=None,
               return_indices=False, debug=False,
               threshold=452.0, threshold_per_clip=False):
    # super().__init__()
    self.data = data_np
    self.N, self.C, self.T, self.V, self.M = self.data.shape
    label_np = label_np.astype(np.int32)
    self.label = list(label_np)

    self.debug = debug
    self.specify_classes = specify_classes
    self.return_indices = return_indices

    if transform_list is None or transform_list == []:
      self.apply_transforms = False
      self.num_transform = 1
    else:
      self.apply_transforms = True
      self.num_transform = len(transform_list)
    self.transform_list = transform_list
    self.num_samples = self.data.shape[0]

    self.load_data()

  def load_data(self):
    if self.debug:
      self.label = self.label[0:100]
      self.data = self.data[0:100]

    if self.specify_classes is not None:
      class_indices = [i for i, lbl in enumerate(self.label) if lbl in self.specify_classes]
      self.label = [self.label[i] for i in class_indices]
      self.data = self.data[class_indices]
      self.num_samples = self.data.shape[0]  # Reduced the number of samples
      self.N = self.num_samples

  def __len__(self):
    return self.num_transform * self.num_samples

  def __getitem__(self, index):
    # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
    # sample 7 is data sample 7%5=2 and transform is 7//5=1
    if self.apply_transforms:
      sample_index = index % self.num_samples
      trans_index = index // self.num_samples
      data_numpy = np.array(self.data[sample_index])
      label = trans_index
      data_transformed = self.transform_list[trans_index](data_numpy)
    else:
      data_transformed = np.array(self.data[index])
      label = self.label[index]

    # processing
    if self.return_indices:
      return data_transformed, label, index
    else:
      return data_transformed, label

def write_skeleton_trajectory(dataset, split, part, type, rate):

  video_writing_dir = os.path.join(arg.out_folder, split, part)
  lbl_writing_dir = os.path.join(video_writing_dir, 'frame_level_masks')
  tr_writing_dir = os.path.join(video_writing_dir, 'trajectories')

  if not os.path.isdir(lbl_writing_dir):
    os.makedirs(lbl_writing_dir)

  for video_id in range(len(dataset)):
    if random.random() > rate:
      continue

    d, _ = dataset[video_id]


    if not os.path.isdir(os.path.join(tr_writing_dir, str(video_id))):
      os.makedirs(os.path.join(tr_writing_dir, str(video_id)))

    # keypoints list per video
    kpts = d[:2]
    scores = d[2]

    # denormalize keypoint coordinates - scale by image size, and add center
    image_size = (856, 480)
    kpts = denormalize_keypoints(kpts, image_size)

    # convert 18 part skeleton to 17 part format
    kpts = convert_18_to_17(kpts)

    A = kpts[0]
    B = kpts[1]
    trajects = np.empty((A.shape[0], A.shape[1] + B.shape[1], A.shape[2]))
    trajects[:, ::2, :] = A
    trajects[:, 1::2, :] = B

    # trajects = np.reshape(kpts,(kpts.shape[0]*kpts.shape[1], kpts.shape[2], kpts.shape[3]))
    # trajects = trajects.transpose((1,0,2))
    if type == 'normal':
      lbl = np.zeros(trajects.shape[0])
    elif type == 'abnormal':
      lbl = np.ones(trajects.shape[0])
    else:
      print('unknown class')

    if part == 'val':
      label_writing_file = os.path.join(lbl_writing_dir, '00_' + str(video_id) + '.npy' )
      np.save(label_writing_file, lbl)

    for skeleton_id in range(2):
      frame_ids = np.array(range(trajects.shape[0]))
      trajectory = np.hstack((frame_ids[:, np.newaxis], trajects[..., skeleton_id]))
      skeleton_writing_file = os.path.join(tr_writing_dir, str(video_id), str(skeleton_id)) + '.csv'
      np.savetxt(skeleton_writing_file, trajectory, fmt='%.4f', delimiter=',') #encoding='utf-8'

  return tr_writing_dir, lbl_writing_dir



# def split_data_inv(data, label, part, split):
#   normal_classes, abnormal_classes = get_exp_classes(split)
#
#
#   # if arg.debug:
#   #   data = data[0:100]
#
#   if part == 'train':
#     dataset = AugNpFeeder(data, np.array(label), specify_classes=normal_classes, debug=arg.debug)
#
#     t_path, _ = write_skeleton_trajectory(dataset, split, part, 'normal', 1)
#     #Todo: cmd line - training with this dataset
#     # cmd_train = 'python train.py --gpu_ids 0 --gpu_memory 0.1 combined_model ' + \
#     #             t_path + '--video_resolution 856x480 --message_passing ' \
#     #                      '--reconstruct_original_data --multiple_outputs --multiple_outputs_before_concatenation --input_length 12 --rec_length 12 --pred_length 6 --reconstruct_reverse --cell gru --global_hidden 8 --local_hidden 16 --output_activation linear --optimiser adam --learning_rate 0.001 --loss mse --epochs 5 --batch_size 256 --global_normalisation robust --local_normalisation robust --out_normalisation robust'
#
#
#   elif part == 'val':
#     dataset = AugNpFeeder(data, np.array(label), specify_classes=normal_classes, debug=arg.debug)
#     dataset_abn = AugNpFeeder(data, np.array(label), specify_classes=abnormal_classes,
#                               debug=arg.debug)
#
#     t_path, _ = write_skeleton_trajectory(dataset, split, part, 'normal', 1)
#     _, l_path = write_skeleton_trajectory(dataset_abn, split, part, 'abnormal', 0.1)
#     #Todo: cmd line - evaluate the dataset (all_fame_level_anomally_masks  to set the gt)


def split_data(data, label, part, split):
  normal_classes, abnormal_classes = get_exp_classes(split)

  if arg.inverse:
    tmp = normal_classes.copy()
    normal_classes = abnormal_classes.copy()
    abnormal_classes = tmp

  # if arg.debug:
  #   data = data[0:100]

  if part == 'train':
    dataset = AugNpFeeder(data, np.array(label), specify_classes=normal_classes, debug=arg.debug)

    t_path, _ = write_skeleton_trajectory(dataset, split, part, 'normal', 1)
    #Todo: cmd line - training with this dataset
    # cmd_train = 'python train.py --gpu_ids 0 --gpu_memory 0.1 combined_model ' + \
    #             t_path + '--video_resolution 856x480 --message_passing ' \
    #                      '--reconstruct_original_data --multiple_outputs --multiple_outputs_before_concatenation --input_length 12 --rec_length 12 --pred_length 6 --reconstruct_reverse --cell gru --global_hidden 8 --local_hidden 16 --output_activation linear --optimiser adam --learning_rate 0.001 --loss mse --epochs 5 --batch_size 256 --global_normalisation robust --local_normalisation robust --out_normalisation robust'


  elif part == 'val':
    dataset = AugNpFeeder(data, np.array(label), specify_classes=normal_classes, debug=arg.debug)
    dataset_abn = AugNpFeeder(data, np.array(label), specify_classes=abnormal_classes,
                              debug=arg.debug)

    if arg.inverse:
      rate_n = 0.1
      rate_ab = 1
    else:
      rate_n = 1
      rate_ab = 0.1
    t_path, _ = write_skeleton_trajectory(dataset, split, part, 'normal', rate_n)
    _, l_path = write_skeleton_trajectory(dataset_abn, split, part, 'abnormal', rate_ab)
    #Todo: cmd line - evaluate the dataset (all_fame_level_anomally_masks  to set the gt)


def main():

  # read npy file
  part = arg.part
  # with np.load("{}/{}_data.npy".format(arg.data_path, part)) as data:
  #   data[]

  label_path = "{}/{}_label.pkl".format(arg.data_path, part)
  with open(label_path, 'rb') as f:
    _, label = pickle.load(f)
  data = np.load("{}/{}_data.npy".format(arg.data_path, part), mmap_mode='r')

  if not arg.all_splits:
    split = arg.split
    split_data(data, label, part, split)
  else:
    for split in tqdm(splits.keys()):
      split_data(data, label, part, split)


if __name__ == '__main__':
    main()