from copy import deepcopy
import os

import numpy as np
from sklearn.externals import joblib
from sklearn.utils import shuffle
import lmdb
from collections import OrderedDict

from tbad.autoencoder.data import load_trajectories, split_into_train_and_test, extract_global_features
from tbad.autoencoder.data import change_coordinate_system, scale_trajectories, aggregate_autoencoder_data
from tbad.autoencoder.data import input_trajectories_missing_steps
from tbad.rnn_autoencoder.data import remove_short_trajectories, aggregate_rnn_autoencoder_data
from tbad.combined_model.fusion import CombinedEncoderDecoder
from tbad.combined_model.message_passing import MessagePassingEncoderDecoder
from tbad.utils import set_up_logging, resume_training_from_last_epoch
from tbad.utils import LMDBdata

# class LMDBdata:
#     def __init__(self, lmdb_filename):
#         self.lmdb_env = lmdb.open(lmdb_filename, map_size=int(1e9))
#         self.idx = 0
#     def write(self, vars_write):
#         with self.lmdb_env.begin(write=True) as lmdb_txn:
#             for name, a in vars_write.items():
#                 lmdb_txn.put(name + '%d' % self.idx, a.astype(np.float32))
#
#         self.idx += 1
#     def read(self):
#
#         X_train = []
#         y_train = []
#         val_data = []
#         with self.lmdb_env.begin() as lmdb_txn:
#             with lmdb_txn.cusor() as lmdb_cursor:
#                 for key, val in lmdb_cursor:
#                     if 'X_global_train' in key or 'X_local_train' in key:
#                         X_train.append(np.fromstring(val, dtype=np.float32))
#                     elif 'y_global_train' in key or 'y_local_train' in key:
#                         y_train.append(np.fromstring(val, dtype=np.float32))
#                     elif 'val' in key:
#                         val_data.append(np.fromstring(val, dtype=np.float32))
#
#         return X_train, y_train, val_data


def train_combined_model(args):
    # General
    trajectories_path = args.trajectories
    camera_id = os.path.basename(trajectories_path)
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    # Architecture
    message_passing = args.message_passing
    reconstruct_original_data = args.reconstruct_original_data
    multiple_outputs = args.multiple_outputs
    multiple_outputs_before_concatenation = args.multiple_outputs_before_concatenation
    input_length = args.input_length
    rec_length = args.rec_length
    pred_length = args.pred_length
    global_hidden_dims = args.global_hidden_dims
    local_hidden_dims = args.local_hidden_dims
    extra_hidden_dims = args.extra_hidden_dims
    output_activation = args.output_activation
    cell_type = args.cell_type
    reconstruct_reverse = args.reconstruct_reverse
    # Training
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    loss = args.loss
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    epochs = args.epochs
    batch_size = args.batch_size
    input_missing_steps = args.input_missing_steps
    global_normalisation_strategy = args.global_normalisation_strategy
    local_normalisation_strategy = args.local_normalisation_strategy
    out_normalisation_strategy = args.out_normalisation_strategy
    # Logging
    root_log_dir = args.root_log_dir
    resume_training = args.resume_training


    #Todo: split the data into load_batches:
    total_batches = args.total_load_batches
    # total_batches = 100
    # X_global_train_tmp = []
    # X_local_train_tmp = []
    # y_global_train_tmp = []
    # y_local_train_tmp = []
    # X_global_val_tmp = []
    # X_local_val_tmp = []
    # y_global_val_tmp = []
    # y_local_val_tmp = []
    lmdb_filename = 'lmdb.db'
    DB = LMDBdata(lmdb_filename, total_batches)
    # lmdb_env = lmdb.open(lmdb_filename, map_size=int(1e9))

    for batch_indx in range(total_batches):
        if args.skip_lmdb and batch_indx > 0:
            continue

        print('\n>>>>>>> Loading %d batch out of %d \n' % (batch_indx, total_batches))
        trajectories = load_trajectories(trajectories_path, batch_indx, total_batches)
        print('\nLoaded %d trajectories.' % len(trajectories))

        trajectories = remove_short_trajectories(trajectories, input_length=input_length,
                                                 input_gap=0, pred_length=pred_length)
        print('\nRemoved short trajectories. Number of trajectories left: %d.' % len(trajectories))

        trajectories_train, trajectories_val = split_into_train_and_test(trajectories, train_ratio=0.8, seed=42)

        if input_missing_steps:
            trajectories_train = input_trajectories_missing_steps(trajectories_train)
            print('\nInputted missing steps of trajectories.')

        # TODO: General function to extract features
        # X_..._train, X_..._val, y_..._train, y_..._val, ..._scaler = general_function()

        # Global
        global_trajectories_train = extract_global_features(deepcopy(trajectories_train), video_resolution=video_resolution)
        global_trajectories_val = extract_global_features(deepcopy(trajectories_val), video_resolution=video_resolution)

        global_trajectories_train = change_coordinate_system(global_trajectories_train, video_resolution=video_resolution,
                                                             coordinate_system='global', invert=False)
        global_trajectories_val = change_coordinate_system(global_trajectories_val, video_resolution=video_resolution,
                                                           coordinate_system='global', invert=False)
        print('\nChanged global trajectories\'s coordinate system to global.')

        _, global_scaler = scale_trajectories(aggregate_autoencoder_data(global_trajectories_train),
                                              strategy=global_normalisation_strategy)

        X_global_train, y_global_train = aggregate_rnn_autoencoder_data(global_trajectories_train,
                                                                        input_length=input_length,
                                                                        input_gap=0, pred_length=pred_length)
        X_global_val, y_global_val = aggregate_rnn_autoencoder_data(global_trajectories_val, input_length=input_length,
                                                                    input_gap=0, pred_length=pred_length)

        X_global_train, _ = scale_trajectories(X_global_train, scaler=global_scaler, strategy=global_normalisation_strategy)
        X_global_val, _ = scale_trajectories(X_global_val, scaler=global_scaler, strategy=global_normalisation_strategy)
        if y_global_train is not None and y_global_val is not None:
            y_global_train, _ = scale_trajectories(y_global_train, scaler=global_scaler,
                                                   strategy=global_normalisation_strategy)
            y_global_val, _ = scale_trajectories(y_global_val, scaler=global_scaler, strategy=global_normalisation_strategy)
        print('\nNormalised global trajectories using the %s normalisation strategy.' % global_normalisation_strategy)

        # Local
        local_trajectories_train = deepcopy(trajectories_train) if reconstruct_original_data else trajectories_train
        local_trajectories_val = deepcopy(trajectories_val) if reconstruct_original_data else trajectories_val

        local_trajectories_train = change_coordinate_system(local_trajectories_train, video_resolution=video_resolution,
                                                            coordinate_system='bounding_box_centre', invert=False)
        local_trajectories_val = change_coordinate_system(local_trajectories_val, video_resolution=video_resolution,
                                                          coordinate_system='bounding_box_centre', invert=False)
        print('\nChanged local trajectories\'s coordinate system to bounding_box_centre.')

        _, local_scaler = scale_trajectories(aggregate_autoencoder_data(local_trajectories_train),
                                             strategy=local_normalisation_strategy)

        X_local_train, y_local_train = aggregate_rnn_autoencoder_data(local_trajectories_train, input_length=input_length,
                                                                      input_gap=0, pred_length=pred_length)
        X_local_val, y_local_val = aggregate_rnn_autoencoder_data(local_trajectories_val, input_length=input_length,
                                                                  input_gap=0, pred_length=pred_length)

        del local_trajectories_train
        del local_trajectories_val

        X_local_train, _ = scale_trajectories(X_local_train, scaler=local_scaler, strategy=local_normalisation_strategy)
        X_local_val, _ = scale_trajectories(X_local_val, scaler=local_scaler, strategy=local_normalisation_strategy)
        if y_local_train is not None and y_local_val is not None:
            y_local_train, _ = scale_trajectories(y_local_train, scaler=local_scaler, strategy=local_normalisation_strategy)
            y_local_val, _ = scale_trajectories(y_local_val, scaler=local_scaler, strategy=local_normalisation_strategy)
        print('\nNormalised local trajectories using the %s normalisation strategy.' % local_normalisation_strategy)

        # (Optional) Reconstruct the original data
        if reconstruct_original_data:
            print('\nReconstruction/Prediction target is the original data.')
            out_trajectories_train = trajectories_train
            out_trajectories_val = trajectories_val

            out_trajectories_train = change_coordinate_system(out_trajectories_train, video_resolution=video_resolution,
                                                              coordinate_system='global', invert=False)
            out_trajectories_val = change_coordinate_system(out_trajectories_val, video_resolution=video_resolution,
                                                            coordinate_system='global', invert=False)
            print('\nChanged target trajectories\'s coordinate system to global.')

            _, out_scaler = scale_trajectories(aggregate_autoencoder_data(out_trajectories_train),
                                               strategy=out_normalisation_strategy)

            X_out_train, y_out_train = aggregate_rnn_autoencoder_data(out_trajectories_train, input_length=input_length,
                                                                      input_gap=0, pred_length=pred_length)
            X_out_val, y_out_val = aggregate_rnn_autoencoder_data(out_trajectories_val, input_length=input_length,
                                                                  input_gap=0, pred_length=pred_length)

            X_out_train, _ = scale_trajectories(X_out_train, scaler=out_scaler, strategy=out_normalisation_strategy)
            X_out_val, _ = scale_trajectories(X_out_val, scaler=out_scaler, strategy=out_normalisation_strategy)
            if y_out_train is not None and y_out_val is not None:
                y_out_train, _ = scale_trajectories(y_out_train, scaler=out_scaler, strategy=out_normalisation_strategy)
                y_out_val, _ = scale_trajectories(y_out_val, scaler=out_scaler, strategy=out_normalisation_strategy)
            print('\nNormalised target trajectories using the %s normalisation strategy.' % out_normalisation_strategy)

        # Shuffle training data and assemble training and validation sets
        if y_global_train is not None:
            if reconstruct_original_data:
                X_global_train, X_local_train, X_out_train, y_global_train, y_local_train, y_out_train = \
                    shuffle(X_global_train, X_local_train, X_out_train,
                            y_global_train, y_local_train, y_out_train, random_state=42)
                # X_train = [X_global_train, X_local_train, X_out_train]
                # y_train = [y_global_train, y_local_train, y_out_train]
                # val_data = ([X_global_val, X_local_val, X_out_val], [y_global_val, y_local_val, y_out_val])
            else:
                X_global_train, X_local_train, y_global_train, y_local_train = \
                    shuffle(X_global_train, X_local_train, y_global_train, y_local_train, random_state=42)
                # X_train = [X_global_train, X_local_train]
                # y_train = [y_global_train, y_local_train]
                # val_data = ([X_global_val, X_local_val], [y_global_val, y_local_val])
        else:
            if reconstruct_original_data:
                X_global_train, X_local_train, X_out_train = \
                    shuffle(X_global_train, X_local_train, X_out_train, random_state=42)
                # X_train = [X_global_train, X_local_train, X_out_train]
                # y_train = None
                # val_data = ([X_global_val, X_local_val, X_out_val],)
            else:
                X_global_train, X_local_train = shuffle(X_global_train, X_local_train, random_state=42)
                # X_train = [X_global_train, X_local_train]
                # y_train = None
                # val_data = ([X_global_val, X_local_val],)
        if args.skip_lmdb:
          continue

        data_dict = {'X_global_train_shape': X_global_train.shape, 'X_local_train_shape': X_local_train.shape,
                     'y_global_train_shape': y_global_train.shape, 'y_local_train_shape': y_local_train.shape,
                     'X_global_val_shape': X_global_val.shape, 'X_local_val_shape': X_local_val.shape,
                     'y_global_val_shape': y_global_val.shape,
                     'y_local_val_shape': y_local_val.shape,
                     'X_global_train': X_global_train, 'X_local_train': X_local_train,
                     'y_global_train': y_global_train, 'y_local_train': y_local_train,
                     'X_global_val': X_global_val, 'X_local_val': X_local_val,
                     'y_global_val': y_global_val,
                     'y_local_val': y_local_val
                     }

        DB.write(OrderedDict(data_dict.items()))
        # with lmdb_env.begin(write=True) as lmdb_txn:
        #   lmdb_txn.put('X_g_t_%d'%batch_indx, X_global_train)
        #   lmdb_txn.put('X_l_t'%batch_indx, X_local_train)
        #   lmdb_txn.put('y_g_t'%batch_indx, y_global_train)
        #   lmdb_txn.put('y_l_t'%batch_indx, y_local_train)
        #   lmdb_txn.put('X_g_v'%batch_indx, X_global_val)
        #   lmdb_txn.put('X_l_v'%batch_indx, X_local_val)
        #   lmdb_txn.put('y_g_v', y_global_val)
        #   lmdb_txn.put('y_l_v', y_local_val)


        # X_global_train_tmp.append(np.copy(X_global_train))
        # X_local_train_tmp.append(np.copy(X_local_train))
        # y_global_train_tmp.append(y_global_train.copy())
        # y_local_train_tmp.append(y_local_train.copy())
        # X_global_val_tmp.append(X_global_val.copy())
        # X_local_val_tmp.append(X_local_val.copy())
        # y_global_val_tmp.append(y_global_val.copy())
        # y_local_val_tmp.append(y_local_val.copy())


        # X_train_tmp.append(np.copy(X_train))
        # y_train_tmp.append(np.copy(y_train))
        # val_data_tmp.append(np.copy(val_data))

    #collecting the data batches
    # X_global_train = np.vstack(X_global_train_tmp)
    # X_local_train = np.vstack(X_local_train_tmp)
    #
    # y_global_train = np.vstack(y_global_train_tmp)
    # y_local_train = np.vstack(y_local_train_tmp)
    # X_global_val = np.vstack(X_global_val_tmp)
    # X_local_val = np.vstack(X_local_val_tmp)
    # y_global_val = np.vstack(y_global_val_tmp)
    # y_local_val = np.vstack(y_local_val_tmp)

    # X_train = [X_global_train, X_local_train]
    # y_train = [y_global_train, y_local_train]
    # val_data = ([X_global_val, X_local_val], [y_global_val, y_local_val])

    # X_train = np.vstack(X_train_tmp)
    # y_train = np.vstack(y_train_tmp)
    # val_data = np.vstack(val_data_tmp)

    # Model
    print('\nInstantiating combined anomaly model ...')
    global_input_dim = X_global_train.shape[-1]
    local_input_dim = X_local_train.shape[-1]
    model_args = {'input_length': input_length, 'global_input_dim': global_input_dim,
                  'local_input_dim': local_input_dim, 'reconstruction_length': rec_length,
                  'prediction_length': pred_length, 'global_hidden_dims': global_hidden_dims,
                  'local_hidden_dims': local_hidden_dims, 'extra_hidden_dims': extra_hidden_dims,
                  'output_activation': output_activation, 'cell_type': cell_type,
                  'reconstruct_reverse': reconstruct_reverse, 'reconstruct_original_data': reconstruct_original_data,
                  'multiple_outputs': multiple_outputs,
                  'multiple_outputs_before_concatenation': multiple_outputs_before_concatenation,
                  'optimiser': optimiser, 'learning_rate': learning_rate, 'loss': loss,
                  'l1_reg': l1_reg, 'l2_reg': l2_reg}
    if message_passing:
        combined_rnn_ae = MessagePassingEncoderDecoder(**model_args)
    else:
        combined_rnn_ae = CombinedEncoderDecoder(**model_args)

    log_dir = set_up_logging(camera_id=camera_id, root_log_dir=root_log_dir,
                             resume_training=resume_training, message_passing=message_passing)
    last_epoch = resume_training_from_last_epoch(model=combined_rnn_ae, resume_training=resume_training)

    combined_rnn_ae.train(DB, epochs=epochs, initial_epoch=last_epoch,
                          batch_size=batch_size, log_dir=log_dir)
    # combined_rnn_ae.train(X_train, y_train, epochs=epochs, initial_epoch=last_epoch, batch_size=batch_size,
    #                       val_data=val_data, log_dir=log_dir)
    print('\nCombined anomaly model successfully trained.')

    if log_dir is not None:
        global_scaler_file_name = os.path.join(log_dir, 'global_scaler.pkl')
        local_scaler_file_name = os.path.join(log_dir, 'local_scaler.pkl')
        joblib.dump(global_scaler, filename=global_scaler_file_name)
        joblib.dump(local_scaler, filename=local_scaler_file_name)
        if reconstruct_original_data:
            out_scaler_file_name = os.path.join(log_dir, 'out_scaler.pkl')
            joblib.dump(out_scaler, filename=out_scaler_file_name)
        print('log files were written to: %s' % log_dir)

    if reconstruct_original_data:
        return combined_rnn_ae, global_scaler, local_scaler, out_scaler

    return combined_rnn_ae, global_scaler, local_scaler
