import os
from datetime import datetime
import lmdb
import numpy as np
from collections import OrderedDict

from keras.layers import SimpleRNNCell, GRUCell, LSTMCell
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1_l2

from tbad.losses import modified_binary_crossentropy_2, modified_mean_absolute_error, modified_mean_squared_error_2
from tbad.losses import modified_mean_squared_error_3, modified_balanced_mean_absolute_error
from utils.score_scaling import ScoreNormalization
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, MinMaxScaler


class LMDBdata:
    def __init__(self, lmdb_filename, total_batches):
        self.lmdb_env = lmdb.open(lmdb_filename, map_size=int(500e9))
        self.idx = 0
        self.total_batches = total_batches

    def write(self, vars_write):
        with self.lmdb_env.begin(write=True) as lmdb_txn:
            for name, a in vars_write.items():
                if 'shape' in name:
                    lmdb_txn.put((name + '%d' % self.idx).encode(), np.array(a,dtype=np.int))
                else:
                    lmdb_txn.put((name + '%d' % self.idx).encode(), a.astype(np.float32))

        self.idx += 1

    def read(self, indx):

        X_train = []
        y_train = []
        X_train_shape = []
        y_train_shape = []
        xval_data = []
        yval_data = []
        xval_data_shape = []
        yval_data_shape = []

        with self.lmdb_env.begin() as lmdb_txn:
            with lmdb_txn.cursor() as lmdb_cursor:
                for bkey, val in lmdb_cursor:
                    key = bkey.decode()
                    if 'X_global_train_shape'+'%d' % indx == key or 'X_local_train_shape'+'%d' % \
                      indx == key:
                        X_train_shape.append(np.fromstring(val,dtype=np.int))
                    elif 'y_global_train_shape'+'%d' % indx == key or 'y_local_train_shape'+'%d' % \
                      indx == key:
                        y_train_shape.append(np.fromstring(val, dtype=np.int))
                    elif 'X_global_val_shape'+'%d' % indx == key or 'X_local_val_shape'+'%d' % \
                      indx == key:
                        xval_data_shape.append(np.fromstring(val, dtype=np.int))
                    elif 'y_global_val_shape' + '%d' % indx == key or 'y_local_val_shape' + '%d' % \
                      indx == key:
                        yval_data_shape.append(np.fromstring(val, dtype=np.int))

                    elif 'X_global_train'+'%d' % indx == key or 'X_local_train'+'%d' % indx == key:
                        X_train.append(np.fromstring(val, dtype=np.float32))
                    elif 'y_global_train'+'%d' % indx == key or 'y_local_train'+'%d' % indx == key:
                        y_train.append(np.fromstring(val, dtype=np.float32))
                    elif 'X_global_val'+'%d' % indx == key or 'X_local_val'+'%d' % indx == key:
                        xval_data.append(np.fromstring(val, dtype=np.float32))
                    elif 'y_global_val'+'%d' % indx == key or 'y_local_val'+'%d' % indx == key:
                        yval_data.append(np.fromstring(val, dtype=np.float32))
            # val_data = [xval_data, yval_data]
        X_train = [x.reshape(x_s) for x, x_s in zip(X_train, X_train_shape)]
        y_train = [x.reshape(x_s) for x, x_s in zip(y_train, y_train_shape)]
        xval_data = [x.reshape(x_s) for x, x_s in zip(xval_data, xval_data_shape)]
        yval_data = [x.reshape(x_s) for x, x_s in zip(yval_data, yval_data_shape)]
        val_data = [xval_data, yval_data]
        return X_train, y_train, val_data



def select_optimiser(optimiser, learning_rate):
    """Select an optimiser to train the RNN."""
    if optimiser == 'rmsprop':
        return RMSprop(lr=learning_rate)
    elif optimiser == 'adam':
        return Adam(lr=learning_rate)
    else:
        raise ValueError('Unknown optimiser. Please select either rmsprop or adam.')


def select_loss(loss_name):
    """Select a loss function for the model."""
    if loss_name == 'log_loss':
        return modified_binary_crossentropy_2
    elif loss_name == 'mae':
        return modified_mean_absolute_error
    elif loss_name == 'mse':
        return modified_mean_squared_error_2
    elif loss_name == 'balanced_mse':
        return modified_mean_squared_error_3
    elif loss_name == 'balanced_mae':
        return modified_balanced_mean_absolute_error
    else:
        raise ValueError('Unknown loss function. Please select one of: log_loss, mae or mse.')


def select_cell(cell_type, hidden_dim, l1=0.0, l2=0.0):
    """Select an RNN cell and initialises it with hidden_dim units."""
    if cell_type == 'vanilla':
        return SimpleRNNCell(units=hidden_dim, kernel_regularizer=l1_l2(l1=l1, l2=l2),
                             recurrent_regularizer=l1_l2(l1=l1, l2=l2))
    elif cell_type == 'gru':
        return GRUCell(units=hidden_dim, kernel_regularizer=l1_l2(l1=l1, l2=l2),
                       recurrent_regularizer=l1_l2(l1=l1, l2=l2))
    elif cell_type == 'lstm':
        return LSTMCell(units=hidden_dim, kernel_regularizer=l1_l2(l1=l1, l2=l2),
                        recurrent_regularizer=l1_l2(l1=l1, l2=l2))
    else:
        raise ValueError('Unknown cell type. Please select one of: vanilla, gru, or lstm.')


def set_up_logging(camera_id, root_log_dir=None, resume_training=None, message_passing=False):
    if message_passing:
        mp = 'mp'
    else:
        mp = ''
    log_dir = None
    if resume_training is not None:
        log_dir = os.path.dirname(resume_training)
    elif root_log_dir is not None:
        time_now = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(root_log_dir, camera_id + '_' + time_now + '_mp' + '_Grobust'
                               + '_Lrobust' + '_Orobust' )
        os.makedirs(log_dir)

    return log_dir


def resume_training_from_last_epoch(model, resume_training=None):
    last_epoch = 0
    if resume_training is not None:
        model.load_weights(resume_training)
        last_epoch = int(os.path.basename(resume_training).split('_')[1])

    return last_epoch


def select_scaler_model(scaler_name):
    if scaler_name == 'standard':
        return StandardScaler()
    elif scaler_name == 'robust':
        return RobustScaler(quantile_range=(0.00, 50.0))
    elif scaler_name == 'quantile':
        return QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=42)
    elif scaler_name == 'max_abs':
        return MaxAbsScaler()
    elif scaler_name == 'min_max':
        return MinMaxScaler()
    elif scaler_name == 'kde':
        return ScoreNormalization(method='KDE')
    elif scaler_name == 'gamma':
        return ScoreNormalization(method='gamma')
    elif scaler_name == 'chi2':
        return ScoreNormalization(method='chi2')
    else:
        raise ValueError('Unknown scaler. Please select one of: standard, robust, quantile, max_abs, min_max, '
                         'kde, gamma, chi2.')
    
    return None
