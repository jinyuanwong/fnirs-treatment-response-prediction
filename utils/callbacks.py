import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_accuracy', threshold=0.65, mode='max', save_best_only=True, save_weights_only=True, filepath='checkpoint'):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.filepath = filepath
        self.best = -np.Inf if mode == 'max' else np.Inf
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        train_accuracy = logs.get('accuracy')

        if self.mode == 'max':
            if current > self.best and train_accuracy > self.threshold:
                self.best = current
                if self.save_best_only:
                    print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, and train accuracy is {train_accuracy:.4f}. Saving model.')
                if self.save_weights_only:
                    self.model.save_weights(self.filepath)
                else:
                    self.model.save(self.filepath)
        else:
            if current < self.best and train_accuracy > self.threshold:
                self.best = current
                if self.save_best_only:
                    print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, and train accuracy is {train_accuracy:.4f}. Saving model.')
                if self.save_weights_only:
                    self.model.save_weights(self.filepath)
                else:
                    self.model.save(self.filepath)

# Example usage:

# Create an instance of the custom callback
checkpoint_filepath = 'best_model_checkpoint'
custom_checkpoint = CustomModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', threshold=0.6, mode='max', save_best_only=True, save_weights_only=False)

def reduceLRonplateau(monitor='val_loss', factor=0.01, patience=5, min_lr=1e-9):
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=min_lr)
    return reduce_lr