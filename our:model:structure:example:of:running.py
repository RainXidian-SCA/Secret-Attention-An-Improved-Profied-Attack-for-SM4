import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected1D, Reshape, Input,Multiply,Permute,RepeatVector,Lambda,LSTM,GRU
from tensorflow.keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, MaxPooling2D,Concatenate,Add,ZeroPadding1D
from tensorflow.keras.layers import Embedding, LSTM, GRU
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Softmax, MultiHeadAttention
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta
from tensorflow.keras import regularizers, constraints
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K 
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow import keras
from datetime import datetime
current_data = datetime.now()
date_string = current_data.strftime("%Y-%m-%d")
experiment_name = 'security/level/'+date_string+'specific/parameter'
path2traces = 'path/to/hdf5/file/of/SM4/algorithm/with/masking'
samples = 19996
num_traces = 20480
num_attack = 5120
batch_size = 64
learningrate = 0.001
epochs = 800
num_heads = 16
head_size = 124
key_dim = num_heads * head_size
layers = 3
class SaveModelCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if((epoch+1)%epochs_per_save==0):
            print('saving model of epoch {}'.format(epoch+1))
            temp_model = self.model
            temp_model.save('./models/'+experiment_name+'/Test_newest.hdf5')
            if((epoch+epoch_offset+1)%20==0):
                temp_model.save('./models/'+experiment_name+'/Test_epoch{:0>6d}.hdf5'.format(epoch+epoch_offset+1))
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./models/'+experiment_name+'/Test_best_acc.hdf5',
    monitor='val_accuracy',  
    save_best_only=True,  
    mode='max',  
    save_weights_only=False,  
    verbose=1  
)
checkpoint_callback_loss = tf.keras.callbacks.ModelCheckpoint(
    filepath='./models/'+experiment_name+'/Test_best_loss.hdf5',
    monitor='val_loss', 
    save_best_only=True,  
    mode='min',  
    save_weights_only=False,  
    verbose=1  
)
def step_decay(epoch, lr):
# #     print("step_decay_out_lr", lr)
    set_lr = learningrate
    initial_lrate = set_lr
    drop = 0.8
    epochs_drop = 100
    lrate = initial_lrate *(drop ** ((1+epoch+epoch_offset)/epochs_drop))
    if lrate>0.1*set_lr:
        print("decayed_lr", lrate)
        return lrate
    else:
        print("decayed_lr", 0.1*set_lr)
        return 0.1*set_lr
class ClassToken(Layer):
    def __init__(self, cls_initializer='zeros', cls_regularizer=None, cls_constraint=None, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.cls_initializer    = keras.initializers.get(cls_initializer)
        self.cls_regularizer    = keras.regularizers.get(cls_regularizer)
        self.cls_constraint     = keras.constraints.get(cls_constraint)
    def get_config(self):
        config = {
            'cls_initializer': keras.initializers.serialize(self.cls_initializer),
            'cls_regularizer': keras.regularizers.serialize(self.cls_regularizer),
            'cls_constraint': keras.constraints.serialize(self.cls_constraint),
        }
        base_config = super(ClassToken, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])
    def build(self, input_shape):
        self.num_features = input_shape[-1]
        self.cls = self.add_weight(
            shape       = (1, 1, self.num_features),
            initializer = self.cls_initializer,
            regularizer = self.cls_regularizer,
            constraint  = self.cls_constraint,
            name        = 'cls',
        )
        super(ClassToken, self).build(input_shape)
    def call(self, inputs):
        batch_size      = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(tf.broadcast_to(self.cls, [batch_size, 1, self.num_features]), dtype = inputs.dtype)
        return tf.concat([cls_broadcasted, inputs], 1)
def model_withvit(trace_length, encoder_num):
    _input = Input(shape = (trace_length,1))
    Local = LocallyConnected1D(filters=1, kernel_size=8, strides=4, padding='valid', activation=None, use_bias=True,
                              )(_input)
    Local = ZeroPadding1D((86, 0))(Local)
    Local_reshaped = Reshape((-1, head_size))(Local)
    x = ClassToken(name="cls_token")(Local_reshaped)
    for i in range(encoder_num):
        x_attention = LayerNormalization()(x)
        x_attention = MultiHeadAttention(num_heads,   
                                         key_dim,   
                                         #dropout=0.1,
                                         )(x_attention, x_attention)
        x = Add()([x, x_attention])
        x_mlp = LayerNormalization()(x)
        x_mlp = Dense(496, activation='relu')(x_mlp)
        x_mlp = Dense(124)(x_mlp)
        x = Add()([x, x_mlp])
    class_tensor = LayerNormalization()(x)
    y = Lambda(lambda v: v[:, 0], name="ExtractToken")(class_tensor)
    y = Dense(256, name="head1")(y)
    y = Dense(256, name="head2")(y)
    y = Softmax()(y)
    model = Model(inputs=_input, outputs=y)
    model.compile(loss='categorical_crossentropy',
                  optimizer=my_Adam,
                  metrics=['accuracy'])
    model.summary()
    return model
def dataload_process(path2profiling, path2val):
    file_profiling = h5py.File(path2profiling, 'r')
    file_val = h5py.File(path2val, 'r')
    traces_profiling = np.array(file_profiling["group/of/random/key"]["datasets/of/power/traces"][:num_traces, 0:samples], dtype=np.float64)
    scaler = preprocessing.StandardScaler()
    traces_profiling = scaler.fit_transform(traces_profiling)
    ################################################################
    label_profiling = np.array(file_profiling["group/of/random/key"]["datasets/of/secret/intermediate/variable"][:num_traces, 0])
    label_profiling = to_categorical(label_profiling, 256)
    traces_profiling = np.expand_dims(traces_profiling, 2)
    ###################################################################
    traces_val = np.array(file_val["group/of/random/key"]["datasets/of/power/traces"][num_traces:num_traces+num_attack, 0:samples], dtype=np.float64)
    traces_val = scaler.fit_transform(traces_val)
    #####################################################################
    label_val = np.array(file_val["group/of/random/key"]["datasets/of/secret/intermediate/variable"][num_traces:num_traces+num_attack, 0])
    label_val = to_categorical(label_val, 256)
    traces_val = np.expand_dims(traces_val, 2)
    #####################################################################
    return (traces_profiling, label_profiling), (traces_val, label_val)
def dataset():
      path2profiling = path2traces
      path2val = path2traces
      (traces_profiling, label_profiling), (traces_val, label_val) = dataload_process(path2profiling, path2val)
      train_dataset = tf.data.Dataset.from_tensor_slices((traces_profiling, label_profiling))
      test_dataset = tf.data.Dataset.from_tensor_slices((traces_val, label_val))
      BATCH_SIZE = batch_size
      SHUFFLE_BUFFER_SIZE = 256
      train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
      test_dataset = test_dataset.batch(BATCH_SIZE)
      return train_dataset, test_dataset
mycallback = SaveModelCallBack()
epochs_per_save = 5
epoch_offset = 0
my_Adam = Adam(lr=learningrate)
lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
model = model_withvit(samples, encoder_num=layers)
train_dataset, test_dataset = dataset()
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[mycallback, checkpoint_callback, checkpoint_callback_loss])
with open('history_'+experiment_name+'.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig('./train_output/val_loss_'+experiment_name+'.png')
plt.close()
plt.show()
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend()
plt.savefig('./train_output/val_acc_'+experiment_name+'.png')
plt.close()
plt.show()