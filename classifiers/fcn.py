# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from utils.utils import save_logs
from utils.utils import calculate_metrics

class Classifier_FCN:

	def __init__(self, output_directory, result_name, input_shape, nb_classes, nb_epochs, verbose=False,build=True):
		self.output_directory = output_directory
		self.result_name = result_name
		self.batch_size = 64
		self.nb_epochs = nb_epochs
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory + 'model_init' + '-' + result_name + '.hdf5')
		return

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=16, kernel_size=14, padding='same')(input_layer)#什么是filter
		# conv1 = keras.layers.BatchNormalization()(conv1) # 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
		#作用 （1）加速收敛 （2）控制过拟合，可以少用或不用Dropout和正则 （3）降低网络对初始化权重不敏感 （4）允许使用较大的学习率
		conv1 = keras.layers.Activation(activation='sigmoid')(conv1)

		conv2 = keras.layers.Conv1D(filters=32, kernel_size=114, padding='same')(conv1)#change 256 ->128
		# conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('sigmoid')(conv2)

		conv3 = keras.layers.Conv1D(32, kernel_size=14,padding='same')(conv2)
		# conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('sigmoid')(conv3)

		conv4 = keras.layers.Conv1D(16, kernel_size=14,padding='same')(conv3)
		# conv4 = keras.layers.BatchNormalization()(conv4)
		conv4 = keras.layers.Activation('sigmoid')(conv4)

		# conv4 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv3)
		# conv4 = keras.layers.BatchNormalization()(conv4)
		# conv4 = keras.layers.Activation('relu')(conv4)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv4)
		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)
		model.summary()
		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
			min_lr=0.0001)

		'''1.  factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
		2.  patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
		'''

		file_path = self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self,x_train, y_train, x_val, y_val,y_true):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  


		#mini_batch_size = int(min(x_train.shape[0]/5, batch_size)) # batch_size 有什么影响呢？
		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')

		model = keras.models.load_model(self.output_directory+'best_model' + '-' + self.result_name + '.hdf5')#学习重复结果

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory,self.result_name, hist, y_pred, y_true, duration)

		keras.backend.clear_session()


	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred
