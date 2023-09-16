import datetime
import keras_tuner
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from hyperopt.pyll.base import scope
from tensorflow import keras
from tensorflow.keras import layers


class ModelTrain:
    '''Class tuning model and then training.'''

    def __init__(self, train_x, val_x, train_y, val_y, module_url):
        '''Args:
            train_x (array): Numpy array of training utterances.
            val_x (array): Numpy array of validation utterances.
            train_y (array): Numpy array of one hot encoded training labels.
            val_y (array): Numpy array of one hot encoded validation labels.
            module_url (str): Variable for url to pretrained Universal
            Sentence Encoder model.
        '''

        self.train_x, self.val_x, self.train_y, self.val_y = train_x, val_x, train_y, val_y

        # initialise input layer with pretrained USE weights
        self.embed = hub.KerasLayer(module_url, input_shape=[], dtype=tf.string, trainable=True)

        # hyperparamter tuning
        self.tuner = keras_tuner.BayesianOptimization(
            self._build_model,
            objective="val_loss",
            max_trials=2,
            overwrite=True,
            directory="outputs/hp_dir",
            project_name="tune_hypermodel"
        )

        # Early stopping
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self._tune_parameters()

        # print best hyperparameters
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        # configure model with best hyperparameters
        self.model = self.tuner.hypermodel.build(best_hps)

        self.best_epoch = self._get_best_epoch()

        # monitor training on tensorboard
        log_dir = "log/intent_recognition/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
        self.tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir)

        self._train()

        print('Training complete!')

        # save model
        self.model.save("../outputs/saved_model_keras")

    def _tune_parameters(self):
        '''Performs hyperparameter tuning.'''

        self.tuner.search(
            self.train_x,
            self.train_y,
            epochs=1,
            validation_data=(self.val_x, self.val_y),
            callbacks=[self.early_stopping]
        )

    def _get_best_epoch(self):
        '''Gets best epoch number for training.'''

        history = self.model.fit(
            self.train_x,
            self.train_y,
            epochs=50,
            validation_data=(self.val_x, self.val_y),
            callbacks=[self.early_stopping],
            workers=4,
            use_multiprocessing=True
        )

        val_loss_per_epoch = history.history['val_loss']
        best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
        print(f'Best epoch: {best_epoch}')

        return best_epoch

    def _build_model(self, hp):
        '''Compile model'''

        model = tf.keras.models.Sequential()
        model.add(self.embed)
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                layers.Dense(
                    units=hp.Int("units", min_value=128, max_value=512, step=128),
                    activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"])
                )
            )
            model.add(
                layers.Dropout(
                    hp.Choice('dropout_rate', values=[0.1, 0.3, 0.5])
                )
            )
        model.add(layers.Dense(41, activation="softmax"))
        learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['categorical_accuracy']
        )
        return model

    def get_search_summary(self):
        '''Get summary of hyperparameter search space.'''

        return self.tuner.search_space_summary()

    def get_results_summary(self):
        '''Get summary of tuning results.'''

        return self.tuner.results_summary()

    def get_model_summary(self):
        '''Get summary of final model.'''

        return self.model.summary()

    def get_model_diagram(self):
        '''Get diagram of model architecture.'''

        return keras.utils.plot_model(self.model, "Model_Diagram.png", show_shapes=True)

    def _train(self):
        '''Train the model.'''

        # Train the model one final time while saving the best model.
        history = self.model.fit(
            self.train_x,
            self.train_y,
            epochs=self.best_epoch,
            validation_data=(self.val_x, self.val_y),
            callbacks=[self.early_stopping, self.tensorboard_cb],
            workers=4,
            use_multiprocessing=True
        )

        with open('../outputs/HistoryDict', 'wb') as f:
            pickle.dump(history.history, f)
