import tensorflow as tf

from src.metrics import precision, recall, accuracy, f1score


class Trainer:
    def __init__(self, model, training_set, validation_set, train_steps, val_steps, epochs,
                 optimizer, loss_fn, loss_balance_factor, device, early_stopping):

        # Variables needed to train a model
        self.model = model
        self.training_set = training_set
        self.validation_set = validation_set
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_balance_factor = tf.Variable(loss_balance_factor)
        self.device = device

        # Model weights are saved when the model outperform best metric for the current training process
        self.best_metric = tf.Variable(1000000.)

        # If early stopping is enabled, training process is stopped after early_stopping epochs since last improvement
        self.last_improvement = tf.Variable(0.)
        self.early_stopping = tf.Variable(epochs, dtype=tf.float32)
        if early_stopping is not None:
            self.early_stopping = tf.Variable(early_stopping, dtype=tf.float32)

        # The following variable are used to accumulate metric during each epoch
        self.train_loss_epoch = tf.Variable(0.)
        self.train_f1_score_epoch = tf.Variable(0.)
        self.train_accuracy_epoch = tf.Variable(0.)
        self.val_loss_epoch = tf.Variable(0.)
        self.val_f1_score_epoch = tf.Variable(0.)
        self.val_accuracy_epoch = tf.Variable(0.)

        self.best_weights = None

    @tf.function
    def forward_step(self, inputs):
        output = self.model(inputs, training=False)
        return output

    @tf.function
    def forward_backward_step(self, inputs, label):
        with tf.GradientTape() as tape:
            logits = self.model(inputs, training=True)
            loss = self.loss_fn(logits, label, activation=None,
                                balance_factor=self.loss_balance_factor, training=True)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return logits, loss

    @tf.function
    def update(self, new_metric, metric_name):
        tf.print('Improve ', metric_name, ' value: ', self.best_metric, ' ----> ', new_metric)
        self.best_metric.assign(new_metric)
        self.last_improvement.assign(0.)
        self.best_weights = self.model.weights
        return tf.constant(0.)

    @tf.function
    def train_loop(self):
        self.train_loss_epoch.assign(0.)
        self.train_f1_score_epoch.assign(0.)
        self.train_accuracy_epoch.assign(0.)
        for step, (image, template, label) in self.training_set.enumerate():
            logits, loss = self.forward_backward_step([image, template], label)
            self.train_loss_epoch.assign_add(loss)
            # prec = precision(logits, label)
            # rec = recall(logits, label)
            # self.train_f1_score_epoch.assign_add(f1score(prec, rec))
            # self.train_accuracy_epoch.assign_add(accuracy(logits, label))
            mean_train_loss = tf.divide(self.train_loss_epoch, tf.cast(step + 1, dtype=tf.float32))
            tf.print('\rTrain ', step + 1, '/', self.train_steps, 'Loss:', mean_train_loss, end='')
        tf.print('\n')

    @tf.function
    def val_loop(self):
        self.val_loss_epoch.assign(0.)
        self.val_f1_score_epoch.assign(0.)
        self.val_accuracy_epoch.assign(0.)
        for step, (image, template, label) in self.validation_set.enumerate():
            logits = self.forward_step([image, template])
            loss = self.loss_fn(logits, label, activation=None, balance_factor=self.loss_balance_factor, training=False)
            self.val_loss_epoch.assign_add(loss)
            # prec = precision(logits, label)
            # rec = recall(logits, label)
            # self.val_f1_score_epoch.assign_add(f1score(prec, rec))
            # self.val_accuracy_epoch.assign_add(accuracy(logits, label))
            mean_val_loss =  tf.divide(self.val_loss_epoch, tf.cast(step + 1, dtype=tf.float32))
            tf.print('\rValidate ', step + 1, '/', self.val_steps, 'Loss:', mean_val_loss, end='')
        tf.print('aaaa\n')

    @tf.function
    def __call__(self):

        train_loss_history = tf.TensorArray(tf.float32, size=self.epochs)
        # train_f1_score_history = tf.TensorArray(tf.float32, size=self.epochs)
        # train_accuracy_history = tf.TensorArray(tf.float32, size=self.epochs)

        val_loss_history = tf.TensorArray(tf.float32, size=self.epochs)
        # val_f1_score_history = tf.TensorArray(tf.float32, size=self.epochs)
        # val_accuracy_history = tf.TensorArray(tf.float32, size=self.epochs)

        pretty_line = '\n# ---------------------------------------------------------'
        for epoch in tf.range(self.epochs):
            tf.print(pretty_line)
            tf.print('Epoch: ', tf.add(epoch, 1), '/', self.epochs)

            self.train_loop()
            train_loss = tf.divide(self.train_loss_epoch, self.train_steps)
            # train_f1_score = tf.divide(self.train_f1_score_epoch, self.train_steps)
            # train_accuracy = tf.divide(self.train_accuracy_epoch, self.train_steps)

            # tf.print('\nLoss: ', train_loss, ' F1 Score: ', train_f1_score, ' Accuracy: ', train_accuracy, '\n')
            train_loss_history = train_loss_history.write(epoch, train_loss)
            # train_f1_score_history = train_f1_score_history.write(epoch, train_f1_score)
            # train_accuracy_history = train_accuracy_history.write(epoch, train_accuracy)
            self.val_loop()
            val_loss = tf.divide(self.val_loss_epoch, self.val_steps)
            # val_f1_score = tf.divide(self.val_f1_score_epoch, self.val_steps)
            # val_accuracy = tf.divide(self.val_accuracy_epoch, self.val_steps)
            # tf.print('\nLoss: ', val_loss, ' F1 Score: ', val_f1_score, ' Accuracy: ', val_accuracy, '\n')

            val_loss_history = val_loss_history.write(epoch, val_loss)
            # val_f1_score_history = val_f1_score_history.write(epoch, val_f1_score)
            # val_accuracy_history = val_accuracy_history.write(epoch, val_accuracy)

            tf.cond(tf.less(val_loss, self.best_metric),
                    lambda: self.update(val_loss, 'Validation Loss'),
                    lambda: self.last_improvement.assign_add(1.))
            epoch = tf.cond(tf.greater_equal(self.last_improvement, self.early_stopping),
                            lambda: self.epochs, lambda: epoch)

        train_loss_history = train_loss_history.stack()
        # train_f1_score_history = train_f1_score_history.stack()
        # train_accuracy_history = train_accuracy_history.stack()

        val_loss_history = val_loss_history.stack()
        # val_f1_score_history = val_f1_score_history.stack()
        # val_accuracy_history = val_accuracy_history.stack()

        # return train_loss_history, train_f1_score_history, train_accuracy_history, \
        #       val_loss_history, val_f1_score_history, val_accuracy_history, self.best_weights

        return train_loss_history, val_loss_history, self.best_weights
