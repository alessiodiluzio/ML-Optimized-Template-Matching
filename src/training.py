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

        # After each epochs a summary of Loss, F1Score and Accuracy for both train and validation phase is saved.
        # It can be used for training analysis and plot
        self.train_loss_history = None
        self.train_f1_score_history = None
        self.train_accuracy_history = None
        self.val_loss_history = None
        self.val_f1_score_history = None
        self.val_accuracy_history = None

    def get_training_data(self):
        return self.train_loss_history, self.train_f1_score_history, \
               self.train_accuracy_history, self.val_loss_history, self.val_f1_score_history, \
               self.val_accuracy_history

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
        print('TRACE UPDATE')
        tf.print('Improve ', metric_name, ' value: ', self.best_metric, ' ----> ', new_metric)
        self.best_metric.assign(new_metric)
        self.last_improvement.assign(0.)
        return tf.constant(0.)

    @tf.function
    def train_loop(self):
        train_loss = tf.TensorArray(tf.float32, size=self.train_steps)
        train_f1_score = tf.TensorArray(tf.float32, size=self.train_steps)
        train_acc = tf.TensorArray(tf.float32, size=self.train_steps)
        for step, (image, template, label) in enumerate(self.training_set):
            logits, loss = self.forward_backward_step([image, template], label)
            step = tf.cast(step, dtype=tf.int32)
            train_loss = train_loss.write(step, loss)
            prec = precision(logits, label)
            rec = recall(logits, label)
            train_f1_score = train_f1_score.write(step, f1score(prec, rec))
            acc = accuracy(logits, label)
            train_acc = train_acc.write(step, acc)
        train_loss = tf.reduce_mean(train_loss.stack())
        train_f1_score = tf.reduce_mean(train_f1_score.stack())
        train_acc = tf.reduce_mean(train_acc.stack())
        return train_loss, train_f1_score, train_acc

    @tf.function
    def val_loop(self):
        val_loss = tf.TensorArray(tf.float32, size=self.val_steps)
        val_f1_score = tf.TensorArray(tf.float32, size=self.val_steps)
        val_acc = tf.TensorArray(tf.float32, size=self.val_steps)
        for step, (image, template, label) in enumerate(self.validation_set):
            step = tf.cast(step, dtype=tf.int32)
            logits = self.forward_step([image, template])
            loss = self.loss_fn(logits, label, activation=None, balance_factor=self.loss_balance_factor, training=False)
            val_loss = val_loss.write(step, loss)
            prec = precision(logits, label)
            rec = recall(logits, label)
            val_f1_score = val_f1_score.write(step, f1score(prec, rec))
            acc = accuracy(logits, label)
            val_acc = val_acc.write(step, acc)
        val_loss = tf.reduce_mean(val_loss.stack())
        val_f1_score = tf.reduce_mean(val_f1_score.stack())
        val_acc = tf.reduce_mean(val_acc.stack())
        return val_loss, val_f1_score, val_acc

    @tf.function
    def __call__(self):

        train_loss_history = tf.TensorArray(tf.float32, size=self.epochs)
        train_f1_score_history = tf.TensorArray(tf.float32, size=self.epochs)
        train_accuracy_history = tf.TensorArray(tf.float32, size=self.epochs)

        val_loss_history = tf.TensorArray(tf.float32, size=self.epochs)
        val_f1_score_history = tf.TensorArray(tf.float32, size=self.epochs)
        val_accuracy_history = tf.TensorArray(tf.float32, size=self.epochs)

        pretty_line = '\n! --------------------------------------------------------- !\n'
        for epoch in tf.range(self.epochs):
            tf.print(pretty_line, 'Epoch: ', tf.add(epoch, 1), '/', self.epochs)

            tf.print('\nTRAIN')
            train_loss, train_f1_score, train_acc = self.train_loop()
            tf.print('\nLoss: ', train_loss, ' F1 Score: ', train_f1_score, ' Accuracy: ', train_acc)

            train_loss_history = train_loss_history.write(epoch, train_loss)
            train_f1_score_history = train_f1_score_history.write(epoch, train_f1_score)
            train_accuracy_history = train_accuracy_history.write(epoch, train_acc)

            tf.print("\nVALIDATE")
            val_loss, val_f1_score, val_acc = self.val_loop()
            tf.print('\nLoss: ', val_loss, ' F1 Score: ', val_f1_score, ' Accuracy: ', val_acc)

            val_loss_history = val_loss_history.write(epoch, val_loss)
            val_f1_score_history = val_f1_score_history.write(epoch, val_f1_score)
            val_accuracy_history = val_accuracy_history.write(epoch, val_acc)
            tf.cond(tf.less(val_loss, self.best_metric),
                    lambda: self.update(val_loss, 'Validation Loss'),
                    lambda: self.last_improvement.assign_add(1.))
            epoch = tf.cond(tf.greater_equal(self.last_improvement, self.early_stopping),
                            lambda: self.epochs, lambda: epoch)

        self.train_loss_history = train_loss_history.stack()
        self.train_f1_score_history = train_f1_score_history.stack()
        self.train_accuracy_history = train_accuracy_history.stack()

        self.val_loss_history = val_loss_history.stack()
        self.val_f1_score_history = val_f1_score_history.stack()
        self.val_accuracy_history = val_accuracy_history.stack()
