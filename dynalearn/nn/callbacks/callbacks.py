class CallbackList:
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = list(callbacks)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch_number, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch_number, logs)

    def on_epoch_end(self, epoch_number, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch_number, logs)

    def on_batch_begin(self, batch_number, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch_number, logs)

    def on_batch_end(self, batch_number, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch_number, logs)

    def on_backward_end(self, batch_number):
        for callback in self.callbacks:
            callback.on_backward_end(batch_number)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback:
    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch_number, logs):
        pass

    def on_epoch_end(self, epoch_number, logs):
        pass

    def on_batch_begin(self, batch_number, logs):
        pass

    def on_batch_end(self, batch_number, logs):
        pass

    def on_backward_end(self, batch_number):
        pass

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass
