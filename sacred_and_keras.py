class LogMetrics(Callback):
    def on_epoch_end(self, _, logs={}):
        my_metrics(logs=logs)

@ex.capture
def my_metrics(_run, logs):
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("accuracy", float(logs.get('accuracy')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_accuracy", float(logs.get('val_accuracy')))
    _run.result = float(logs.get('val_accuracy'))

model.fit_generator(train_gen, callbacks=[LogMetrics()])