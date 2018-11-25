from fastai import *

# Checkpoint

class SaveBestModel(Callback):
    def __init__(self, model, batch_val, name='best_model'):
        super(SaveBestModel, self).__init__()
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None
        self.batch_val = batch_val
        self.num_batch = 0

    def on_batch_end(self, metrics):
        super().on_batch_end(metrics)
        self.num_batch += 1
        if self.num_batch == self.batch_val:
            loss, acc = metrics
            if self.best_acc == None or acc > self.best_acc:
                self.best_acc = acc
                self.best_loss = loss
                self.model.save(f'{self.name}')
            elif acc == self.best_acc and loss < self.best_loss:
                self.best_loss = loss
                self.model.save(f'{self.name}')
            self.num_batch = 0

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        loss, acc = metrics
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')
        elif acc == self.best_acc and  loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')


