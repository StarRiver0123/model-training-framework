from src.modules.trainer.trainer_framework import Trainer

config = 'start'
trainer = Trainer(config)
model = 1
train_iter = 2
valid_iter = 3
epoch_validate = True

def epoch_train(model, train_iter):
    print("epoch_train")
    a = model + train_iter
    print("epoch_train -> ",a)
    return a


a = trainer.train(model=model, train_iter=train_iter, epoch_train = epoch_train, valid_iter=valid_iter, epoch_validate = epoch_validate)
print("test page tend ->", a)

