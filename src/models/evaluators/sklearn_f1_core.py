from sklearn.metrics import f1_score
class SklearnF1Score():
    def __call__(self, predict,  target, labels=None, average='weighted'):
        return f1_score(y_true=target, y_pred=predict, labels=labels, average=average, zero_division=1)