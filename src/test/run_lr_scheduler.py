from src.models.optimizers.lr_scheduler import WarmUpLRScheduler, ExpDecayLRScheduler, CosDecayLRScheduler
import matplotlib.pyplot as plt
import numpy as np

# lrs = [WarmUpLRScheduler(None, 768, 1, 1, 40)]
lrs = [ExpDecayLRScheduler(None, 1, 50, 1807, 4, 0.001, 1e-6, 0.1),
       CosDecayLRScheduler(None, 1, 50, 1807, 4, 0.001, 1e-6, 0.1)]
plt.plot(np.arange(0, 22587), [[lr.rate(i) for lr in lrs] for i in range(0,22587)])
plt.legend(['1','2'])
plt.show()
