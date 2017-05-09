import best
import best.plot
from pymc import MCMC
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy.random import normal, random
rc('text', usetex=True)

central = normal(0, 1, 100)

shifted = normal(0.2, 1, 100)

uniform = random(100)

data = {'Central Gauss': central,
        'Uniform': uniform}
        # 'Shifted Gauss': shifted}

model = best.make_model(data, separate_nu=True)

M = MCMC(model)
M.sample(iter=110000, burn=10000)

fig = best.plot.make_figure(M)

plt.show()

