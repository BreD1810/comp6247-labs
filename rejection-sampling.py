# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: 'Python 3.9.2 64-bit (''labs'': pipenv)'
#     metadata:
#       interpreter:
#         hash: 041ae1cec996e784c13a5e39f6c67964c28c09b992a33ed74944aab3055650c1
#     name: python3
# ---

# # Lab Worksheet, Week 5: Rejection Sampling

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import scipy as sp

# +
fontsize = 40
scattersize=100
surf_labelpad = 50.

plt.style.use("seaborn")
plt.rcParams.update({'figure.figsize': (20,15), 'font.size': fontsize, 'axes.labelsize': fontsize, 'axes.labelpad': 15., 'text.usetex':True, 'xtick.labelsize': fontsize, 'xtick.major.pad': 20., 'ytick.labelsize': fontsize, 'ytick.major.pad': 20., })
# -

rng = np.random.RandomState(13)


# Define target, $p(x)$, and proposal, $q(x)$:

# +
def p(x):
    return 0.3*sp.stats.norm.pdf(x, loc=30, scale=10) + 0.7*sp.stats.norm.pdf(x, loc=80, scale=20)


def q(x):
    return sp.stats.norm.pdf(x, loc=50, scale=30)



# -

# Rejection sampling:

def rejection_sampling(iter=1000, k=1.):
    samples = []
    for i in range(iter):
        # Insert your algorithm here!
        z = np.random.normal(50, 30)
        u = np.random.uniform(0, k * q(z))
        while u > p(z):
            z = np.random.normal(50, 30)
            u = np.random.uniform(0, k * q(z))
        samples.append(z)

    return np.array(samples)


# Run!

# +
x = np.linspace(-50,150,1000)
k = max(p(x) / q(x))

samples = rejection_sampling(iter=100000, k=k)
# samples = rejection_sampling(iter=100, k=k)
# -

# Plot distribution of samples and compare to target:

fig, ax = plt.subplots()
plt.plot(x, p(x), color='b', label='Target pdf, $p(z)$')
plt.plot(x, k*q(x), color='r', label='Comparison function, $k\,q(z)$')
plt.hist(samples, density=True, bins=50, alpha=0.5, label='Histogram of samples', color='b')
plt.xlabel('$z$')
plt.ylabel('Probability density')
plt.legend(fontsize=fontsize, loc='upper left')
plt.show()



