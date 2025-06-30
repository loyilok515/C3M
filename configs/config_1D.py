import numpy as np
from utils import temp_seed

# for ours training
# CAR
v_l = -2.
v_h = 2.


X_MIN = np.array([-50., v_l]).reshape(-1,1)
X_MAX = np.array([50., v_h]).reshape(-1,1)

lim = 1.  # Larger than the initial error for better performance
XE_MIN = np.array([-25, -lim]).reshape(-1,1)
XE_MAX = np.array([25, lim]).reshape(-1,1)

UREF_MIN = np.array([-1.]).reshape(-1,1)
UREF_MAX = np.array([ 1.]).reshape(-1,1)

# for sampling ref
X_INIT_MIN = np.array([0., 0.])
X_INIT_MAX = np.array([0., 0.])

XE_INIT_MIN = np.array([-10, 0.])
XE_INIT_MAX = np.array([ 10, 0.])

time_bound = 6.
time_step = 0.03
t = np.arange(0, time_bound, time_step)

state_weights = np.array([1,1,0.1,0.1])

def system_reset(seed):
    SEED_MAX = 10000000
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = np.array([0.,0.])
        xe_0 = np.array([0.,0.])
        x_0 = xref_0 + xe_0

        freqs = list(range(1,10+1))
        # freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        uref = []
        for _t in t:
            u = np.array([0.]) # ref
            #for freq, weight in zip(freqs, weights):
            #    u += np.array([weight[0] * np.sin(freq * _t/time_bound * 2*np.pi), 0])
            # u += 0.01*np.random.randn(2)
            uref.append(u)

    return x_0, xref_0, uref
