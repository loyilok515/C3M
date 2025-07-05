import numpy as np
from utils import temp_seed

# for training
X_REF = np.array([0., 0., 0., 0., 0., 2, 0, 0, 0, 0, 0, 0]).reshape(-1,1)

# X_MIN = np.array([-0.5, -0.5, -30., -30., -30., 1, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5]).reshape(-1,1)
# X_MAX = np.array([0.5, 0.5, 30., 30., 30., 3, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]).reshape(-1,1)

X_MIN = np.array([-30., -30., -30., -0.5, -0.5, 1, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5]).reshape(-1,1)  # For state reversal
X_MAX = np.array([30., 30., 30., 0.5, 0.5, 3, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]).reshape(-1,1)  # For state reversal

f_x_bound = 0.1
f_y_bound = 0.1
f_z_bound = 0.1
tau_bound = 0.1

UREF_MIN = np.array([-f_x_bound, -f_y_bound, -f_z_bound, -tau_bound]).reshape(-1,1)
UREF_MAX = np.array([f_x_bound, f_y_bound, f_z_bound, tau_bound]).reshape(-1,1)

lim = 1
# XE_MIN = np.array([-lim/10, -lim/10, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1,1)
# XE_MAX = np.array([lim/10, lim/10, lim, lim, lim, lim, lim, lim, lim, lim, lim, lim]).reshape(-1,1)

XE_MIN = np.array([-lim, -lim, -lim, -lim/10, -lim/10, -lim, -lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1,1)  # For state reversal
XE_MAX = np.array([lim, lim, lim, lim/10, lim/10, lim, lim, lim, lim, lim, lim, lim]).reshape(-1,1)  # For state reversal

# for sampling ref
# X_INIT_MIN = np.array([-0.2, -0.2, -0.5, -0.5, -0.5, 1.5, 0, 0, 0, 0, 0, 0])
# X_INIT_MAX = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 1.5, 0, 0, 0, 0, 0, 0])

X_INIT_MIN = np.array([-0.5, -0.5, -0.5, -0.2, -0.2, 1.5, 0, 0, 0, 0, 0, 0])  # For state reversal
X_INIT_MAX = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 1.5, 0, 0, 0, 0, 0, 0])  # For state reversal

# XE_INIT_MIN = np.array([-0.05, -0.05, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0])
# XE_INIT_MAX = np.array([0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0])

XE_INIT_MIN = np.array([-0.2, -0.2, -0.2, -0.05, -0.05, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0])  # For state reversal
XE_INIT_MAX = np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0])  # For state reversal

time_bound = 16.
time_step = 0.03
t = np.arange(0, time_bound, time_step)

def system_reset(seed):
    SEED_MAX = 10000000
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        x_0 = xref_0 + xe_0

        g = 9.81
        freqs = list(range(1, 11))
        # freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (2. * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        uref = []
        for _t in t:
            u = np.array([0., 0., 0., 0.])  # ref

            # Try circular path
            # ...
            
            # for freq, weight in zip(freqs, weights):
            #     u += np.array([weight[0] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[1] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[2] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[2] * np.sin(freq * _t/time_bound * 2*np.pi)])
            # u += 0.01*np.random.randn(2)
            uref.append(u)

    return x_0, xref_0, uref
