#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Optimal Control tutorial for three-level population transfer
# Copyright (C) 2022 Luigi Giannelli

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# In[1]:


import os
import sys

module_path = os.path.abspath(os.path.join("../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

import copy
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from ThreeLS import *


# ## hyperparameters

# In[3]:


T = 1

Ωmax = 20
γ = 5

n_steps = 30

np.random.seed(11235813)

env_parameters = {
    "Ωmax": Ωmax,
    "n_steps": n_steps,
    "γ": γ,
    "T": T,
}

env_py = ThreeLS_v0_env(**env_parameters)

init_vals = np.random.rand(2 * n_steps) * Ωmax
init_amps = vals2amps(init_vals)
bounds = [(0, Ωmax) for a in range(2 * n_steps)]


# ## speed check

# In[4]:


# get_ipython().run_line_magic('timeit', 'env_py.run_mesolvevolution(init_amps)')
env_py.run_mesolvevolution(init_amps).states[-1]


# In[5]:


# get_ipython().run_line_magic('timeit', 'env_py.run_qstepevolution(init_amps)')
env_py.run_qstepevolution(init_amps)[-1]


# ## initial efficiency with random pulses

# In[4]:


# check initial efficiency
print(f"initial efficiency = {env_py.final_efficiency(init_amps)}")
print(f"initial inefficiency = {env_py.inefficiency(init_vals)}")
print(
    f"their sum is = {env_py.final_efficiency(init_amps) + env_py.inefficiency(init_vals)}"
)

# check initial efficiency with qstep
print("--- now with qstep")
print(f"initial efficiency = {env_py.final_qstepefficiency(init_amps)}")
print(f"initial inefficiency = {env_py.qstepinefficiency(init_vals)}")


# In[5]:


plot_evolution(init_amps, env_parameters, mesolve_check=True);


# ## optimization of the pulses - Powell

# In[10]:


#get_ipython().run_cell_magic('time', '', '\nsolver_opts_Powell = {\n    "disp": True,\n    "ftol": 1e-2,\n    "xtol": 1e-3,\n    "maxfev": int(1e4),\n    "maxiter": int(1e3),\n}\n\nres = minimize(\n    env_py.inefficiency,\n    init_vals,\n    args=(),\n    method="Powell",\n    bounds=bounds,\n    options=solver_opts_Powell,\n)\n')


# In[11]:


# final efficiency
print(f"final efficiency = {env_py.final_efficiency(vals2amps(res.x))}")
print(f"final inefficiency = {env_py.inefficiency(res.x)}")
print(
    f"their sum is = {env_py.final_efficiency(vals2amps(res.x)) + env_py.inefficiency(res.x)}"
)


# In[12]:


plot_evolution(vals2amps(res.x), env_parameters, mesolve_check=True);


# ## optimization of the pulses - Nelder-Mead

# In[14]:


#get_ipython().run_cell_magic('time', '', '\nsolver_opts_NM = {\n    "disp": True,\n    "fatol": 1e-2,\n    "xatol": 1e-3,\n    "maxfev": int(1e4),\n    #"maxiter": int(1e3),\n}\n\nres = minimize(\n    env_py.inefficiency,\n    init_vals,\n    args=(),\n    method="Nelder-Mead",\n    bounds=bounds,\n    options=solver_opts_NM,\n)\n')


# In[15]:


# final efficiency
print(f"final efficiency = {env_py.final_efficiency(vals2amps(res.x))}")
print(f"final inefficiency = {env_py.inefficiency(res.x)}")
print(
    f"their sum is = {env_py.final_efficiency(vals2amps(res.x)) + env_py.inefficiency(res.x)}"
)


# In[16]:


plot_evolution(vals2amps(res.x), env_parameters, mesolve_check=True);


# ## optimization of the pulses - L-BFGS-B

# In[6]:


solver_opts_LBFGSB = {
    "disp": 0,
    # "maxcor":10,
    "ftol": 1e-10,
    "gtol": 1e-8,
    "eps": 1e-3,
    "maxfun": int(2e5),
    "maxiter": int(1e3),
    "maxls": 50
}


# In[7]:


get_ipython().run_cell_magic('time', '', 'res = minimize(\n    env_py.inefficiency,\n    init_vals,\n    args=(),\n    method="L-BFGS-B",\n    bounds=bounds,\n    options=solver_opts_LBFGSB,\n)\n')


# In[8]:


# final efficiency
print(f"final efficiency = {env_py.final_efficiency(vals2amps(res.x))}")
print(f"final inefficiency = {env_py.inefficiency(res.x)}")
print(f"res.fun = {res.fun}")
print(f"result: {res.message}")


# In[9]:


plot_evolution(vals2amps(res.x), env_parameters, mesolve_check=True);


# # parallel execution

# In[10]:


solver_opts_novb = solver_opts_LBFGSB.copy()
solver_opts_novb["disp"] = 0

parameters = {
    "env_parameters": env_parameters,
    "init_vals": init_vals,
    "bounds": bounds,
    "solver_opts": solver_opts_novb,
}


# ### as a function of $\Omega_\mathrm{max}$

# In[11]:


env_parameters


# In[13]:


Ωmaxs = np.linspace(1., 100., 32)

np.random.seed(1123581321)

init_funcs = [Om, Omm1rand, Omgaussian, Omrand]

input_pars = []
for Ω_ in Ωmaxs:
    pars_ = copy.deepcopy(parameters)
    pars_["env_parameters"]["Ωmax"] = Ω_
    pars_["bounds"] = [
        (0, Ω_) for a in range(2 * pars_["env_parameters"]["n_steps"])
    ]
    
    # temptemp = np.random.rand(2 * env_parameters["n_steps"])
    
    for func in init_funcs:
        pars_ = copy.deepcopy(pars_)
        pars_["init_vals"] = func(Ω_, pars_["env_parameters"])
        input_pars.append(pars_)


# In[183]:


data = []
with Pool(processes=32) as pool:
    for x in tqdm(pool.imap(optimize_vs_pars, input_pars),
                  total=len(input_pars)):
        data += [x]


# In[188]:


ip = np.reshape(input_pars, (-1, len(init_funcs)))
d = np.reshape(data, (-1, len(init_funcs)))
# save_object([ip, d], "./data/gamma5_vs_Om.pkl")


# In[14]:


ip, d = load_object("./data/gamma5_vs_Om.pkl")


# In[15]:


nx, ny = ip.shape
assert nx, ny == d.shape

Ωmax_list = np.zeros((nx, ny))
ineff_list = np.zeros((nx, ny))
converging = np.zeros((nx, ny), dtype=bool)
for i in range(nx):
    for j in range(ny):
        Ωmax_list[i, j] = ip[i, j]["env_parameters"]["Ωmax"]
        try:
            ineff_list[i, j] = d[i, j].fun
            converging[i, j] = d[i, j].success
        except:
            ineff_list[i, j] = 1.
            converging[i, j] = False

ineff_list[ineff_list < 1e-6] = 1e-6


# In[16]:


# check if the minimization algorithm converged for every Ωmax
converged = np.all(converging)
print(f"has the minimization algorithm converged for every Ωmax? {converged}")
if not converged:
    print(
        f"indexes at which I did not converge: {np.argwhere(converging==False)}"
    )


# In[17]:


fig, ax = plt.subplots(1, 1)

ax.set_xlabel(r"$\Omega_\mathrm{max}$")
ax.set_ylabel("inefficiency")

# ax.set_xscale("log")
ax.set_yscale("log")

ax.plot(Ωmax_list[:, 0], ineff_list.min(axis=1), ".")
ax.plot(Ωmax_list[:, 0], ineff_list, ".")
ax.plot(Ωmax_list[~converging], ineff_list[~converging].T, ".", color="red")


# In[18]:


# plot evolution `i`
i = -1
plot_evolution(vals2amps(d[i, 0].x),
               ip[i, 0]["env_parameters"],
               mesolve_check=True)
print(f"result: {d[i,0].message}")


# # from saved data

# In[19]:


filenames = ["./data/gamma1_vs_Om.pkl", "./data/gamma5_vs_Om.pkl"]

labels = [
    r"$T\gamma = 1$",
    r"$T\gamma = 5$",
]

markers = [
    ".",
    "+",
]


# In[20]:


input_pars = []
data = []
Ωmax_lists = []
ineff_lists = []

for filename in filenames:
    ip, d = load_object(filename)

    nx, ny = ip.shape
    assert nx, ny == d.shape

    Ωmax_list = np.zeros((nx, ny))
    ineff_list = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            Ωmax_list[i, j] = ip[i, j]["env_parameters"]["Ωmax"]
            try:
                ineff_list[i, j] = d[i, j].fun
            except:
                ineff_list[i, j] = 1.

    input_pars.append(ip)
    data.append(d)
    Ωmax_lists.append(Ωmax_list)
    ineff_lists.append(ineff_list)


# In[21]:


fig, ax = plt.subplots(1, 1)

ax.set_xlabel(r"$T\Omega_\mathrm{max}$")
ax.set_ylabel("inefficiency")

#ax.set_ylim(5e-4, 1)

# ax.set_xscale("log")
ax.set_yscale("log")
for i in range(len(filenames)):
    # ax.plot(Ωmax_list[:, 0], ineff_list, ".")
    ax.scatter(Ωmax_lists[i][:, 0],
               ineff_lists[i].min(axis=1),
               marker=markers[i],
               label=labels[i])

ax.legend()
fig.tight_layout()


# In[ ]:




