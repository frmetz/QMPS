import numpy as np
import sys
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

tsize = 17 # 17
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=tsize)
plt.rc('ytick', labelsize=tsize)
plt.rc('axes', labelsize=tsize)
plt.rc('legend', fontsize=17) # 16
# plt.rc('legend', handlelength=2)
plt.rc('font', size=tsize)


ylabel = r'success [$\%$]'
ylabel2 = r'$\bar{T}$'

#########################################
#########################################
#########################################
# Main plot
#########################################
#########################################
#########################################

############################################################################
# horizontal
############################################################################

xlabel = r"error parameter $\lambda_{2/3}$"

# fig,aa = plt.subplots(figsize=(6.4, 4.8*1.1), dpi=300)
fig,aa = plt.subplots(figsize=(6.4*2, 4.8*1.6/2), dpi=300)

xx, y, h = 1, 20, 1
ax1 = plt.subplot2grid(shape=(xx,y), loc=(0, 0), rowspan=h, colspan=10)
# ax2 = plt.subplot2grid(shape=(xx,y), loc=(11, 0), rowspan=9, colspan=h)#, xticklabels=[])
ax2 = plt.subplot2grid(shape=(xx,y), loc=(0, 10), rowspan=h, colspan=10, yticklabels=[])

# plt.subplots_adjust(hspace=0)
axs = []
axs.append(ax1)
axs.append(ax2)

#########################################
# random depol
#########################################

save_name = "depol_new3"

full = np.load(f"data/_qiskit_{save_name}_full.npy")
trunc = np.load(f"data/_qiskit_{save_name}_trunc.npy")
x = np.load(f"data/_qiskit_{save_name}_x.npy")

data = [full[:2, :], trunc[:2, :]]

colors = ["purple", "tab:orange"]
# colors = ["purple", "tab:green"]
facecolors = ["magenta", 'orange']
# facecolors = ["magenta", 'lime']
markers = ["o", "s"]
linestyles = ["-", "-."]
labels = [r"$\chi=4$", r"$\chi=2$"]

ax1.axhline(data[0][0,0], 0, 1, color="black", linestyle='--', linewidth=2)#, label=r"exact")
# ax1.axhline(0, 0, 1, color="gray", linestyle='dotted', linewidth=2)#, label=r"random")


for i,d in enumerate(data):
    ax1.plot(x[1:], d[0,1:], color=colors[i], marker=markers[i], linestyle=linestyles[i], linewidth=2, label=labels[i])

ax1.set_ylabel(ylabel)
ax1.set_xlabel(xlabel)
ax1.yaxis.set_label_coords(-0.08, 0.5) #0.45
ax1.set_xscale('log')
ax1.legend(loc = "upper right")

ax1.text(0.95,0.3,"(b)", fontweight="bold", size=17, transform = ax1.transAxes, va='bottom', ha='right')
ax2.text(0.95,0.3,"(c)", fontweight="bold", size=17, transform = ax2.transAxes, va='bottom', ha='right')

data = [full[4:, :], trunc[4:, :]]

width, height = 1.99, 1.3
# axin = inset_axes(ax1, width=2.0, height=1.3, loc=3, borderpad=0, bbox_to_anchor=(0.09, 0.12), bbox_transform=ax1.transAxes) #height=1.54
axin = inset_axes(ax1, width=width, height=height, loc=3, borderpad=0, bbox_to_anchor=(0.085, 0.12), bbox_transform=ax1.transAxes) #height=1.54

axin.axhline(data[0][0,0], 0, 1, color="black", linestyle='--', linewidth=2)#, label=r"exact")
axin.axhline(50, 0, 1, color="gray", linestyle='dotted', linewidth=2)#, label=r"random")

for i,d in enumerate(data):
    axin.plot(x[1:], d[0,1:], color=colors[i], marker=markers[i], linestyle=linestyles[i], linewidth=2, label=labels[i])

    up_lim, low_lim = 50, 0
    data = [d+1 for d in data]
    up = d[0,1:]+d[1,1:]
    up[up>up_lim] = up_lim
    low = d[0,1:]-d[1,1:]
    low[low<low_lim] = low_lim
    axin.fill_between(x[1:], low, up, alpha=0.2, facecolor=facecolors[i], linewidth=0)

axin.set_yticks([20,40,50])
axin.set_xscale('log')
axin.set_ylabel(ylabel2)
axin.yaxis.set_label_coords(-0.07, 0.45) #0.45

#########################################
# AFM, Z, GHZ depol
#########################################

save_name = "all_depol_new3"

full_z = np.load(f"data/_qiskit_{save_name}_z.npy")
full_ghz = np.load(f"data/_qiskit_{save_name}_ghz.npy")
full_afm = np.load(f"data/_qiskit_{save_name}_afm.npy")
x = np.load(f"data/_qiskit_{save_name}_x.npy")

data = [full_z[:2, :], full_ghz[:2, :], full_afm[:2, :]]

colors = ["green", "blue", "crimson"]
facecolors = ["lime", 'cyan', 'magenta']
markers = ["o", "s", "d"]
linestyles = ["-", "--", "-."]
labels = [r"Z pol.", "GHZ", r"AFM"]

ax2.axhline(100, 0, 1, color="black", linestyle='--', linewidth=2)#, label=r"exact")
for i,d in enumerate(data):
    print(d)
    print(x)
    # plt.errorbar(x[1:], d[0,1:], yerr = d[1,1:], marker=markers[i], linestyle=linestyles[i], linewidth=2, label=labels[i], ecolor = 'magenta')
    ax2.plot(x[1:], d[0,1:], color=colors[i], marker=markers[i], linestyle=linestyles[i], linewidth=2, label=labels[i])

ax2.set_xlabel(xlabel)
# ax2.set_ylabel(ylabel)
ax2.yaxis.set_label_coords(-0.08, 0.5) #0.45
ax2.set_xscale('log')
ax2.legend(loc = "upper right")

data = [full_z[4:, :], full_ghz[4:, :], full_afm[4:, :]]

axin = inset_axes(ax2, width=width, height=height, loc=3, borderpad=0, bbox_to_anchor=(0.085, 0.12), bbox_transform=ax2.transAxes) #height=1.54

# axin.axhline(data[0][0,0], 0, 1, color="black", linestyle='--', linewidth=2)#, label=r"exact")
axin.axhline(50, 0, 1, color="gray", linestyle='dotted', linewidth=2)#, label=r"random")

for i,d in enumerate(data):
    axin.plot(x[1:], d[0,1:], color=colors[i], marker=markers[i], linestyle=linestyles[i], linewidth=2, label=labels[i])

    up_lim, low_lim = 50, 0
    data = [d+1 for d in data]
    up = d[0,1:]+d[1,1:]
    up[up>up_lim] = up_lim
    low = d[0,1:]-d[1,1:]
    low[low<low_lim] = low_lim
    axin.fill_between(x[1:], low, up, alpha=0.2, facecolor=facecolors[i], linewidth=0)

axin.set_yticks([10,30,50])
axin.set_xscale('log')
axin.set_ylabel(ylabel2)
axin.yaxis.set_label_coords(-0.07, 0.4) #0.45


plt.tight_layout()
# for ax in axs[:-1]:
#     ax.label_outer()
plt.savefig(f'cs4.png', dpi=300, bbox_inches = "tight")
plt.savefig(f'cs4.pdf', dpi=300, bbox_inches = "tight")
plt.close()


