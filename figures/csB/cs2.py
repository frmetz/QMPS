
import math
import time
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

tsize = 17
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=tsize)
plt.rc('ytick', labelsize=tsize)
plt.rc('axes', labelsize=tsize)
plt.rc('legend', fontsize=17) # 16
# plt.rc('legend', handlelength=2)
plt.rc('font', size=tsize)

L = 32

################################################
# main figure
################################################

phi=1.01
stepss = np.load(f"data/steps_{phi}.npy")
acts = np.load(f"data/actions_{phi}.npy")
tot_steps = np.sum(stepss)

xlabel = r'steps'
# [hy, hz, nhz, Jx, nJx, Jy, nJy]
# colors = ["#762a83", "#762a83", "#af8dc3", "#af8dc3", "#e7d4e8", "#e7d4e8", "#1b7837", "#1b7837", "#7fbf7b", "#7fbf7b", "#d9f0d3", "#d9f0d3"]
# colors = ["#af8dc3", "#af8dc3", "#7fbf7b", "#7fbf7b", "#FFE399", "#FFE399","#762a83", "#762a83", "#1b7837", "#1b7837", "#A7811F", "#A7811F"]
# colors = ["#7fbf7b", "#FFE399", "#FFE399","#762a83", "#762a83", "#1b7837", "#1b7837"]
colors = ["#B2DFA4", "#FFA59E", "#FFA59E","blue", "blue", "green", "green"]
patterns = ["////"]*3 + [None]*4
alphas = [1.0, 1.0, 0.6, 0.6, 0.2, 0.2] *2
colorlist = [colors[int(a)] for a in acts]
alphalist = [alphas[int(a)] for a in acts]
patternlist = [patterns[int(a)] for a in acts]
aa = np.array([1.,1,-1,1,-1,1,-1])
actions = np.array([aa[int(a)] for a in acts])
color = "#364089"


fig,aa = plt.subplots(figsize=(6.4, 4.8*1.1), dpi=300)

x, y = 20, 1
v1, v2, h = 2, 3, 1
w = 8
r = w+2
s = r + h + 2
print(s)

print(2*h + w + 2*2)

ax1 = plt.subplot2grid(shape=(x,y), loc=(0, 0), rowspan=3, colspan=h)
ax2 = plt.subplot2grid(shape=(x,y), loc=(5, 0), rowspan=4, colspan=h, xticklabels=[])
ax3 = plt.subplot2grid(shape=(x,y), loc=(9, 0), rowspan=11, colspan=h)

plt.subplots_adjust(hspace=0)
axs = []
axs.append(ax2)
axs.append(ax3)
axs.append(ax1)


axs[2].axvspan(11+0.5, 24.5, ymin=0, ymax=1, color="cyan", alpha=0.2)
axs[2].axvspan(0.5, 3.5, ymin=0, ymax=1, color="yellow", alpha=0.2)
axs[2].bar(np.linspace(1,tot_steps,tot_steps), actions, color = colorlist, width = 0.8, hatch=patternlist)
axs[2].set_ylabel('action')
axs[2].yaxis.labelpad = 10
x1 = [-0.5,0,0.7]
squad = [r'$-$','',r'$+$']
axs[2].set_yticks(x1)
axs[2].set_yticklabels(squad, minor=False)
axs[2].yaxis.set_ticks_position('none')
axs[2].set_xlabel('time steps')
axs[2].set_xlim(xmin = 0.5, xmax=24.5)
# axs[2].xaxis.tick_top() # x axis on top
# axs[2].xaxis.set_label_position('top')
axs[2].xaxis.labelpad = -12
axs[2].set_xticks([4, 8, 16, 20, 24])
axs[2].text(0.99,0.1,r"$g_x=1.01$", fontweight="bold", transform = axs[2].transAxes, va='bottom', ha='right')
axs[2].text(-0.13,0.65,"(a)", fontweight="bold", size=17, transform = axs[2].transAxes, va='bottom', ha='right')


n=16
reward_list = np.load("data/_reward_list.npy")#[n:]
initial_reward_list = np.load("data/_initial_reward_list.npy")#[n:]
steps_list = np.load("data/_steps_list.npy")#[n:]
count_entangling = np.load("data/_count_entangling.npy")#[n:]
phis = np.linspace(1.0, 1.5, 100)#[n:]

iterations = 40000#20000
entropies = np.zeros((iterations))
string = "n_episodes40000_batch_size32_eps_init1.0_eps_final0.01_eps_decay1.0_target_update10_update_frequency1_D16_learning_rate5e-05_gamma0.98_buffer_size8000_n_feat72_hidden_dim200_uniformFalse_scale4.0_factor4.0_chiq32_seed33"
entropies = np.load("data/entropies_" + string + "_.npy")
final_entropy = 0.
# max_entropy = 2* np.log(16)/L
max_entropy = np.log(16)
skip = 200 #80
skip2 = 1#20
entropiess = np.mean(entropies.reshape(-1, skip), axis=1)
p = np.linspace(1,iterations,iterations)


# axs[0].axhline(np.log(1-0.01), 0, 1, color="tab:gray", linestyle='--', linewidth=2)
axs[0].axhline(1-0.01, 0, 1, color="tab:gray", linestyle='--', linewidth=2)
axs[0].axvline(1.0, 0.0, 1.0, color="tab:red", linestyle='--', linewidth=1.2)
axs[0].axvline(1.1, 0.0, 1.0, color="tab:red", linestyle='--', linewidth=1.2)
axs[0].plot(phis[1:],np.exp(reward_list[1:]), linestyle='None', marker="o", markersize=3, color="#364089")
axs[0].plot(phis[0],np.exp(reward_list[0]), linestyle='None', marker="h", markersize=5, color="deepskyblue") #dodgerblue
# axs[0].plot(phis[1:],reward_list[1:], linestyle='None', marker="o", markersize=3, color="#364089")
# axs[0].plot(phis[0],reward_list[0], linestyle='None', marker="h", markersize=5, color="deepskyblue") #dodgerblue
axs[0].set_xlim(xmin = 0.99, xmax=1.51)
# axs[0].set_ylim(ymin = -0.0114)
axs[0].set_ylim(ymin = 0.9888, ymax = 0.9953)
# axs[0].set_ylabel(r"$F^{N^{-1}}$")
# axs[0].set_ylabel(r"$F^{1/N}$")
axs[0].set_ylabel(r"$F_{\mathrm{sp}}$")#, color="#364089")
# axs[0].tick_params(axis='y')#, labelcolor="#364089")
# axs[0].set_ylabel(r"$N^{-1} \log F$")
axs[0].text(-0.13,1.0,"(b)", fontweight="bold", size=17, transform = axs[0].transAxes, va='bottom', ha='right')



axs[1].axvline(1.0, 0.0, 1.0, color="tab:red", linestyle='--', linewidth=1.2)
axs[1].axvline(1.1, 0.0, 1.0, color="tab:red", linestyle='--', linewidth=1.2)
axs[1].plot(phis,steps_list, label="all unitaries", color="black", linewidth=2)#, linestyle='None', marker="o", markersize=4)
axs[1].plot(phis, count_entangling, label="two-site unitaries", color="tab:purple", linewidth=2)#, linestyle='-.')#, marker="x", markersize=2)
axs[1].set_ylabel(r"protocol length")
axs[1].yaxis.labelpad = 10
axs[1].legend(loc = "upper left")
axs[1].set_xlim(xmin = 0.99, xmax=1.51)
axs[1].set_ylim(ymin = 0, ymax=70)
axs[1].yaxis.set_ticks(np.arange(0, 70, 10))
axs[1].set_xlabel(r'$g_x$')
axs[1].text(-0.13,0.85,"(c)", fontweight="bold", size=17, transform = axs[1].transAxes, va='bottom', ha='right')
axs[1].text(1.06,0.85,"(c)", fontweight="bold", color="white", size=17, transform = axs[1].transAxes, va='bottom', ha='right')


axin = inset_axes(axs[1], width=2.1, height=1.3, loc=1, borderpad=0, bbox_to_anchor=(0.96, 0.95), bbox_transform=axs[1].transAxes) #height=1.54
axin.scatter(p[::skip2], entropies[::skip2], s=0.05, c="limegreen", marker='o')#, cmap='viridis')
axin.plot(p[::skip], entropiess, color="darkgreen", linewidth=2)
# axin.yaxis.set_label_coords(-0.13, 0.15) #0.45
# axin.axhline(final_entropy, 0, 1, color="tab:orange", linestyle='--', linewidth=2)
# axin.axhline(max_entropy, 0, 1, color="tab:orange", linestyle='--', linewidth=2)
axin.set_xlabel(r'training episodes')
# axin.set_ylabel(r"$2N^{-1}S_{\textrm{ent}}^{N/2}$")
axin.set_ylabel(r"$S_{\textrm{ent}}^{N/2}$")
axin.yaxis.set_label_coords(-0.13, 0.32) #0.45

plt.tight_layout()
for ax in axs[:-1]:
    ax.label_outer()
plt.savefig(f'cs2.png', dpi=300, bbox_inches = "tight")
plt.savefig(f'cs2.pdf', dpi=200, bbox_inches = "tight")
plt.close()

