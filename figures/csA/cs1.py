import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
cmap = plt.get_cmap("plasma") #'viridis'
import matplotlib.colors as mcolors
import pickle as pkl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


tsize = 16
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=tsize)
plt.rc('ytick', labelsize=tsize)
plt.rc('axes', labelsize=tsize)
plt.rc('legend', fontsize=tsize)
# plt.rc('legend', handlelength=2)
plt.rc('font', size=tsize)


fig,aa = plt.subplots(figsize=(16,4.2), dpi=300)

x, y = 11, 34
v1, v2, h = 2, 3, 10
w = 8
r = w+2
s = r + h + 2
print(s)

print(2*h + w + 2*2)

# x = 2*v1+2*v2+1
# ax00 = plt.subplot2grid(shape=(x,y), loc=(0, 0), rowspan=2, colspan=5, xticklabels=[], yticklabels=[])
ax01 = plt.subplot2grid(shape=(x,y), loc=(3, 0), rowspan=8, colspan=w)
ax10 = plt.subplot2grid(shape=(x,y), loc=(0, r), rowspan=v1, colspan=h, xticklabels=[])
ax11 = plt.subplot2grid(shape=(x,y), loc=(v1, r), rowspan=v2, colspan=h)
ax12 = plt.subplot2grid(shape=(x,y), loc=(v1+v2+1, r), rowspan=v1, colspan=h, xticklabels=[])
ax13 = plt.subplot2grid(shape=(x,y), loc=(2*v1+v2+1, r), rowspan=v2, colspan=h)

ax20 = plt.subplot2grid(shape=(x,y), loc=(0, s), rowspan=v1, colspan=h, xticklabels=[])
ax21 = plt.subplot2grid(shape=(x,y), loc=(v1, s), rowspan=v2, colspan=h)
ax22 = plt.subplot2grid(shape=(x,y), loc=(v1+v2+1, s), rowspan=v1, colspan=h, xticklabels=[])
ax23 = plt.subplot2grid(shape=(x,y), loc=(2*v1+v2+1, s), rowspan=v2, colspan=h)

plt.subplots_adjust(hspace=0)



# colors = ["#762a83", "#762a83", "#af8dc3", "#af8dc3", "#e7d4e8", "#e7d4e8", "#1b7837", "#1b7837", "#7fbf7b", "#7fbf7b", "#d9f0d3", "#d9f0d3"]
# colors = ["#af8dc3", "#af8dc3", "#7fbf7b", "#7fbf7b", "#FFE399", "#FFE399","#762a83", "#762a83", "#1b7837", "#1b7837", "#A7811F", "#A7811F"]
colors = ["#C2BCFE", "#C2BCFE", "#B2DFA4", "#B2DFA4", "#FFA59E", "#FFA59E","blue", "blue", "green", "green", "red", "red"]
patterns = ["////"]*6 + [None]*6
alphas = [1.0, 1.0, 0.6, 0.6, 0.2, 0.2] *2
color = "#364089"
# color = "c"
color2 = "tab:brown"
# color2 = "firebrick"
color3="m"
alpha3=0.1

##########################################################
# Legend
##########################################################
l1 = mpatches.Patch(edgecolor="black", facecolor=colors[0], hatch="////", label=r'$\hat{X}$', lw=0.5)
l2 = mpatches.Patch(facecolor=colors[2], edgecolor="black", hatch="////", label=r'$\hat{Y}$', lw=0.5)
l3 = mpatches.Patch(facecolor=colors[4], edgecolor="black", hatch="////", label=r'$\hat{Z}$', lw=0.5)
l4 = mpatches.Patch(facecolor=colors[6], edgecolor="black", label=r'$\hat{X}\hat{X}$', lw=0.5)
l5 = mpatches.Patch(facecolor=colors[8], edgecolor="black", label=r'$\hat{Y}\hat{Y}$', lw=0.5)
l6 = mpatches.Patch(facecolor=colors[10], edgecolor="black", label=r'$\hat{Z}\hat{Z}$', lw=0.5)
ax01.legend(handles=[l1,l4,l2,l5,l3,l6], ncol=3, bbox_to_anchor=(0.04,1.2,1,0.2), loc="upper right", columnspacing=1.0, handletextpad=0.4, handlelength=1.5, labelspacing=0.5)

##########################################################
# Reward curve
##########################################################
eps = np.load("data/eps.npy")
rewards = np.load("data/rewards.npy")
up = np.load("data/rewards_best.npy")
low = np.load("data/rewards_worst.npy")

steps = np.load("data/steps.npy")
sup = np.load("data/steps_confint.npy")


ax01.plot(eps, rewards, color=color, linewidth=1.5)
# ax01.fill_between(eps, rewards+up, rewards-up, alpha=0.5)#, color=color)
ax01.fill_between(eps, up, low, alpha=0.5)#, color=color)
ax01.axhline(np.exp(np.log(1.-0.008)*4), 0.0, 1.0, color="tab:gray", linestyle='--', linewidth=1.5)
ax01.axhline(1.0, 0.0, 1.0, color="black", linestyle='dotted', linewidth=1.0)
#plt.title("Learning curve")
ax01.set_xlabel("training episodes")
ax01.set_ylabel(r'$\bar{F}$')

tmp = steps + sup
print(tmp)
tmp[tmp>50] = 50
print(tmp)

axin = inset_axes(ax01, width=1.3, height=1.05, loc=4, borderpad=0, bbox_to_anchor=(0.97, 0.15), bbox_transform=ax01.transAxes)
# axin = inset_axes(ax01, width=1.6, height=1.2, loc=4, borderpad=0, bbox_to_anchor=(0.95, 0.15), bbox_transform=ax01.transAxes)
axin.plot(eps, steps, color=color2, linewidth=1.5)
# axin.fill_between(eps, tmp, steps-sup, alpha=0.5, color=color2)
# axin.fill_between(eps, steps-slow, steps+sup, facecolor=color2, alpha=0.5)
axin.set_ylabel(r'$\bar{T}$')

#ax01.text(-0.27,1.05,"(a)", fontweight="bold", size=tsize, transform = ax01.transAxes)
# ax01.text(0.5,-0.35,"(a)", fontweight="bold", size=tsize, transform = ax01.transAxes)
ax01.text(0.155,0.08,"(a)", fontweight="bold", size=17, transform = ax01.transAxes)
# ax01.text(0.105,0.1,"(a)", fontweight="bold", size=17, transform = ax01.transAxes)

##########################################################
# z polarized
##########################################################
title = r"z polarized state"
rewards = np.load("data/rewards_z.npy")
acts = np.load("data/actions_z.npy")
stepss = np.load("data/steps_z.npy")
tot_steps = np.sum(stepss)

colorlist = [colors[int(a)] for a in acts]
alphalist = [alphas[int(a)] for a in acts]
patternlist = [patterns[int(a)] for a in acts]
actions = np.array([(-1)**a for a in acts], dtype=np.float64)
actions[stepss[0]:] = actions[stepss[0]:]/1.5

ax13.set_xlabel("time steps")
ax10.set_xlim((0, tot_steps+0.5))
# ax10.axvline(stepss[0]+0.5, 0.0, 1.0, color="black", linestyle='--', linewidth=1.5)
ax10.axvspan(0, stepss[0]+0.5, ymin=0, ymax=1, color=color3, alpha=alpha3)
ax10.bar(np.linspace(1,tot_steps,tot_steps), actions, color = colorlist, hatch=patternlist, width = 0.8)
ax10.set_ylim(ymin = -1.1, ymax=1.1)
ax10.set_ylabel('action')
x1 = [-0.5,0,0.7]
squad = [r'$-$','',r'$+$']
ax10.set_yticks(x1)
ax10.set_yticklabels(squad, minor=False)
ax10.yaxis.set_ticks_position('none')

ax11.set_xlim((0, tot_steps+0.5))
ax11.axvspan(0, stepss[0]+0.5, ymin=0, ymax=1, color=color3, alpha=alpha3)
ax11.axhline(np.exp(np.log(1-0.008)*4), 0.0, 1.0, color="gray", linestyle='--', linewidth=1.5)
ax11.axhline(np.exp(np.log(1-0.04)*4), 0.0, 1.0, color="gray", linestyle='--', linewidth=1.5)
ax11.plot(rewards, color=color, linewidth=1.5)
ax11.set_ylabel(r'$F$')#, color=color)
# ax11.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
import matplotlib.ticker as ticker
ax11.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
ax11.set_ylim([0, 1.1])
props = dict(facecolor='white', alpha=0.9)
# ax11.text(0.96, 0.12, title, transform=ax11.transAxes, fontsize=tsize, va='bottom', ha='right')
ax11.text(0.86, 0.14, title, transform=ax11.transAxes, fontsize=tsize, va='bottom', ha='right')

ax11.text(0.9,0.2,"(b)", fontweight="bold", size=17, transform = ax11.transAxes)

##########################################################
# GHZ state
##########################################################
title = r"GHZ state"
rewards = np.load("data/rewards_ghz.npy")
acts = np.load("data/actions_ghz.npy")
stepss = np.load("data/steps_ghz.npy")
tot_steps = np.sum(stepss)

colorlist = [colors[int(a)] for a in acts]
alphalist = [alphas[int(a)] for a in acts]
patternlist = [patterns[int(a)] for a in acts]
actions = np.array([(-1)**a for a in acts], dtype=np.float64)
actions[stepss[0]:] = actions[stepss[0]:]/1.5

ax12.set_xlim((0, tot_steps+0.5))
# ax12.axvline(stepss[0]+0.5, 0.0, 1.0, color="black", linestyle='--', linewidth=1.5)
ax12.axvspan(0, stepss[0]+0.5, ymin=0, ymax=1, color=color3, alpha=alpha3)
ax12.bar(np.linspace(1,tot_steps,tot_steps), actions, color = colorlist, hatch=patternlist, width = 0.8)
ax12.set_ylim(ymin = -1.1, ymax=1.1)
ax12.set_ylabel('action')
x1 = [-0.5,0,0.7]
squad = [r'$-$','',r'$+$']
ax12.set_yticks(x1)
ax12.set_yticklabels(squad, minor=False)
ax12.yaxis.set_ticks_position('none')

ax13.set_xlim((0, tot_steps+0.5))
ax13.axvspan(0, stepss[0]+0.5, ymin=0, ymax=1, color=color3, alpha=alpha3)
ax13.axhline(np.exp(np.log(1-0.008)*4), 0.0, 1.0, color="gray", linestyle='--', linewidth=1.5)
ax13.axhline(np.exp(np.log(1-0.04)*4), 0.0, 1.0, color="gray", linestyle='--', linewidth=1.5)
ax13.plot(rewards, color=color, linewidth=1.5)
ax13.set_ylabel(r'$F$')#, color=color)
ax13.set_ylim([0, 1.1])
ax13.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
props = dict(facecolor='white', alpha=0.9)
# ax13.text(0.96, 0.12, title, transform=ax13.transAxes, fontsize=tsize, va='bottom', ha='right')
ax13.text(0.86, 0.14, title, transform=ax13.transAxes, fontsize=tsize, va='bottom', ha='right')

ax13.text(0.9,0.2,"(c)", fontweight="bold", size=17, transform = ax13.transAxes)

# ax13.text(0.5,-0.45,"(b)-(c)", fontweight="bold", size=tsize, transform = ax13.transAxes)

##########################################################
# AFM
##########################################################
title = r"AFM Ising ground state"
rewards = np.load("data/rewards_afm.npy")
acts = np.load("data/actions_afm.npy")
stepss = np.load("data/steps_afm.npy")
tot_steps = np.sum(stepss)

colorlist = [colors[int(a)] for a in acts]
alphalist = [alphas[int(a)] for a in acts]
patternlist = [patterns[int(a)] for a in acts]
actions = np.array([(-1)**a for a in acts], dtype=np.float64)
actions[stepss[0]:] = actions[stepss[0]:]/1.5

ax23.set_xlabel("time steps")
ax20.set_xlim((0, tot_steps+0.5))
ax20.axvspan(0, stepss[0]+0.5, ymin=0, ymax=1, color=color3, alpha=alpha3)
# ax20.axvline(stepss[0]+0.5, 0.0, 1.0, color="black", linestyle='--', linewidth=1.5)
ax20.bar(np.linspace(1,tot_steps,tot_steps), actions, color = colorlist, hatch=patternlist, width = 0.8)
ax20.set_ylim(ymin = -1.1, ymax=1.1)
ax20.set_ylabel('action')
x1 = [-0.5,0,0.7]
squad = [r'$-$','',r'$+$']
ax20.set_yticks(x1)
ax20.set_yticklabels(squad, minor=False)
ax20.yaxis.set_ticks_position('none')

ax21.set_xlim((0, tot_steps+0.5))
ax21.axvspan(0, stepss[0]+0.5, ymin=0, ymax=1, color=color3, alpha=alpha3)
ax21.axhline(np.exp(np.log(1-0.008)*4), 0.0, 1.0, color="gray", linestyle='--', linewidth=1.5)
ax21.axhline(np.exp(np.log(1-0.04)*4), 0.0, 1.0, color="gray", linestyle='--', linewidth=1.5)
ax21.plot(rewards, color=color, linewidth=1.5)
ax21.set_ylabel(r'$F$')#, color=color)
ax21.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
ax21.set_ylim([0, 1.1])
props = dict(facecolor='white', alpha=0.9)
# ax21.text(0.96, 0.12, title, transform=ax21.transAxes, fontsize=tsize, va='bottom', ha='right')
ax21.text(0.86, 0.14, title, transform=ax21.transAxes, fontsize=tsize, va='bottom', ha='right')

ax21.text(0.9,0.2,"(d)", fontweight="bold", size=17, transform = ax21.transAxes)
##########################################################
# random state
##########################################################
title = r"random state"
rewards = np.load("data/rewards_r.npy")
acts = np.load("data/actions_r.npy")
stepss = np.load("data/steps_r.npy")
tot_steps = np.sum(stepss)

colorlist = [colors[int(a)] for a in acts]
alphalist = [alphas[int(a)] for a in acts]
patternlist = [patterns[int(a)] for a in acts]
actions = np.array([(-1)**a for a in acts], dtype=np.float64)
actions[stepss[0]:] = actions[stepss[0]:]/1.5

ax22.set_xlim((0, tot_steps+0.5))
ax22.axvspan(0, stepss[0]+0.5, ymin=0, ymax=1, color=color3, alpha=alpha3)
# ax22.axvline(stepss[0]+0.5, 0.0, 1.0, color="black", linestyle='--', linewidth=1.5)
ax22.bar(np.linspace(1,tot_steps,tot_steps), actions, color = colorlist, hatch=patternlist, width = 0.8)
ax22.set_ylim(ymin = -1.1, ymax=1.1)
ax22.set_ylabel('action')
x1 = [-0.5,0,0.7]
squad = [r'$-$','',r'$+$']
ax22.set_yticks(x1)
ax22.set_yticklabels(squad, minor=False)
ax22.yaxis.set_ticks_position('none')

ax23.set_xlim((0, tot_steps+0.5))
ax23.axvspan(0, stepss[0]+0.5, ymin=0, ymax=1, color=color3, alpha=alpha3)
ax23.axhline(np.exp(np.log(1-0.008)*4), 0.0, 1.0, color="gray", linestyle='--', linewidth=1.5)
ax23.axhline(np.exp(np.log(1-0.04)*4), 0.0, 1.0, color="gray", linestyle='--', linewidth=1.5)
ax23.plot(rewards, color=color, linewidth=1.5)
ax23.set_ylabel(r'$F$')#, color=color)
ax23.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
ax23.set_ylim([0, 1.1])
props = dict(facecolor='white', alpha=0.9)
# ax23.text(0.96, 0.12, title, transform=ax23.transAxes, fontsize=tsize, va='bottom', ha='right')
ax23.text(0.86, 0.14, title, transform=ax23.transAxes, fontsize=tsize, va='bottom', ha='right')

ax23.text(0.9,0.2,"(e)", fontweight="bold", size=17, transform = ax23.transAxes)



plt.savefig('cs1.png', bbox_inches = "tight")
plt.savefig('cs1.pdf', bbox_inches = "tight", dpi=200)
