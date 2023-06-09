import sys
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cmap = plt.get_cmap("plasma") #'viridis'
import pickle as pkl


tsize = 16
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=tsize)
plt.rc('ytick', labelsize=tsize)
plt.rc('axes', labelsize=tsize)
plt.rc('legend', fontsize=tsize)
# plt.rc('legend', handlelength=2)
plt.rc('font', size=tsize)

points = 20
thetas = np.linspace(0.0, 3.0, points)
phis = np.linspace(0.0, 4.0, points)
L=16

################################################
# Main plot
################################################


cmap1 = "Blues_r"
cmap2 = "Wistia"
# cmap3 = "pink"#"copper_r"#"pink"#"OrRd_r"
# cmap4 = "summer"

cmap3 = "Greens_r"#"copper_r"#"pink"#"OrRd_r"
cmap4 = "spring_r"

fig,aa = plt.subplots(figsize=(16,5.8), dpi=300)
# fig,aa = plt.subplots(figsize=(6.4*2,5.8/16*6.4*2), dpi=300)
# fig,aa = plt.subplots(figsize=(6.4*2, 4.8), dpi=300)

rs = 10
cs = 2*5
r = cs+1
x, y = 2*rs+2, 3*r+6 + cs+2*3 + 1


ax00 = plt.subplot2grid(shape=(x,y), loc=(0, 0), rowspan=rs, colspan=cs, xticklabels=[])
ax01 = plt.subplot2grid(shape=(x,y), loc=(rs+2, 0), rowspan=rs, colspan=cs)
ax10 = plt.subplot2grid(shape=(x,y), loc=(0, r), rowspan=rs, colspan=cs, xticklabels=[], yticklabels=[])
ax11 = plt.subplot2grid(shape=(x,y), loc=(rs+2, r), rowspan=rs, colspan=cs, yticklabels=[])
ax20 = plt.subplot2grid(shape=(x,y), loc=(0, 2*r), rowspan=rs, colspan=cs, xticklabels=[], yticklabels=[])
ax21 = plt.subplot2grid(shape=(x,y), loc=(rs+2, 2*r), rowspan=rs, colspan=cs, yticklabels=[])

ax30 = plt.subplot2grid(shape=(x,y), loc=(0, 3*r), rowspan=5, colspan=1, xticklabels=[], yticklabels=[])
ax301 = plt.subplot2grid(shape=(x,y), loc=(5, 3*r), rowspan=5, colspan=1, xticklabels=[], yticklabels=[])
ax31 = plt.subplot2grid(shape=(x,y), loc=(rs+2, 3*r), rowspan=5, colspan=1, yticklabels=[])
ax311 = plt.subplot2grid(shape=(x,y), loc=(rs+2+5, 3*r), rowspan=5, colspan=1, yticklabels=[])

ax40 = plt.subplot2grid(shape=(x,y), loc=(0, 3*r+7), rowspan=rs, colspan=cs+2*3)
ax41 = plt.subplot2grid(shape=(x,y), loc=(rs+2, 3*r+7), rowspan=rs, colspan=cs+2*3)
plt.subplots_adjust(hspace=0)


smin = 21
smax = 60.5

# -0.022740580141544342 -0.35256266593933105*****
# -0.022537967190146446**** -0.20184282958507538
# -0.023022601380944252 -0.1832265853881836


rmin = -0.352562665939
# rmin = -0.06
r = np.log(1-0.03)
rmax = -0.02253796719
rmax = np.log(0.9805)
# print(np.log(0.981))

norm = mpl.colors.Normalize(vmin=0.97, vmax=np.exp(rmax))
# norm = mpl.colors.Normalize(vmin=rmin, vmax=rmax)
cbar = ax30.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap1),
            ax=ax30, pad=.05, fraction=1.5,aspect=10)#, spacing='proportional')#, extend='both', fraction=1)
# cbar.ax.set_title(r'$N^{-1} \log F$')#,fontsize=20)
# cbar.ax.set_title(r"$F^{1/N}$")
cbar.ax.set_title(r"$F_{\mathrm{sp}}$", size=tsize)
cbar.ax.axhline(1-0.03, -0.5, 1.5, color = 'black', linestyle = '-', linewidth=2.0, clip_on=False)
# cbar.ax.axhline(0.97, 0, 1, color= 'r', linewidth = 4, linestyle = ':')
# cbar.ax.axhline(np.log(1-0.03), 0, 1, color = 'w', linestyle = '-', linewidth=3.0)
cbar.ax.set_yticks([0.97,0.975, 0.98])
# cbar.ax.get_yticklabels()[1].set_weight("bold")
labels = [item.get_text() for item in cbar.ax.get_yticklabels()]
labels[0] = r'$\mathbf{0.97}$'
labels[1] = 0.975
labels[2] = 0.98

cbar.ax.set_yticklabels(labels)
# cbar.ax.set_major_formatter(dates.DateFormatter(r'\textbf{%m/%Y}'))
# labels = cbar.ax.get_yticklabels()
# [label.set_fontweight('bold') for label in labels]
ax30.axis('off')

norm = mpl.colors.Normalize(vmin=np.exp(rmin), vmax=0.97)
cbar = ax301.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap2),
            ax=ax301, pad=.05, fraction=1.5,aspect=10)#, spacing='proportional')#, extend='both', fraction=1)
cbar.ax.axhline(1-0.03, -0.5, 1.5, color = 'black', linestyle = '-', linewidth=2.0, clip_on=False)
ax301.axis('off')


norm = mpl.colors.Normalize(vmin=50, vmax=smax)
cbar = ax31.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap4),
            ax=ax31, pad=.05, fraction=1.5,aspect=10)#, spacing='proportional')#, extend='both', fraction=1)
cbar.ax.set_title(r"protocol length", size=tsize)
cbar.ax.axhline(50, -0.5, 1.5, color = 'black', linestyle = '-', linewidth=2.0, clip_on=False)
cbar.ax.set_yticks([50,55, 60])
labels = [item.get_text() for item in cbar.ax.get_yticklabels()]
labels[0] = r'$\mathbf{50}$'
labels[1] = 55
labels[2] = 60

cbar.ax.set_yticklabels(labels)
ax31.axis('off')

norm = mpl.colors.Normalize(vmin=smin, vmax=50)
cbar = ax311.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap3),
            ax=ax311, pad=.05, fraction=1.5,aspect=10)#, spacing='proportional')#, extend='both', fraction=1)
cbar.ax.axhline(50, -0.5, 1.5, color = 'black', linestyle = '-', linewidth=2.0, clip_on=False)
cbar.ax.set_yticks([30,40, 50])
labels = [item.get_text() for item in cbar.ax.get_yticklabels()]
labels[0] = 30
labels[1] = 40
# labels[2] = 60

cbar.ax.set_yticklabels(labels)
ax311.axis('off')




reward_list = np.load("data/_reward_0.npy", allow_pickle=True).reshape((points,points))
steps_list = np.load("data/_steps_0.npy", allow_pickle=True).reshape((points,points))

print(np.max(reward_list), np.min(reward_list))
print(np.max(steps_list), np.min(steps_list))
print()
# rmax = -0.003

rew = np.exp(reward_list)
idx = np.where(rew < 0.97)
idx2 = np.where(rew >= 0.97)
rew[idx] = np.nan
im = ax00.pcolormesh(phis, thetas, rew.T, cmap=cmap1, shading="auto", vmin=0.97, vmax=np.exp(rmax))
rew = np.exp(reward_list)
rew[idx2] = np.nan
im = ax00.pcolormesh(phis, thetas, rew.T, cmap=cmap2, shading="auto", vmin=np.exp(rmin), vmax=0.97)# cbar = fig.colorbar(im, ax=ax00,location='left')
# # cbar.ax.set_ylabel(r'$N^{-1} \log F$')#,fontsize=20)
# cbar.ax.set_title(r'$N^{-1} \log F$')#,fontsize=20)
ax00.hlines(0.5, 1.0, 1.5, colors="white", linestyles='-', linewidth=3)
ax00.vlines(1.0, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax00.vlines(1.5, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
# ax00.set_xlabel(r"$g_x$")
ax00.set_ylabel(r"$g_z$")
ax00.set_title(r"w/o noise", size=tsize)
ax00.text(-0.,1.05,"(a)", fontweight="bold", size=17, transform = ax00.transAxes)
ax00.set_xticks([0,1,2,3,4])
# cbar.ax.axhline(np.log(1-0.03), 0, 1, color = 'w', linestyle = '-', linewidth=3.0)

rew = steps_list.copy()
idx = np.where(rew <= 50)
idx2 = np.where(rew > 50)
rew[idx2] = np.nan
im = ax01.pcolormesh(phis, thetas, rew.T, cmap=cmap3, shading="auto", vmin=smin, vmax=50)
rew = steps_list
rew[idx] = np.nan
im = ax01.pcolormesh(phis, thetas, rew.T, cmap=cmap4, shading="auto", vmin=50, vmax=smax)
# cbar = fig.colorbar(im, ax=ax01,location='left')
# # cbar.ax.set_ylabel("protocol length")
# cbar.ax.set_title(r"protocol length")
ax01.hlines(0.5, 1.0, 1.5, colors="white", linestyles='-', linewidth=3)
ax01.vlines(1.0, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax01.vlines(1.5, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax01.set_xlabel(r"$g_x$")
ax01.set_ylabel(r"$g_z$")
ax01.text(-0.,1.05,"(b)", fontweight="bold", size=17, transform = ax01.transAxes)
ax01.set_xticks([0,1,2,3,4])

reward_list = np.load("data/_reward_no1.npy", allow_pickle=True).reshape((points,points))
steps_list = np.load("data/_steps_no1.npy", allow_pickle=True).reshape((points,points))
print(np.max(reward_list), np.min(reward_list))
print(np.max(steps_list), np.min(steps_list))
print()


rew = np.exp(reward_list)
idx = np.where(rew < 0.97)
idx2 = np.where(rew >= 0.97)
rew[idx] = np.nan
im = ax10.pcolormesh(phis, thetas, rew.T, cmap=cmap1, shading="auto", vmin=0.97, vmax=np.exp(rmax))
rew = np.exp(reward_list)
rew[idx2] = np.nan
im = ax10.pcolormesh(phis, thetas, rew.T, cmap=cmap2, shading="auto", vmin=np.exp(rmin), vmax=0.97)# cbar = fig.colorbar(im, ax=ax10)
# cbar.ax.set_ylabel(r'$N^{-1} \log F$')#,fontsize=20)
ax10.hlines(0.5, 1.0, 1.5, colors="white", linestyles='-', linewidth=3)
ax10.vlines(1.0, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax10.vlines(1.5, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
# ax00.set_xlabel(r"$g_x$")
# ax10.set_ylabel(r"$g_z$")
# cbar.ax.axhline(np.log(1-0.03), 0, 1, color = 'w', linestyle = '-', linewidth=3.0)
ax10.set_title(r"$\epsilon=0.02$", size=tsize)
ax10.text(-0.0,1.05,"(c)", fontweight="bold", size=17, transform = ax10.transAxes)
ax10.set_xticks([0,1,2,3,4])

rew = steps_list.copy()
idx = np.where(rew <= 50)
idx2 = np.where(rew > 50)
rew[idx2] = np.nan
im = ax11.pcolormesh(phis, thetas, rew.T, cmap=cmap3, shading="auto", vmin=smin, vmax=50)
rew = steps_list
rew[idx] = np.nan
im = ax11.pcolormesh(phis, thetas, rew.T, cmap=cmap4, shading="auto", vmin=50, vmax=smax)
# cbar.ax.set_ylabel("protocol length")
ax11.hlines(0.5, 1.0, 1.5, colors="white", linestyles='-', linewidth=3)
ax11.vlines(1.0, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax11.vlines(1.5, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax11.set_xlabel(r"$g_x$")
ax11.text(-0.0,1.05,"(d)", fontweight="bold", size=17, transform = ax11.transAxes)
ax11.set_xticks([0,1,2,3,4])



reward_list = np.load("data/_reward_no3.npy", allow_pickle=True).reshape((points,points))
steps_list = np.load("data/_steps_no3.npy", allow_pickle=True).reshape((points,points))
print(np.max(reward_list), np.min(reward_list))
print(np.max(steps_list), np.min(steps_list))
print()

rew = np.exp(reward_list)
idx = np.where(rew < 0.97)
idx2 = np.where(rew >= 0.97)
rew[idx] = np.nan
im = ax20.pcolormesh(phis, thetas, rew.T, cmap=cmap1, shading="auto", vmin=0.97, vmax=np.exp(rmax))
rew = np.exp(reward_list)
rew[idx2] = np.nan
im = ax20.pcolormesh(phis, thetas, rew.T, cmap=cmap2, shading="auto", vmin=np.exp(rmin), vmax=0.97)
# # cbar.ax.set_ylabel(r'$N^{-1} \log F$')#,fontsize=20)
# cbar.ax.set_title(r'$N^{-1} \log F$')#,fontsize=20)
ax20.hlines(0.5, 1.0, 1.5, colors="white", linestyles='-', linewidth=3)
ax20.vlines(1.0, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax20.vlines(1.5, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
# cbar.ax.axhline(np.log(1-0.03), 0, 1, color = 'w', linestyle = '-', linewidth=3.0)
ax20.set_title(r"$\sigma=0.01$", size=tsize)
ax20.text(-0.0,1.05,"(e)", fontweight="bold", size=17, transform = ax20.transAxes)
ax20.set_xticks([0,1,2,3,4])

rew = steps_list.copy()
idx = np.where(rew <= 50)
idx2 = np.where(rew > 50)
rew[idx2] = np.nan
im = ax21.pcolormesh(phis, thetas, rew.T, cmap=cmap3, shading="auto", vmin=smin, vmax=50)
rew = steps_list
rew[idx] = np.nan
im = ax21.pcolormesh(phis, thetas, rew.T, cmap=cmap4, shading="auto", vmin=50, vmax=smax)
# im = ax21.pcolormesh(phis, thetas, steps_list.T, cmap=cmap, shading="auto")#, vmin=smin)
# cbar = fig.colorbar(im, ax=ax21)#,location='left')
# # cbar.ax.set_ylabel("protocol length")
# cbar.ax.set_title(r"protocol length")
ax21.hlines(0.5, 1.0, 1.5, colors="white", linestyles='-', linewidth=3)
ax21.vlines(1.0, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax21.vlines(1.5, -0.06, 0.53, colors="white", linestyles='-', linewidth=3)
ax21.set_xlabel(r"$g_x$")
# ax11.set_ylabel(r"$g_z$")
ax21.text(-0.0,1.05,"(f)", fontweight="bold", size=17, transform = ax21.transAxes)
ax21.set_xticks([0,1,2,3,4])



##########
# reward vs time steps
##########


phi = 2.0 #2.0
theta = 2.0 # 2.0
rewardss = np.load(f"data/__reward_{phi}_{theta}_5_4.npy", allow_pickle=True)
rewardss2 = np.load(f"data/__reward_{phi}_{theta}_001.npy", allow_pickle=True)

p = len(rewardss)
p=6
cmap = mpl.cm.get_cmap('winter')
rgba = cmap(0.5)
rgbas = [cmap(i/(p-1)) for i in range(p)]

aa = 4
j = 0
for i,r in enumerate(rewardss):
    if i % 2 != 0: continue
    ax40.plot(np.exp(r), color=rgbas[j])#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
    # ax40.plot(r, color=rgbas[i])#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
    if i != aa: ax40.plot(len(r)-1, np.exp(r[-1]), color=rgbas[j], marker = "o", markersize=4)#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
    j +=1
ax40.plot(np.exp(rewardss[aa]), color="red", linewidth=2.0)
ax40.plot(len(rewardss[aa])-1, np.exp(rewardss[aa][-1]), color="red", marker = "o", markersize=4)
# plt.plot(rewardss[0], color="black")
ax40.axhline(1-0.03, 0.0, 1.0, color="tab:gray", linestyle='--', linewidth=1.5)
ax40.arrow(4, 0.965, 0, -0.035, length_includes_head=True, width=0.4,
          head_width=1.2, head_length=0.01, color="black")
# ax40.axhline(np.log(1-0.03), 0.0, 1.0, color="tab:gray", linestyle='--', linewidth=1.5)
# ax40.set_ylabel(r'$N^{-1} \log F$')
# ax40.set_ylabel(r"$F^{1/N}$")
ax40.set_ylabel(r"$F_{\mathrm{sp}}$")
ax40.set_title(r"$g_x=g_z=2.0$", size=tsize)
ax40.text(-0.1,1.05,"(g)", fontweight="bold", size=17, transform = ax40.transAxes)
ax40.set_yticks([0.8, 0.9, 0.97])

axin = inset_axes(ax40, width=1.8, height=0.8, loc=4, borderpad=0, bbox_to_anchor=(0.95, 0.18), bbox_transform=ax40.transAxes)
j = 0
for i,r in enumerate(rewardss):
    if i % 2 != 0: continue
    axin.plot(np.exp(r), color=rgbas[j])#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
    # ax40.plot(r, color=rgbas[i])#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
    if i != aa: axin.plot(len(r)-1, np.exp(r[-1]), color=rgbas[j], marker = "o", markersize=4)#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
    j +=1
axin.plot(np.exp(rewardss[aa]), color="red", linewidth=2.0)
axin.plot(len(rewardss[aa])-1, np.exp(rewardss[aa][-1]), color="red", marker = "o", markersize=4)
axin.set_ylim(0.959,0.977)
axin.set_xlim(xmin=22)
axin.axhline(1-0.03, 0.0, 1.0, color="tab:gray", linestyle='--', linewidth=1.5)


aa = 0
j = 0
for i,r in enumerate(rewardss2):
    if i in [1,4,5,7,8]: #[1,4,5,6,7,8]
        ax41.plot(np.exp(r), color=rgbas[j])#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
        if i != aa: ax41.plot(len(r)-1, np.exp(r[-1]), color=rgbas[j], marker = "o", markersize=4)#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
        j +=1
ax41.plot(np.exp(rewardss2[aa]), color="red", linewidth=2.0)
ax41.plot(len(rewardss2[aa])-1, np.exp(rewardss2[aa][-1]), color="red", marker = "o", markersize=4)
# plt.plot(rewardss[0], color="black")
ax41.axhline(1-0.03, 0.0, 1.0, color="tab:gray", linestyle='--', linewidth=1.5)
# ax41.axhline(np.log(1-0.03), 0.0, 1.0, color="tab:gray", linestyle='--', linewidth=1.5)
# ax41.set_ylabel(r'$N^{-1} \log F$')
# ax41.set_ylabel(r"$F^{1/N}$")
ax41.set_ylabel(r"$F_{\mathrm{sp}}$")
ax41.set_xlabel('time steps')

ax41.text(0.7,0.63,r"$\sigma = 0.01$", fontweight="bold", size=tsize, transform = ax41.transAxes)
ax41.text(-0.1,1.05,"(h)", fontweight="bold", size=17, transform = ax41.transAxes)
ax41.set_yticks([0.8, 0.9, 0.97])

axin = inset_axes(ax41, width=1.8, height=0.8, loc=4, borderpad=0, bbox_to_anchor=(0.95, 0.18), bbox_transform=ax41.transAxes)

j = 0
for i,r in enumerate(rewardss2):
    if i in [1,4,5,7,8]: #[1,4,5,6,7,8]
        axin.plot(np.exp(r), color=rgbas[j])#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
        if i != aa: axin.plot(len(r)-1, np.exp(r[-1]), color=rgbas[j], marker = "o", markersize=4)#, linestyle=plot_styles[i])#, "o", color="tab:blue")#, plot_styles[i])#, label=r"$F={:.3f}$".format(rewards[i,-1]))
        j +=1
axin.plot(np.exp(rewardss2[aa]), color="red", linewidth=2.0)
axin.plot(len(rewardss2[aa])-1, np.exp(rewardss2[aa][-1]), color="red", marker = "o", markersize=4)
# plt.plot(rewardss[0], color="black")
axin.set_ylim(0.959,0.977)
axin.set_xlim(xmin=22)
axin.axhline(1-0.03, 0.0, 1.0, color="tab:gray", linestyle='--', linewidth=1.5)

# plt.tight_layout()
plt.savefig("cs3.png", dpi=300,bbox_inches='tight')
plt.savefig("cs3.pdf", dpi=200,bbox_inches='tight')
