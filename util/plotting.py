import numpy as np
import matplotlib.pyplot as plt
from mpanze_scripts.util import allen_utils

new_rc_params = { "svg.fonttype": 'none' }
plt.rcParams.update(new_rc_params)

def overlay_allen(ax, show_bregma=True, remove_axis=True, line_color="k-", line_width=0.5, line_alpha=1, res=(512,512)):
    masks, area_names, edges, mask_total, bregma = allen_utils.load_allen(res)

    for i in range(len(area_names)-14):
        ax.plot(edges[i,0], edges[i,1], line_color, linewidth=line_width, alpha=line_alpha)
    
    if show_bregma==True:
        ax.plot(bregma[0], bregma[1], 'r.', label="bregma")
    
    if remove_axis==True:
        ax.set_axis_off()
    
    return ax

def plot_mean_std(ax, t, X, color='k', alpha=0.1):
    m = np.nanmean(X, axis=0)
    s = np.nanstd(X, axis=0)
    ax.plot(t, m, color=color)
    ax.fill_between(t, m-s, m+s, color=color, alpha=alpha)
    return ax