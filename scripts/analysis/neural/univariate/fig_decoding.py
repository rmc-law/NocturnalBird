import os
import os.path as op
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
from plot_decoding import *
import fig_constants
import fig_helpers as fh
import config
# mpl.rc_file('fake_diamond.rc')


# 1. CONFIGURATION
classifier   = 'logistic'
window       = 'single'
micro_ave    = True
data_type    = 'ROI'
roi          = 'anteriortemporal-lh'
to_plot      = 'concreteness_xcond'
analyses     = ['concreteness_trainWord_testSub','concreteness_trainWord_testPri']
subjects     = [f"sub-{s}" for s in config.subject_ids]
figures_dir = op.join(config.project_repo, f'figures/paper/')
os.makedirs(figures_dir, exist_ok=True)

# 2. LOAD DECODING SCORES
scores_diag = [read_decoding_scores(subjects, a, classifier, data_type, window, roi, timegen=False, micro_ave=micro_ave)
               for a in analyses]
scores_tgm  = [read_decoding_scores(subjects, a, classifier, data_type, window, roi, timegen=True,  micro_ave=micro_ave)
               for a in analyses]

sfreq = int(scores_diag[0].shape[1] / 0.8)

# 3. SET UP FIGURE 
mosaic = [
    ['A','B','.'],
    ['A','B','E'],
    ['C','D','E'],
    ['C','D','.']]
fig, ax_dict = plt.subplot_mosaic(
    mosaic,  # Specify the layout of subplots using the mosaic parameter
    figsize=(fig_constants.FIG_WIDTH, 4.),  # Set the size of the figure in inches
    dpi=300,  # Set the resolution of the figure in dots per inch
    constrained_layout=True,  # Enable constrained layout for automatic adjustment
    gridspec_kw={
        'height_ratios': [1, 1, 1, 1], # Set the relative heights of the rows
        'width_ratios': [1.5, 1., 0.5], # Set the relative widths of the columns
        'wspace': 0.001,
        'hspace': 0.001}
)
# 4. PANEL A: Time series for test‑on‑subsective
plot_scores(ax_dict['A'], scores_diag[0], analysis=analyses[0], chance=0.5)
good_clusters, cluster_pvals = permutation_tests(scores_diag[0], timegen=False, against_chance=True)
plot_clusters(good_clusters, scores=scores_diag[0], analysis=analyses[0], ax=ax_dict['A'], cluster_pvals=cluster_pvals)
ax_dict['A'].set_title(r'train on word $\rightarrow$ test on subsective')
ax_dict['A'].set_yticks([0.48, 0.5, 0.52, 0.54])
ax_dict['A'].set_xlim([0.0, 0.8])
ax_dict['A'].axvspan(0.3, 0.5, alpha=0.50, color='lightgrey', zorder=0)
ax_dict['A'].axvspan(0.5, 0.8, alpha=0.75, color='lightgrey', zorder=0)
ax_dict['A'].text(0.4, 0.485, 'early', ha='center', va='center')
ax_dict['A'].text(0.65, 0.485, 'late', ha='center', va='center')

# # 5. PANEL C: Time series for test‑on‑privative
plot_scores(ax_dict['C'], scores_diag[1], analysis=analyses[1], chance=0.5)
ax_dict['C'].set_yticks([0.48, 0.5, 0.52, 0.54])
ax_dict['C'].set_xlim([0.0, 0.8])
ax_dict['C'].set_title(r'train on word $\rightarrow$ test on privative')
ax_dict['C'].axvspan(0.3, 0.5, alpha=0.50, color='lightgrey', zorder=0)
ax_dict['C'].axvspan(0.5, 0.8, alpha=0.75, color='lightgrey', zorder=0)
ax_dict['C'].text(0.4, 0.472, 'early', ha='center', va='center')
ax_dict['C'].text(0.65, 0.472, 'late', ha='center', va='center')

# # 6. PANEL B: TGM for subsective
group_avg = np.mean(scores_tgm[0], axis=0)
vmax = round(np.max(group_avg), 2)
imB = ax_dict['B'].imshow(group_avg, origin='lower',
                 extent=[0,0.8,0,0.8], vmin=0.5, vmax=vmax,
                 cmap='Purples')
good_clusters, cluster_pv = permutation_tests(scores_tgm[0], timegen=True, against_chance=True, t_threshold=None)
extent = np.array([0., 0.8])
times = np.linspace(extent[0], extent[1], scores_tgm[0][0].shape[0])
X, Y = np.meshgrid(times, times)
if cluster_pv:
    cluster_stats = []
    for i, (cluster, pval) in enumerate(zip(good_clusters, cluster_pv)):
        # Plot contours
        if pval < 0.05:
            ax_dict['B'].contour(X, Y, cluster, colors=['black'], linewidths=1)
        else:
            ax_dict['B'].contour(X, Y, cluster, colors=['grey'], linewidths=1)
fh.locatable_axes(ax_dict['B'], [0.5,vmax], imB)
ax_dict['B'].plot([0,0.8],[0,0.8],'--',color='grey')
ax_dict['B'].set_xlabel('Test time (s)')
ax_dict['B'].set_ylabel('Train time (s)')

# # 7. PANEL D: TGM for privative
group_avg = np.mean(scores_tgm[1], axis=0)
imD = ax_dict['D'].imshow(group_avg, origin='lower',
                 extent=[0,0.8,0,0.8], vmin=0.5, vmax=vmax,
                 cmap='Oranges')
fh.locatable_axes(ax_dict['D'], [0.5,vmax], imD)
ax_dict['D'].plot([0,0.8],[0,0.8],'--',color='grey')
ax_dict['D'].set_xlabel('Test time (s)')
ax_dict['D'].set_ylabel('Train time (s)')

# 8. PANEL E: Bar plot (early vs late)
scores_timewindowed = pd.read_csv('/imaging/hauk/rl05/fake_diamond/figures/decoding/concreteness_xcond/diagonal/logistic/ROI/single/micro_ave/scores_timewindow-averaged_anteriortemporal-lh_100Hz.csv')
color_palette = [plt.get_cmap(color_scheme['concreteness_trainWord_testSub'])(0.7),
                plt.get_cmap(color_scheme['concreteness_trainWord_testPri'])(0.7)]
sns.barplot(
    data=scores_timewindowed, x='timewindow', y='score', hue='test_on',
    errorbar='se', 
    palette=color_palette,
    ax=ax_dict['E']
)
ax_dict['E'].set(ylim=(0.45, 0.55), xlabel='Time window', ylabel='AUC')
leg = ax_dict['E'].legend(
    handlelength=0, handletextpad=0,
    loc='upper left', bbox_to_anchor=(0., 1.3)) 
leg.get_frame().set_facecolor('none')
leg.get_frame().set_edgecolor('none')
for text, color in zip(leg.get_texts(), color_palette):
    text.set_color(color)
    text.set_fontweight('bold')

early_block = patches.Rectangle((0, 0), 0.5, 1,facecolor='lightgrey',alpha=0.5,transform=ax_dict['E'].transAxes,zorder=0) # zorder=0 places the patch behind plot elements
late_block = patches.Rectangle((0.5, 0), 0.5, 1, facecolor='lightgrey', alpha=0.75, transform=ax_dict['E'].transAxes,zorder=0)
ax_dict['E'].add_patch(early_block)
ax_dict['E'].add_patch(late_block)

fh.label_panels_mosaic(fig, ax_dict, size = 14)



# 9. FINALIZE & SAVE
out_fname = op.join(figures_dir, 'figure6.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")
