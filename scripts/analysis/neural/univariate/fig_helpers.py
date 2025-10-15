import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import fig_constants

def label_panels_mosaic(fig, axes, xloc=0, yloc=1.0, size=fig_constants.BIGGER_SIZE):
    """
    Labels the panels in a mosaic plot.

    Parameters:
        - fig: The figure object.
        - axes: A dictionary of axes objects representing the panels.
        - xloc: The x-coordinate for the label position (default: 0).
        - yloc: The y-coordinate for the label position (default: 1.0).
        - size: The font size of the labels (default: constants.BIGGER_SIZE).
    """
    for key in axes.keys():
        # label physical distance to the left and up:
        ax = axes[key]
        trans = transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(xloc, yloc, key, transform=ax.transAxes + trans,
                fontsize=size, va='bottom')

def make_blank_panel(ax):
    """
    Makes a panel blank by turning off the axis and setting aspect ratio to 'auto'.

    Parameters:
        - ax: The axis object representing the panel.

    Returns:
        The modified axis object.
    """
    ax.axis('off')
    ax.set_aspect('auto')
    return ax

def add_time_window_annotation(axis, x, y, width, height, label, facecolor, alpha, **kwargs):
    """
    Adds a labeled rectangle patch to an axes object.
    """
    rect = patches.Rectangle((x, y), width, height, linewidth=0., facecolor=facecolor, alpha=alpha)
    axis.add_patch(rect)
    text_x = x + width / 2
    text_y = y + height / 2
    axis.text(text_x, text_y, label, ha='center', va='center', **kwargs)

def add_significance_bar(ax, x1, x2, y_level, p_value, text_offset=0.001):
    """
    Adds a significance bar with asterisks to a plot.

    ax: The axes object to plot on.
    x1, x2: The x-coordinates (categorical indices) for the bar.
    y_level: The y-position for the horizontal bar.
    p_value: The p-value to determine the number of asterisks.
    """
    # Determine the number of asterisks
    if p_value < 0.001:
        asterisks = '***'
    elif p_value < 0.01:
        asterisks = '**'
    elif p_value < 0.05:
        asterisks = '*'
    else:
        asterisks = 'ns' # Not significant

    # Plot the horizontal bar and whiskers
    line_x = [x1, x1, x2, x2]
    line_y = [y_level, y_level + text_offset, y_level + text_offset, y_level]
    ax.plot(line_x, line_y, lw=1.5, color='black')

    # Add the asterisks
    ax.text((x1 + x2) * 0.5, y_level + text_offset, asterisks,
            ha='center', va='bottom', color='black', fontsize=14)

def locatable_axes(ax, ticks, im):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    cb = plt.colorbar(im, cax=ax_cb, ticks=ticks)
    cb.outline.set_visible(False)
    cb.set_label('AUC', labelpad=-10, rotation=270)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)

def add_colorbar(fig, ax, im, ticks, label):
    cb = fig.colorbar(im, ax=ax, ticks=ticks, fraction=0.05, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(label, labelpad=-10, rotation=270)



def add_background_spans(ax, span_coords_list, x_limits, alphas=None, color='lightgrey', p_value=None):
    """
    Add background rectangles to an axis based on span coordinates and x-axis limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to add rectangles to.

    span_coords_list : list of tuple
        Each tuple is (x_start, x_end) in data coordinates.

    x_limits : tuple
        The (x_min, x_max) range of the data coordinates for the axis.

    alphas : list of float, optional
        List of alpha (transparency) values for each rectangle. Defaults to 0.5 for all.

    color : str
        Fill color for rectangles.
    """
    x_min, x_max = x_limits
    axis_width = x_max - x_min

    if alphas is None:
        alphas = [0.5] * len(span_coords_list)

    for (x_start, x_end), alpha in zip(span_coords_list, alphas):
        start_rel = (x_start - x_min) / axis_width
        width_rel = (x_end - x_start) / axis_width

        rect = patches.Rectangle(
            (start_rel, 0), width_rel, 1,
            facecolor=color,
            alpha=alpha,
            transform=ax.transAxes,
            zorder=0
        )
        ax.add_patch(rect)

        if p_value:
            if p_value < 0.001:
                asterisks = '***'
            elif p_value < 0.01:
                asterisks = '**'
            elif p_value < 0.05:
                asterisks = '*'
            else:
                asterisks = 'ns' # Not significant
            pos = (x_start + x_end)/2 # in the middle of rect
            ax.text(pos, min([min(line.get_ydata()) for line in ax.lines]), asterisks)

def label_bars(bar, axis, *argv):
    for i, arg in enumerate(argv):
        # height = bar[i].get_height()
        axis.annotate(
            arg,
            xy=(bar[i].get_x() + bar[i].get_width() / 2, 0),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom',
            color='white',
            rotation=90, zorder=100
        )

def add_time_window_annotation(axis, x, y_offset_pct, width, height_pct, label, facecolor, alpha, **kwargs):
    """Adds a labeled rectangle patch to an axes object."""
    ymin, ymax = axis.get_ylim()
    y_range = ymax - ymin

    rect_height = height_pct * y_range
    y_base = ymin + y_offset_pct * y_range

    # Add rectangle
    rect = patches.Rectangle((x, y_base), width, rect_height,
                             linewidth=0., facecolor=facecolor, alpha=alpha, zorder=10)
    axis.add_patch(rect)

    # Add centered text label
    text_x = x + width / 2
    text_y = y_base + rect_height / 2
    axis.text(text_x, text_y, label, ha='center', va='center', zorder=11, **kwargs)

def save_figure(figure, fignum, folder=''):
    """
    Saves the figure as an SVG file.

    Parameters:
        - figure: The figure object to be saved.
        - fignum: The figure number or name.
        - folder: The folder path to save the figure.
    """
    figure.savefig(folder + str(fignum) + '.svg', dpi=figure.dpi)