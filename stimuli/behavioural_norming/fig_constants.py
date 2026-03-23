from matplotlib import pyplot as plt, rcParams

FIG_WIDTH = 7.5  # width of figure in inches
BASE_FONT_SIZE = 6
RATIO = FIG_WIDTH / BASE_FONT_SIZE

TINY_SIZE = 6
SMALL_SIZE = TINY_SIZE + 2
MEDIUM_SIZE = SMALL_SIZE + 2
BIGGER_SIZE = MEDIUM_SIZE + 2

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE, titleweight='bold')  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False