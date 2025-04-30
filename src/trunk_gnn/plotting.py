import matplotlib.pyplot as plt
import scienceplots

colors = {
    "red": "#ff3f17",
    "blue": "#0c5da5",
    "yellow": "#ff9500",
    "green": "#16a085",
    "purple": "#8e44ad",
    "dark_blue": "#34495e",
}
plt.style.use(["science","ieee"])
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler('color', [
        colors["blue"], colors["red"], colors["yellow"], colors["green"], colors["purple"], colors["dark_blue"]
    ]) + plt.cycler('linestyle', ['-' for _ in range(6)]),
})
plt.rcParams.update({
    "lines.linewidth": 1,
})

# small legends
plt.rcParams.update({
    "legend.fontsize": 6,
    "legend.frameon": False,
    "legend.loc": "upper right",
})