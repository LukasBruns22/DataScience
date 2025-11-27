from matplotlib.pyplot import savefig, show, subplots
from scipy.stats import norm, expon, lognorm
from matplotlib.figure import Figure
from utils.dslabs_functions import (
    get_variable_types,
    define_grid,
    HEIGHT,
    plot_multiline_chart,
    plot_bar_chart,
    plot_multibar_chart,
)
from pandas import read_csv, DataFrame, Series
from matplotlib.axes import Axes
from numpy import ndarray, log, linspace


def read_data() -> DataFrame:
    """
    Reads the dataset and returns both the DataFrame and a file tag.
    """
    filename = "Combined_Flights_2022.csv"
    file_tag = "combined_flights"

    # Load CSV and treat empty strings as NaN
    data: DataFrame = read_csv(filename, na_values="")
    return data, file_tag


def numeric_variables_boxplot(show_plot: bool = True, save_plot: bool = False):
    """
    Plots boxplots for all numeric variables using a grid layout.
    """
    data, file_tag = read_data()
    variables_types = get_variable_types(data)
    numeric = variables_types["numeric"]

    if numeric:
        # Determine grid size based on number of numeric variables
        rows, cols = define_grid(len(numeric))

        fig: Figure
        axs: ndarray
        fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
        i, j = 0, 0
        for n in range(len(numeric)):
            axs[i, j].set_title(f"Boxplot for {numeric[n]}")
            axs[i, j].boxplot(data[numeric[n]].dropna().values)

            # Move to next position in grid
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        if save_plot: savefig(f"images/distribution/{file_tag}_numerical_sub_boxplot.png")
        if show_plot: show()
        fig.clf()
    else:
        print("There are no numeric variables.")


def compute_known_distributions(raw_data: ndarray) -> tuple[list, dict]:
    """
    Fits known continuous distributions to the data and returns
    a compact x-axis and the corresponding PDFs.
    """
    distributions = {}

    # Define a compact x-axis for plotting PDFs
    x_min, x_max = raw_data.min(), raw_data.max()
    x_axis = linspace(x_min, x_max, 1000)

    # Normal distribution
    mean, sigma = norm.fit(raw_data)
    distributions[f"Normal({mean:.1f},{sigma:.2f})"] = norm.pdf(x_axis, mean, sigma)

    # Exponential distribution
    try:
        loc, scale = expon.fit(raw_data)
        distributions[f"Exp({1 / scale:.2f})"] = expon.pdf(x_axis, loc, scale)
    except Exception:
        pass

    # Lognormal distribution (only if all values are positive)
    if x_min > 0:
        try:
            s, loc, scale = lognorm.fit(raw_data)
            distributions[f"LogNor({log(scale):.1f},{s:.2f})"] = lognorm.pdf(
                x_axis, s, loc, scale
            )
        except Exception:
            pass

    return x_axis.tolist(), distributions


def histogram_with_distributions(ax: Axes, series: Series, var: str):
    """
    Plots a histogram of a numeric variable and overlays fitted distributions.
    """
    # Use raw numpy array for performance
    raw_values = series.dropna().values

    if len(raw_values) == 0:
        return

    # Histogram of the original data (density=True for PDF scale)
    ax.hist(raw_values, bins=20, density=True, alpha=0.5)

    # Compute fitted distributions on a reduced x-axis
    x_axis, distributions = compute_known_distributions(raw_values)

    # Overlay fitted distributions
    plot_multiline_chart(
        x_axis,
        distributions,
        ax=ax,
        title=f"Best fit for {var}",
        xlabel=var,
        ylabel="Density",
    )


def probability_density(show_plot: bool = True, save_plot: bool = False):
    """
    Plots probability density (histogram + fitted distributions)
    for all numeric variables.
    """
    data, file_tag = read_data()
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]

    if numeric:
        rows, cols = define_grid(len(numeric))

        # Ensure axs is always 2D
        fig, axs = subplots(
            rows, cols, figsize=(cols * 5, rows * 5), squeeze=False
        )

        i, j = 0, 0
        for n in range(len(numeric)):
            print(f"Processing {numeric[n]}...")
            histogram_with_distributions(axs[i, j], data[numeric[n]], numeric[n])

            # Move across the grid
            if (n + 1) % cols == 0:
                i += 1
                j = 0
            else:
                j += 1

        if save_plot: savefig(f"images/distribution/{file_tag}_prabability_density.png")
        if show_plot: show()
        fig.clf()
    else:
        print("There are no numeric variables.")


def binary_variables_histogram(show_plot: bool = True, save_plot: bool = False):
    """
    Plots histograms for all binary variables (False/True counts).
    """
    data, file_tag = read_data()
    variables_types: dict[str, list] = get_variable_types(data)
    binary: list[str] = variables_types["binary"]

    if binary:
        rows, cols = define_grid(len(binary))
        fig: Figure
        axs: ndarray
        fig, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )

        i, j = 0, 0
        for n in range(len(binary)):
            var = binary[n]
            # Convert to int and count occurrences of 0/1
            vals = data[var].dropna().astype(int)
            counts = vals.value_counts().sort_index()
            x = counts.index.to_list()
            y = counts.to_list()
            x_labels = ["False", "True"]

            plot_bar_chart(
                x_labels,
                y,
                ax=axs[i, j],
                title=f"Histogram for {var}",
                xlabel=var,
                ylabel="nr records",
                percentage=False,
            )

            # Move in grid
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        if save_plot: savefig(f"images/distribution/{file_tag}_binary_histogram.png")
        if show_plot: show()
        fig.clf()
    else:
        print("There are no binary variables.")


def symbolic_values_histogram(show_plot: bool = True, save_plot: bool = False):
    """
    Plots histograms for all symbolic (categorical) and binary variables.
    """
    data, file_tag = read_data()
    variables_types: dict[str, list] = get_variable_types(data)
    symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]

    if not symbolic:
        print("There are no symbolic variables.")
        return

    rows: int = len(symbolic)
    cols: int = 1

    # One chart per row, wide figure
    fig, axs = subplots(rows, cols, figsize=(HEIGHT * 20, rows * HEIGHT), squeeze=False)

    for i, var in enumerate(symbolic):
        counts: Series = data[var].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, 0],
            title=f"Histogram for {var}",
            xlabel=var,
            ylabel="nr records",
            percentage=False,
        )

    if save_plot: savefig(f"images/distribution/{file_tag}_symbolic_histogram.png", bbox_inches="tight")
    if show_plot: show()


def class_distribution(show_plot: bool = True, save_plot: bool = False):
    """
    Plots the distribution of the target class 'Cancelled'.
    """
    data, file_tag = read_data()
    target = "Cancelled"
    values: Series = data[target].value_counts()

    fig, ax = subplots(figsize=(HEIGHT * 4, HEIGHT * 2))
    plot_bar_chart(
        values.index.to_list(),
        values.to_list(),
        ax=ax,
        title=f"Target distribution (target={target})",
        ylabel="nr records",
        percentage=False,
    )

    if save_plot:savefig(f"images/distribution/{file_tag}_class_distribution_histogram.png")
    if show_plot: show()
    fig.clf()


NR_STDEV: int = 2
IQR_FACTOR: float = 1.5


def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    """
    Computes upper and lower outlier thresholds for a variable using
    either standard deviation or IQR.
    """
    top: float = 0
    bottom: float = 0

    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr: float = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom


def count_outliers(
    data: DataFrame,
    numeric: list[str],
    nrstdev: int = NR_STDEV,
    iqrfactor: float = IQR_FACTOR,
) -> dict:
    """
    Counts outliers per numeric variable using both std-based and IQR-based rules.
    """
    outliers_iqr: list = []
    outliers_stdev: list = []

    # Summary statistics for numeric variables
    summary5: DataFrame = data[numeric].describe()

    for var in numeric:
        # Std-based thresholds
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev
        )
        outliers_stdev.append(
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        )

        # IQR-based thresholds
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor
        )
        outliers_iqr.append(
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        )

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}


def standard_outliers(show_plot: bool = True, save_plot: bool = False):
    """
    Plots the number of outliers per numeric variable (IQR vs StdDev).
    """
    data, file_tag = read_data()
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]

    if not numeric:
        print("There are no numeric variables.")
        return

    outliers = count_outliers(data, numeric)
    rows, cols = define_grid(len(numeric))

    fig, axs = subplots(
        rows,
        cols,
        figsize=(cols * HEIGHT * 2, rows * HEIGHT * 2),
        squeeze=False,
    )

    for idx, var in enumerate(numeric):
        i, j = divmod(idx, cols)
        xvalues = [var]
        series = {
            "iqr": [outliers["iqr"][idx]],
            "stdev": [outliers["stdev"][idx]],
        }

        plot_multibar_chart(
            xvalues,
            series,
            ax=axs[i, j],
            title=f"Nr of standard outliers ({var})",
            xlabel="variables",
            ylabel="nr outliers",
            percentage=False,
        )

    if save_plot: savefig(f"images/distribution/{file_tag}_standard_outliers_histogram.png", bbox_inches="tight")
    if show_plot: show()
    fig.clf()


if __name__ == "__main__":
    #numeric_variables_boxplot(show_plot=False, save_plot=True)
    #numeric_variables_histogram(show_plot=False, save_plot=True)
    probability_density(show_plot=False, save_plot=True)
    #binary_variables_histogram(show_plot=False, save_plot=True)
    #symbolic_values_histogram(show_plot=False, save_plot=True)
    #class_distribution(show_plot=False, save_plot=True)
    #standard_outliers(show_plot=False, save_plot=True)