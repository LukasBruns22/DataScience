from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
from utils.dslabs_functions import plot_bar_chart, get_variable_types
import matplotlib.pyplot as plt


def read_data() -> DataFrame:
    """
    Reads the dataset and returns both the DataFrame and a file tag.
    """
    filename = "Combined_Flights_2022.csv"
    file_tag = "combined_flights"

    # Load CSV and treat empty strings as NaN
    data: DataFrame = read_csv(filename, na_values="")
    return data, file_tag


def records_vs_variables(show_plot: bool = True, save_plot: bool = False):
    """
    Plots the number of records vs number of variables in the dataset.
    """
    data, file_tag = read_data()

    # Basic count of rows and columns
    values = {
        "nr records": data.shape[0],
        "nr variables": data.shape[1]
    }

    figure(figsize=(12, 4))
    plot_bar_chart(
        list(values.keys()),
        list(values.values()),
        title="Nr of records vs nr variables"
    )

    # Use log scale to make large differences easier to see
    ax = plt.gca()
    ax.set_yscale("log")

    # Save or show plot
    if save_plot: savefig(f"images/dimensionality/{file_tag}_records_vs_variables.png", bbox_inches="tight")
    if show_plot: show()


def missing_values(show_plot: bool = True, save_plot: bool = False):
    """
    Plots the number of missing values for each variable with at least one missing entry.
    """
    data, file_tag = read_data()
    mv = {}

    # Count missing values per column
    for var in data.columns:
        nr = data[var].isna().sum()
        if nr > 0:
            mv[var] = nr

    figure(figsize=(12, 4))
    plot_bar_chart(
        list(mv.keys()),
        list(mv.values()),
        title="Nr of missing values per variable",
        xlabel="variables",
        ylabel="nr missing values"
    )

    # Save or show plot
    if save_plot: savefig(f"images/dimensionality/{file_tag}_missing_values.png", bbox_inches="tight")
    if show_plot: show()


def variable_types(show_plot: bool = True, save_plot: bool = False):
    """
    Plots how many variables belong to each detected data type.
    """
    data, file_tag = read_data()

    # Detect variable types using helper function
    variable_types = get_variable_types(data)

    # Count how many variables each type contains
    counts = {tp: len(cols) for tp, cols in variable_types.items()}

    figure(figsize=(12, 4))
    plot_bar_chart(
        list(counts.keys()),
        list(counts.values()),
        title="Nr of variables per type"
    )

    # Save or show plot
    if save_plot: savefig(f"images/dimensionality/{file_tag}_variable_types.png", bbox_inches="tight")
    if show_plot: show()


if __name__ == "__main__":
    # Generate all plots (saved but not displayed)
    records_vs_variables(save_plot=True, show_plot=False)
    missing_values(save_plot=True, show_plot=False)
    variable_types(save_plot=True, show_plot=False)