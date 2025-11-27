from numpy import ndarray
from pandas import Series, read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
from utils.dslabs_functions import get_variable_types, plot_bar_chart, HEIGHT
from os import makedirs


def read_data() -> DataFrame:
    """
    Reads the dataset and returns both the DataFrame and a file tag.
    """
    filename = "Combined_Flights_2022.csv"
    file_tag = "combined_flights"

    # Load CSV and treat empty strings as NaN
    data: DataFrame = read_csv(filename, na_values="")
    return data, file_tag


def derive_date_variables(df: DataFrame, date_vars: list[str]) -> DataFrame:
    """
    Adds derived date components (quarter, month, day) for each date variable.
    """
    for date in date_vars:
        df[date + "_quarter"] = df[date].dt.quarter
        df[date + "_month"] = df[date].dt.month
        df[date + "_day"] = df[date].dt.day
    return df


def analyse_date_granularity(data: DataFrame, var: str, levels: list[str]) -> ndarray:
    """
    Plots bar charts for different granularities (e.g., quarter, month, day)
    of a given date variable.
    """
    rows: int = len(levels)
    cols: int = 1

    fig: Figure
    axs: ndarray
    fig, axs = subplots(rows, cols, figsize=(HEIGHT * 2, rows * HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {var}")

    for i in range(rows):
        counts: Series[int] = data[f"{var}_{levels[i]}"].value_counts()

        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, 0],
            title=levels[i],
            xlabel=levels[i],
            ylabel="nr records",
            percentage=False,
        )

    return axs


def flights_date_granularity(save_plot: bool = False, show_plot: bool = True):
    """
    Runs granularity analysis (quarter/month/day) for all date variables.
    """
    data, file_tag = read_data()
    variables_types: dict[str, list] = get_variable_types(data)

    # Extend dataset with derived date components
    data_ext: DataFrame = derive_date_variables(data, variables_types["date"])

    if save_plot:
        makedirs("images/granularity", exist_ok=True)

    for v_date in variables_types["date"]:
        analyse_date_granularity(data, v_date, ["quarter", "month", "day"])
        if save_plot: savefig(f"images/granularity/{file_tag}_{v_date}_bar_chart.png", bbox_inches="tight")
        if show_plot: show()


def analyse_property_granularity(
    data: DataFrame, property: str, vars: list[str]
) -> ndarray:
    """
    Plots bar charts for a set of related variables (e.g., location fields)
    to see how values are distributed.
    """
    rows: int = len(vars)
    cols: int = 1

    fig: Figure
    axs: ndarray
    fig, axs = subplots(rows, cols, figsize=(HEIGHT * 4, rows * HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {property}")

    for i in range(rows):
        counts: Series[int] = data[vars[i]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, 0],
            title=vars[i],
            xlabel=vars[i],
            ylabel="nr records",
            percentage=False,
        )

    return axs


def origin_granularity(save_plot: bool = False, show_plot: bool = True):
    """
    Plots granularity for origin-related location variables.
    """
    data, file_tag = read_data()
    analyse_property_granularity(data,"location", ["OriginStateName", "OriginCityName"])

    if save_plot: savefig(f"images/granularity/{file_tag}_origin_granularity4.png")
    if show_plot: show()


if __name__ == "__main__":
    #flights_date_granularity(save_plot=True, show_plot=False)
    origin_granularity(save_plot=True, show_plot=False)