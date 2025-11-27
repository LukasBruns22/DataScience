from utils.dslabs_functions import get_variable_types, HEIGHT
from seaborn import heatmap
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show

def read_data() -> DataFrame:
    """
    Reads the dataset from disk and returns both the DataFrame and a file tag.
    """
    filename = "Combined_Flights_2022.csv"
    file_tag = "combined_flights"

    # Load the CSV file (empty strings treated as NaN)
    data: DataFrame = read_csv(filename, na_values="")
    return data, file_tag


def correlation_analysis(show_plot: bool = True, save_plot: bool = False):
    """
    Creates a full correlation heatmap for all numeric variables in the dataset.
    """
    # Load data
    data, file_tag = read_data()

    # Get variable types and extract numeric columns only
    variables_types = get_variable_types(data)
    numeric = variables_types["numeric"]

    # Compute the absolute correlation matrix
    corr_mtx = data[numeric].corr().abs()

    # Adjust figure size based on number of variables
    n = len(numeric)
    fig = figure(figsize=(max(8, n * 0.4), max(6, n * 0.4)))

    # Draw the heatmap WITHOUT any mask (show full matrix)
    ax = heatmap(
        corr_mtx,
        xticklabels=numeric,
        yticklabels=numeric,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.7},
    )

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Fit layout to avoid clipping labels
    fig.tight_layout()

    # Save the figure if requested
    if save_plot:
        savefig(
            f"images/correlation/{file_tag}_correlation_analysis.png",
            bbox_inches="tight",
        )

    # Show the figure if requested
    if show_plot:
        show()


if __name__ == "__main__":
    correlation_analysis(show_plot=False, save_plot=True)