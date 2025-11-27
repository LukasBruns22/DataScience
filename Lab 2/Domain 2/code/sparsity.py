from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
from utils.dslabs_functions import HEIGHT, plot_multi_scatters_chart


def read_data() -> DataFrame:
    """
    Reads the dataset and returns both the DataFrame and a file tag.
    """
    filename = "Combined_Flights_2022.csv"
    file_tag = "combined_flights"

    # Load CSV and treat empty strings as NaN
    data: DataFrame = read_csv(filename, na_values="")
    return data, file_tag


def sparsity_study(
    sample_size: int | None = 100_000,
    random_state: int = 42,
) -> None:
    """
    Creates a pairwise scatter-plot matrix to study variable sparsity.
    Uses sampling to keep the plot manageable on large datasets.
    """
    data, file_tag = read_data()

    # Remove rows with missing values
    data = data.dropna()

    # Sample the dataset if it exceeds the allowed size
    if sample_size is not None and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_state)

    # Get the list of variables to compare
    vars: list[str] = data.columns.to_list()
    if not vars:
        print("Sparsity study: there are no variables.")
        return

    n: int = len(vars)

    # Create an n × n grid of scatter plots
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    fig.suptitle(f"Sparsity study for {file_tag} (n={len(data):,} sampled rows)")

    # Fill each cell with either a scatter plot or leave diagonal empty
    for i, var1 in enumerate(vars):
        for j, var2 in enumerate(vars):
            if i == j:
                # Leave diagonal plots blank
                axs[i, j].axis("off")
            else:
                plot_multi_scatters_chart(data, var1, var2, ax=axs[i, j])

    # Save the final matrix plot
    savefig(f"images/sparsity/{file_tag}_sparsity_study.png", bbox_inches="tight")


if __name__ == "__main__":
    sparsity_study()