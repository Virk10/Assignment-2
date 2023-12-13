import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def manipulate_data(filename):
    """
    Read and manipulate data from the provided file.

    Parameters:
    - filename (str): The path to the data file.

    Returns:
    - df_t (pd.DataFrame): Transposed DataFrame.
    - df (pd.DataFrame): Original DataFrame.
    """
    df = pd.read_excel(filename, skiprows=[0, 1, 2])
    df = df.drop(columns=["Indicator Code", "Country Code"])
    df_t = pd.DataFrame.transpose(df)
    df_t.columns = df_t.iloc[0]
    return df_t, df


def custom_df_per_indicator(df_global, indicator, countries):
    """
    Create a DataFrame based on the specified indicator and countries.

    Parameters:
    - df_global (pd.DataFrame): Original DataFrame.
    - indicator (str): The indicator for DataFrame filtering.
    - countries (list): List of countries to include.

    Returns:
    - df (pd.DataFrame): Custom DataFrame.
    """
    df = df_global[df_global['Indicator Name'] == indicator]
    df = df[df['Country Name'].isin(countries)]
    df.index = df['Country Name']
    df.drop(columns=['Country Name', 'Indicator Name'], inplace=True)
    col_ints = np.arange(1960, 2000, 1).tolist() + [2022]
    col_str = [str(x) for x in col_ints]
    df = df.drop(columns=col_str)
    return df


def custom_df_per_indicator_tr(df_global, indicator, countries):
    """
    Create a transposed DataFrame based on the specified indicator and countries.

    Parameters:
    - df_global (pd.DataFrame): Original DataFrame.
    - indicator (str): The indicator for DataFrame filtering.
    - countries (list): List of countries to include.

    Returns:
    - df_t (pd.DataFrame): Transposed custom DataFrame.
    """
    df = custom_df_per_indicator(df_global, indicator, countries)
    df_t = pd.DataFrame.transpose(df)
    df_t.columns = df_t.iloc[0]
    df_t = df_t.iloc[1:].astype(float)
    return df_t


def explore_data(data_years):
    """
    Explore and print summary statistics of the DataFrame.

    Parameters:
    - data_years (pd.DataFrame): DataFrame for exploration.
    """
    indicator = "Forest area (% of land area)"
    countries = ["Australia", "Bangladesh", "India", "Colombia", "Germany", "United States", "Brazil"]
    
    new_df = custom_df_per_indicator_tr(data_years, indicator, countries)
    print("DataFrame Description:")
    print(new_df.describe())
    print("\nMean of the DataFrame:")
    print(new_df.mean().mean())
    print("\nSum of the DataFrame:")
    print(new_df.sum().sum())


def explore_corelation(data_years, indicator1, indicator2, countries):
    """
    Explore correlation between two indicators for specified countries.

    Parameters:
    - data_years (pd.DataFrame): Original DataFrame.
    - indicator1 (str): First indicator for comparison.
    - indicator2 (str): Second indicator for comparison.
    - countries (list): List of countries for analysis.

    Returns:
    - df_cov (pd.DataFrame): Covariance matrix.
    - df_std_cov (pd.DataFrame): Standardized covariance matrix.
    - df_corr (pd.DataFrame): Correlation matrix.
    """
    df1 = custom_df_per_indicator(data_years, indicator1, countries)
    df2 = custom_df_per_indicator(data_years, indicator2, countries)
    df_merged = df1.merge(df2, on="Country Name")
    df_cov = df_merged.cov()
    df_std_cov = df_cov / df_merged.std()
    df_corr = df_merged.corr(method="kendall")

    print(f"Covariance Matrix between {indicator1} and {indicator2}:")
    print(df_cov)
    
    print(f"\nStandardized Covariance Matrix between {indicator1} and {indicator2} (divided by standard deviation):")
    print(df_std_cov)

    print(f"\nCorrelation Matrix between {indicator1} and {indicator2} (Kendall method):")
    print(df_corr)

    return df_cov, df_std_cov, df_corr


def plot_indicator_comparison(data_years, countries, indicator_name):
    """
    Plot comparison of an indicator for selected countries.

    Parameters:
    - data_years (pd.DataFrame): Original DataFrame.
    - countries (list): List of countries for comparison.
    - indicator_name (str): Indicator for comparison.
    """
    selected_countries_data = data_years[(data_years['Country Name'].isin(countries)) & (data_years['Indicator Name'] == indicator_name)]

    if not selected_countries_data.empty:
        years = selected_countries_data.columns[4:]
        aggregated_data = selected_countries_data.groupby('Country Name')[years].sum().reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 8))

        for country in countries:
            ax1.plot(years, aggregated_data[aggregated_data['Country Name'] == country].iloc[0, 1:].astype(float),
                     marker='o', label=f'{country} - {indicator_name}')

        ax1.set_xlabel('Year')
        ax1.set_ylabel(indicator_name)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f'{indicator_name} Comparison for Selected Countries')
        plt.show()
    else:
        print(f"No data available for {indicator_name} for the selected countries.")


def main():
    filename = "API_19_DS2_en_excel_v2_6224517.xls"
    countries_to_compare = ["Australia", "Bangladesh", "India", "Colombia", "Germany", "United States", "Brazil"]
    indicator_to_compare1 = 'Forest area (% of land area)'
    indicator_to_compare2 = 'Arable land (% of land area)'
    df_t, df = manipulate_data(filename)
    
    print("Transposed DataFrame:")
    print(df_t)

    explore_corelation(df, indicator_to_compare1, indicator_to_compare2, countries_to_compare)
    
    explore_data(df)
    plot_indicator_comparison(df, countries_to_compare, indicator_to_compare1)
    plot_indicator_comparison(df, countries_to_compare, indicator_to_compare2)


if __name__ == "__main__":
    main()
