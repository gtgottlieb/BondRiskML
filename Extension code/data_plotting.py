import matplotlib.pyplot as plt
import pandas as pd

yields_data = pd.read_excel("data-folder/Aligned_Yields.xlsx", index_col=0, parse_dates=True)

def plot_yield_curve_for_maturity(yields_data, maturity):
    """
    Plots the yield curve for a specified maturity.

    Parameters:
    - yields_data: DataFrame containing yield data with dates as index and maturities as columns.
    - maturity: Integer specifying the maturity in months to plot.
    """
    plt.figure()
    plt.plot(yields_data.index, yields_data.iloc[:, maturity], label=f"{maturity}-month maturity")
    plt.xlabel("Date")
    plt.ylabel("Yield")
    plt.title(f"Yield Curve for {maturity}-Month Maturity")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_yield_curve_for_date(yields_data, date):
    """
    Plots the yield curve for a specified date.

    Parameters:
    - yields_data: DataFrame containing yield data with dates as index and maturities as columns.
    - date: String or datetime specifying the date to plot the yield curve.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    if date not in yields_data.index:
        raise ValueError(f"Date {date} not found in the data.")
    
    plt.figure()
    plt.plot(yields_data.columns, yields_data.loc[date], label=f"Yield Curve on {date.date()}")
    plt.xlabel("Maturity (Months)")
    plt.ylabel("Yield")
    plt.title(f"Yield Curve on {date.date()}")
    plt.legend()

    # Set x-axis ticks at every 12 months
    tick_positions = range(0, len(yields_data.columns), 12)
    tick_labels = yields_data.columns[tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.grid(False)
    plt.show()
