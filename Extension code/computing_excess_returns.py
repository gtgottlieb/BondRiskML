import pandas as pd

def compute_excess_returns_for_all_bonds(df,
                                         one_year_col_index=11,
                                         annual_yield_scale=100.0):
    """
    Compute 1-year excess returns for all bonds that have at least 12 months 
    to maturity. For an n-month bond (n >= 12):
    
    - Buy at date t (column index i => yield for n = i+1 months).
    - Sell at date t+12. After 1 year, the bond has (n - 12) months left:
       * If (n - 12) > 0, we get its price from the (n-12) column in the shifted DataFrame.
       * If (n - 12) = 0, we assume the bond matures at par = 1.0.
    - 1-year Holding-Period Return (HPR) = (SellPrice / BuyPrice) - 1.
    - Excess Return = HPR - (the 1-year yield at time t).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of yields (in percent) with Date as the index. 
        Column 0 => 1-month yield, column 1 => 2-month yield, ..., etc.
        Must have at least 13 columns if you want a 12-month yield in column 11.
    one_year_col_index : int
        The column index that corresponds to the 12-month yield. Default=11 
        (since 1-month is col=0, so 12-month is col=11).
    annual_yield_scale : float
        Factor to convert yields from percent to decimal. If yields are 2.50 => 2.50%,
        use 100.0. If your yields are already in decimals, set this to 1.0.

    Returns
    -------
    df_excess : pd.DataFrame
        A DataFrame with new columns 'ExRet_{n}m' for n >= 12, indexed by the same dates 
        as df but dropping the last 12 rows (because shift(-12) loses them).
    """
    # Shift the DataFrame by -12 rows so row t sees yields from row t+12
    df_future = df.shift(-12)

    # We'll skip the last 12 rows as they don't have a t+12
    valid_index = df.index[:-12]

    # Prepare an output DataFrame
    df_excess = pd.DataFrame(index=valid_index)

    # Helper: yield -> zero-coupon price with monthly compounding
    def yield_to_price(y_decimal, months):
        """
        For yield y_decimal (e.g., 0.025 for 2.5%),
        zero-coupon price = 1 / (1 + y_decimal/12)^months
        """
        return 1.0 / ((1.0 + (y_decimal/12.0)) ** months)

    ncols = df.shape[1]
    # each col i => maturity i+1 months

    # Loop over all columns, i => maturity = i+1
    for i in range(ncols):
        maturity = i + 1  # e.g. col=0 => 1-month bond, col=11 => 12-month bond
        if maturity < 12:
            # cannot hold a <12 month bond for one full year
            continue

        # yields for the n-month bond at time t (in decimal)
        yield_n_t = df.iloc[:, i] / annual_yield_scale
        price_n_t = yield_to_price(yield_n_t, maturity)

        # After 1 year, the bond has (maturity - 12) months left
        future_m = maturity - 12
        if future_m < 0:
            # shouldn't happen if maturity>=12, but just in case:
            continue

        if future_m == 0:
            # The bond matures exactly, so we assume final price = $1.0
            price_n_future = 1.0
        else:
            # otherwise we get the yields from the future DataFrame
            future_col = future_m - 1  # 0-based index for that column
            if future_col < 0 or future_col >= ncols:
                # skip if the future column doesn't exist
                continue
            yield_n_future = df_future.iloc[:, future_col] / annual_yield_scale
            price_n_future = yield_to_price(yield_n_future, future_m)

        # 1-year holding-period return (simple)
        hpr = (price_n_future / price_n_t) - 1.0

        # Subtract the 1-year yield from time t
        if one_year_col_index < 0 or one_year_col_index >= ncols:
            raise ValueError("Invalid one_year_col_index. Check your DataFrame columns.")
        yield_12_t = df.iloc[:, one_year_col_index] / annual_yield_scale
        ex_ret = hpr - yield_12_t

        # add to output
        col_name = f"ExRet_{maturity}m"
        df_excess[col_name] = ex_ret

    # restrict to valid_index
    df_excess = df_excess.loc[valid_index]
    return df_excess


