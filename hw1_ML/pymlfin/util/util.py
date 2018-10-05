'''
utility module for loading data from various data sources
'''
import pandas as pd
import os.path
import numpy as np
import sklearn.decomposition

def get_eps_data(ticker, file_name='CRSP_Compustat_Merged_Sample_Q_Jan1980_Dec312016.csv', path='data/finratio'):

    df_crsp_sel = pd.read_csv(os.path.join(path, file_name),
                              date_parser=lambda dt:
                              pd.to_datetime(dt, format='%Y/%m/%d'),
                              parse_dates=['datadate'],
                              index_col=[0, 1])

    df_hon = df_crsp_sel[df_crsp_sel['tic'] == ticker]  # XOM, INTC, MSFT, HAL
    df_hon_sel = df_hon.loc[:,
                 ('txtq', 'piq', 'invtq', 'rectq', 'saleq', 'cogsq', 'xsgaq', 'prccq', 'epspxq', 'capxy', 'cshprq',)]

    # data cleaning: fill missing data with mean
    df_hon_sel = df_hon_sel.fillna(df_hon_sel.mean())

    # gross margin calculation
    df_hon_sel['gm'] = df_hon_sel['saleq'] - df_hon_sel['cogsq']  # GM = Sales - COGS
    # effective tax rate calculation
    df_hon_sel['etr'] = df_hon_sel['txtq'] / df_hon_sel['piq']  # ETR = Income Taxes / Pretax Income

    # Variables INV, AR, CAPX, GM, and SA are scaled by the weighted average number
    # of common shares used in calculating the basic quarterly EPS

    df_hon_sel['gm'] = df_hon_sel['gm'] / df_hon_sel['cshprq']  # gross margin
    df_hon_sel['sa'] = df_hon_sel['xsgaq'] / df_hon_sel['cshprq']  # selling & administrative expenses
    df_hon_sel['ar'] = df_hon_sel['rectq'] / df_hon_sel['cshprq']  # accounts receivable
    df_hon_sel['inv'] = df_hon_sel['invtq'] / df_hon_sel['cshprq']  # inventories
    df_hon_sel['capx'] = df_hon_sel['capxy'] / df_hon_sel['cshprq']  # capital expenditures


    df_hon_sel['EPS_t1'] = df_hon_sel['epspxq'].shift(1)
    df_hon_sel['EPS_t4'] = df_hon_sel['epspxq'].shift(4)
    df_hon_sel['EPS_t5'] = df_hon_sel['epspxq'].shift(5)
    df_hon_sel['INV_t1'] = df_hon_sel['inv'].shift(1)
    df_hon_sel['AR_t1'] = df_hon_sel['ar'].shift(1)
    df_hon_sel['SA_t1'] = df_hon_sel['sa'].shift(1)
    df_hon_sel['GM_t1'] = df_hon_sel['gm'].shift(1)
    df_hon_sel['CAPX_t1'] = df_hon_sel['capx'].shift(1)
    df_hon_sel['ETR_t1'] = df_hon_sel['etr'].shift(1)
    df_hon_sel['EPS'] = df_hon_sel['epspxq']
    df_hon_sel = df_hon_sel.dropna()
    return df_hon_sel


def coalece_stock_prices(path='../../data/spx_stocks', what='Last', override=False):

    parent_folder = os.path.dirname(path)
    out_file = os.path.join(parent_folder, 'spx_500.csv')
    if os.path.exists(out_file) and os.path.isfile(out_file) and not override:
        df = pd.read_csv(os.path.join(parent_folder, out_file),
                         date_parser=lambda dt:
                         pd.to_datetime(dt, format='%Y-%m-%d'),
                         parse_dates=['Index'],
                         index_col=0)
        pickle_file = os.path.join(parent_folder, 'spx_500.pickle')
        df.to_pickle(pickle_file)
        return df

    files = os.listdir(path)
    all_stocks = None
    for fname in files:
        print(os.path.join(path, fname))
        df = pd.read_csv(os.path.join(path, os.path.join(path, fname)),
                    date_parser=lambda dt:
                    pd.to_datetime(dt, format='%Y-%m-%d'),
                    parse_dates=['Index'],
                    index_col=0)
        stock_name = fname[:len(fname)-4]
        df.rename(columns={what: stock_name}, inplace=True)
        if all_stocks is None:
            all_stocks = df[[stock_name]]
        else:
            all_stocks = pd.concat([all_stocks, df[[stock_name]]], axis=1, join='outer')

    all_stocks.to_csv(out_file)
    print('Done combining stock prices')

def covar_to_correl_matrix(covar_mat):
    '''
    Calculate correlation matrix from covariance matrix
    :param covar_mat: covariance matrix, i.e. symmetric square matrix
    :return: correlation matrix
    '''
    diag_mat = np.diag(np.sqrt(np.diag(covar_mat)))
    diag_inv = np.linalg.inv(diag_mat)
    return diag_inv @ covar_mat @ diag_inv


def kritzman_ar(pca, num_components, asset_variances):
    '''
    Principal Components as Measure of Systemic Risk
    Absorption Ratio as a measure of market fragility
    :param pca: trained PCA model
    :param num_components: number of principal components in the numerator
    :param asset_variances: asset variances
    :return: absorption ratio
    '''

    return np.sum(pca.explained_variance_[:num_components]) / np.sum(asset_variances)


def plot_twin_axis(spx_vs_absorp, w=None, h=None):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    if w is None or h is None:
        pass
    else:
        fig.set_size_inches(w, h)
    ax2 = ax1.twinx()

    h1, = ax1.plot_date(spx_vs_absorp.index, spx_vs_absorp['Absorption Ratio'],
                  color='cornflowerblue', linestyle='solid', marker=None, label='Absorption Ratio')

    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Absorption Ratio')
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    ax1.grid(True)

    h2, = ax2.plot_date(spx_vs_absorp.index, spx_vs_absorp['S&P 500'],
                  color='darkorchid', linestyle='solid', marker=None, label='S&P 500 Index')


    ax2.set_ylabel('S&P 500 Index')
    ax1.legend(handles=[h1, h2])
    # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
    fig.autofmt_xdate()
    plt.show()


def get_weight(ar_delta):
    '''
    Calculate EQ / FI portfolio weights based on Absorption Ratio delta
    :param ar_delta: Absorption Ratio delta
    :return: portfolio weights
    '''
    if ar_delta > -1 and ar_delta < 1:
        hold = '50/50'
        return [0.5, 0.5], hold
    elif ar_delta < -1:
        hold = '100_pct_EQ'
        return [1., 0.], hold  # 100% equity, 0% bonds
    elif ar_delta > 1.:
        hold = '100_pct_FI'
        return [0., 1.], hold  # 0% equity, 100% bonds


def sharpe_ratio(ts_returns, periods_per_year=252):
    cumulative_return = np.cumprod(ts_returns + 1)[-1]
    year_frac = (ts_returns.index[-1] - ts_returns.index[0]).days / periods_per_year
    annualized_return = cumulative_return ** (1 / year_frac) - 1
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol
    return annualized_return, annualized_vol, annualized_sharpe


if __name__ == "__main__":
    spx_comp_file = 'SPX_index_constituents_csv_1980_2017.csv'

    # data frame which defines components of S&P 500 through time
    spx_components = pd.read_csv(os.path.join('../../data/spx_components', spx_comp_file),
                         date_parser=lambda dt:
                         pd.to_datetime(dt, format='%Y/%m/%d'),
                         parse_dates=['from', 'thru'])
    ever_present = spx_components[spx_components.thru.isnull()]
    spx_ticker_names = ever_present.loc[:, ('co_tic',)].values

    df = coalece_stock_prices()
    spx_ticker_names = spx_ticker_names.squeeze()
    res = [col_name in df.columns for col_name in spx_ticker_names]
    spx_ticker_names = spx_ticker_names[np.array(res)]
    spx_prices = df[spx_ticker_names]

    first_row = spx_prices.head(1)
    idx_arr = np.logical_not(pd.isnull(first_row).values)
    spx_subset = spx_prices.iloc[:, idx_arr.squeeze()]
    spx_subset = spx_subset.fillna(method='ffill')

    spx_returns = spx_subset.pct_change(periods=1)
    mean_returns = spx_returns.mean(axis=0)
    sd_returns = spx_returns.std(axis=0)
    norm_spx_returns = (spx_returns - mean_returns) / sd_returns
    print('%d stocks have data from %s till %s' % (spx_subset.shape[1], spx_subset.index[0], spx_subset.index[-1]))

    norm_spx_returns = norm_spx_returns.iloc[1:, :]
    pca = sklearn.decomposition.PCA().fit(norm_spx_returns)

    # computing PCA on
    # cumulative variance explained
    var_threshold = 0.8
    var_explained = np.cumsum(pca.explained_variance_ratio_)
    num_comp = np.where(np.logical_not(var_explained < var_threshold))[0][0] + 1  # +1 due to zero based-arrays

    # Principal axes in feature space, representing the directions of maximum variance in the data.
    # The components are sorted by explained_variance_

    # first eigen-portfolio weights
    n_days = 252
    recent_asset_returns = spx_returns.tail(n_days)
    first_eigen_prtf_wgts = pca.components_[0] / pca.explained_variance_[0]
    second_eigen_prtf_wgts = pca.components_[1] / pca.explained_variance_[1]
    first_wgts = np.sort(first_eigen_prtf_wgts)[::-1]       # first eigen portfolio
    second_wgts = np.sort(second_eigen_prtf_wgts)[::-1]     # second eigen portfolio

    first_eigen_prtf_returns = np.sum(recent_asset_returns * first_eigen_prtf_wgts, axis=1)
    second_eigen_prtf_returns = np.sum(recent_asset_returns * second_eigen_prtf_wgts, axis=1)

    cum_ret = first_eigen_prtf_returns + 1
    cum_ret = cum_ret.cumprod()
    first_portfolio_ret = cum_ret[-1] - 1

    cum_ret = second_eigen_prtf_returns + 1
    cum_ret = cum_ret.cumprod()
    second_portfolio_ret = cum_ret[-1] - 1

    n_portfolios = 120
    annualized_ret = np.array([np.nan] * n_portfolios)
    sharpe_metric = np.array([np.nan] * n_portfolios)

    sqrt_time = np.sqrt(n_days)
    for ix in range(n_portfolios):
        # eigen portfolio weights are per Avellaneda Jeong-Hyun Lee - Statistical Arbitrage in U.S. Markets
        eigen_prtf_wgts = pca.components_[ix] / sd_returns

        # is this correct ?
        # eigen_prtf_wgts = pca.components_[ix] / pca.explained_variance_

        # re-normalize:
        eigen_prtf_wgts = eigen_prtf_wgts / sum(eigen_prtf_wgts)

        eigen_prtf_returns = np.sum(recent_asset_returns * eigen_prtf_wgts, axis=1)
        mean_return = eigen_prtf_returns.mean()

        annualized_return = (mean_return + 1.)**n_days - 1    # annualize daily returns
        vol = eigen_prtf_returns.std() * sqrt_time

        sharpe_metric[ix] = annualized_return / vol
        annualized_ret[ix] = annualized_return

    # portfolio with the highest Sharpe ratio
    idx_highest_sharpe = sharpe_metric.argmax()

    ########### begin PCA via correlation matrix ############
    cov_matrix = norm_spx_returns.cov()
    print(cov_matrix)
    correl_mat = covar_to_correl_matrix(cov_matrix)
    correl_mat = pd.DataFrame(correl_mat, columns=cov_matrix.columns, index=cov_matrix.columns)

    print(correl_mat)
    n_asset = cov_matrix.shape[0]
    eigen_values, eigen_vec = np.linalg.eigh(cov_matrix)
    # The loadings (eigen vectors) of a PCA decomposition can be treated as principal factor weights.
    # In other words, they represents asset weights towards each principal component portfolio.
    # The total number of principal portfolios equals to the number of principal components.
    # The variance of each principal portfolio is its corresponding eigenvalue.

    pca_scores = eigen_values / np.sum(eigen_values)

    # variance-covariance matrix of principal components:
    pc_cov_matrix = eigen_vec.T @ cov_matrix @ eigen_vec
    # based on Meucci's "Managing Diversification": given asset weights calculate exposure to each principal component
    # calculate exposure to each principal component
    pc_exposure = np.linalg.inv(eigen_vec) @ np.array([[1 / n_asset] * n_asset]).T
    print(pc_exposure)

    # eigen-portfolios from eigen vectors
    # does not seem to tie out with PCA calculations, but eigen_vec matches pca.components_
    for ix in range(1, n_portfolios + 1):
        eigen_prtf_wts = eigen_vec[:, -ix] / sum(eigen_vec[:, -ix])

    ############ end of eigen values from correlation matrix ##############

    import matplotlib.pyplot as plt

    bar_width = 0.9
    n_asset = int((1 / 8) * norm_spx_returns.shape[1])
    x_indx = np.arange(n_asset)
    fig, ax = plt.subplots()
    rects = ax.bar(x_indx, pca.explained_variance_ratio_[:n_asset], bar_width, color='olive')
    ax.set_xticks(x_indx + bar_width / 2)
    ax.set_xticklabels(['PC%d' % ix for ix in range(n_asset, 0, -1)], rotation=45)
    ax.set_title('Percent variance explained')
    ax.legend((rects[0],), ('Percent variance explained by principal components',))
    print('Percent variance explained:\n%s' % pca.explained_variance_ratio_)

    #############################
    # compute absorption ratio
    # the idea is to fix the variance explained and see how many components are needed
    # during the market crisis asset returns are highly correlated, hence the number of PCA components for the same
    # amount of variance explained drops

    # Kritzman
    # A high value for the absorption ratio corresponds to a high level of systemic risk because it
    # implies that the sources of risk are more unified. A low absorption ratio indicates less systemic risk
    # because it implies that the sources of risk are more disparate. We should not expect high systemic risk to
    # necessarily lead to asset depreciation or financial turbulence. It is simply a measure of market fragility in
    # the sense that a shock is more likely to propagate quickly  and broadly when sources of risk are tightly coupled.
    lookback_window = 252   # days
    step_size = 1           # days : 7 - weekly, 21 - monthly, 63 - quarterly
    var_threshold = 0.8     # require of that much variance to be explained
    pca_ts_index = norm_spx_returns.index[list(range(lookback_window, len(norm_spx_returns), step_size))]

    saved_ar_file = os.path.join('../../data/', 'ts_pca_components_ar.pickle')
    if not os.path.exists(saved_ar_file) and os.path.isfile(saved_ar_file):
        # exponentially moving average parameter for computing
        half_life = 0.4
        alpha_decay = 1 - np.exp(np.log(0.5) / half_life)

        pca_components = np.array([np.nan]*len(pca_ts_index))
        absorb_ratio = np.array([np.nan]*len(pca_ts_index))
        kritzman_absorb_ratio = np.array([np.nan]*len(pca_ts_index))
        absorb_comp = int((1 / 5) * norm_spx_returns.shape[1])  # fix 20% of principal components

        ik = 0
        for ix in range(lookback_window, len(norm_spx_returns), step_size):
            ret_frame = norm_spx_returns.iloc[ix - lookback_window:ix, :]
            pca = sklearn.decomposition.PCA().fit(ret_frame)
            var_explained = np.cumsum(pca.explained_variance_ratio_)

            # exponentially weighted moving standard deviation
            exp_wgt_sd = \
                ret_frame.ewm(adjust=True, halflife=half_life, min_periods=0, ignore_na=False).std(bias=False).iloc[-1, :]
            exp_wgt_var = exp_wgt_sd ** 2

            kritzman_absorb_ratio[ik] = kritzman_ar(pca, absorb_comp, exp_wgt_var)

            absorb_ratio[ik] = np.sum(pca.explained_variance_[:absorb_comp]) / np.sum(pca.explained_variance_)

            num_comp = np.where(np.logical_not(var_explained < var_threshold))[0][0] + 1  # +1 due to zero based-arrays
            pca_components[ik] = num_comp
            ik += 1

        print('Done computing absorption ratio')
        ts_pca_components = pd.Series(pca_components, index=pca_ts_index)
        ts_kritzman_ar = pd.Series(kritzman_absorb_ratio, index=pca_ts_index)
        ts_absorb_ratio = pd.Series(absorb_ratio, index=pca_ts_index)
    else:
        ts_pca_components = pd.read_pickle(os.path.join('../../data/', 'ts_pca_components_ar.pickle'))
        ts_kritzman_ar = pd.read_pickle(os.path.join('../../data/', 'ts_ar_kritzman.pickle'))
        ts_absorb_ratio = pd.read_pickle(os.path.join('../../data/', 'absorption_ratio.pickle'))

    # plt.plot(ts_kritzman_ar)

    # following Kritzman and computing AR_delta = (15d_AR -1yr_AR) / sigma_AR
    ar_mean_1yr = ts_absorb_ratio.rolling(252).mean()
    ar_mean_15d = ts_absorb_ratio.rolling(15).mean()
    ar_sd_1yr = ts_absorb_ratio.rolling(252).std()
    ar_delta = (ar_mean_15d - ar_mean_1yr) / ar_sd_1yr    # standardized shift in absorption ratio

    ########## load S&P 500 index price data
    spx_index_prices = pd.read_csv(os.path.join('../../data', 'spx_index_1980_2017.csv'),
                                   date_parser=lambda dt:
                                   pd.to_datetime(dt, format='%d-%b-%Y'),
                                   dtype={'PX_LAST': np.float32, 'PX_OPEN': np.float32,
                                          'PX_HIGH': np.float32, 'PX_LOW': np.float32, 'PX_ASK': np.float32,
                                          'PX_BID': np.float32, 'PX_VOLUME': np.float32},
                                   parse_dates=[0],
                                   na_values=['#N/A N/A'],
                                   index_col=0)

    spx_index_returns = spx_index_prices.pct_change(periods=1)
    matching_spx_returns = spx_index_returns.loc[pca_ts_index, :]
    matching_spx_prices = spx_index_prices.loc[pca_ts_index, :]
    ####### done loading S&P 500 data and computing returns


    spx_vs_absorp = pd.concat([ts_kritzman_ar, matching_spx_prices['PX_LAST']], axis=1, join='inner')
    spx_vs_absorp.rename(columns={0: 'Absorption Ratio', 'PX_LAST': 'S&P 500'}, inplace=True)
    plot_twin_axis(spx_vs_absorp)

    spx_vs_absorp = pd.concat([ts_absorb_ratio, matching_spx_prices['PX_LAST']], axis=1, join='inner')
    spx_vs_absorp.rename(columns={0: 'Absorption Ratio', 'PX_LAST': 'S&P 500'}, inplace=True)
    plot_twin_axis(spx_vs_absorp)

    print('Done plotting absorption ratio')

    # trading strategy based on Absorption Ratio Delta
    ar_delta = ar_delta[251:]
    rebal_dates = np.zeros(len(ar_delta))
    wgts = pd.DataFrame(data=np.zeros((len(ar_delta.index), 2)), index=ar_delta.index, columns=('EQ', 'FI'))
    wgts.iloc[0, :], hold = get_weight(ar_delta.values[0])
    for ix in range(1, len(ar_delta)):
        wgts.iloc[ix, :], hold_new = get_weight(ar_delta.values[ix])
        if hold != hold_new:
            print('switch from %s to %s on %s' % (hold, hold_new, ar_delta.index[ix]))
            hold = hold_new
            rebal_dates[ix] = 1

    ts_rebal_dates = pd.Series(rebal_dates, index=ar_delta.index)
    ts_trades_per_year = ts_rebal_dates.groupby([ts_rebal_dates.index.year]).sum()
    print('Average number of trades per year %.2f' % ts_trades_per_year.mean())

    # 'AGG' - iShares Core U.S. Aggregate Bond ETF - Broad Market Investment Grade - 2003-09-26
    # 'VTI' - Vanguard Total Stock Market ETF - 2001-05-03
    # 'BND' - Vanguard Total Bond Market
    vti_prices = pd.read_csv(os.path.join('~/data/betterment_v4/ETF', 'VTI.csv'),
                             date_parser=lambda dt:
                             pd.to_datetime(dt, format='%Y-%m-%d'),
                             parse_dates=['Index'],
                             index_col=0)
    vti_prices.rename(columns={'Last': 'VTI'}, inplace=True)

    agg_prices = pd.read_csv(os.path.join('~/data/betterment_v4/ETF', 'AGG.csv'),
                             date_parser=lambda dt:
                             pd.to_datetime(dt, format='%Y-%m-%d'),
                             parse_dates=['Index'],
                             index_col=0)
    agg_prices.rename(columns={'Last': 'AGG'}, inplace=True)

    prices_data = pd.concat([vti_prices[['VTI']], agg_prices[['AGG']]], axis=1, join='inner')
    returns_data = prices_data.pct_change(periods=1)
    returns_data = returns_data.iloc[1:, ]
    strat_wgts = wgts.loc[returns_data.index, :].dropna()   # dropping NAs because weights might not be available
    # use returns data for which we have weights:
    strat_returns = np.sum(returns_data.loc[strat_wgts.index] * strat_wgts.values, axis=1)

    # compute strategy's annualized return
    year_frac = (strat_returns.index[-1] - strat_returns.index[0]).days / 252
    ar_delta_er, ar_delta_vol, ar_delta_sharpe = sharpe_ratio(strat_returns[:'2011-01-03'])

    # compute benchmark (equally weighted) strategy's annualized return
    eq_wgts = strat_wgts.copy()
    eq_wgts.iloc[:, ] = 0.5
    eq_wgts_strat_returns = np.sum(returns_data.loc[strat_wgts.index] * eq_wgts.values, axis=1)
    eq_wgt_er, eq_wgt_vol, eq_wgt_sharpe = sharpe_ratio(eq_wgts_strat_returns[:'2011-01-03'])

    print('Done')
    # further research : does this apply to Betterment Core ?

    # read SPX index data
    # data for financial ratios
    # df = get_eps_data('HON',  path='../../data/finratio')