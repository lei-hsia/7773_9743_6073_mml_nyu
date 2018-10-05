import tarfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sm_api
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import seaborn as sns

from pymlfin.util.util import get_eps_data

class Dummy:
    pass

def ibesFinRatios():
    '''
    https://wrds-web.wharton.upenn.edu/wrds/ds/ibes/act/index.cfm?navId=223
    :return:
    '''
    pass


def combineFinRatios():
    '''
    @see https://github.com/justmarkham/DAT7/blob/master/notebooks/10_linear_regression.ipynb

    read WRDS financial ratios data and SPX500 constituents in CSV format and combine them
    in one CSV file
    :return:
    '''

    path = '~/PycharmProjects/ml/WRDS_data/Financial_Ratios/'

    # load index constituents
    index_comp_fn = 'SPX_index_constituents_csv_1980_2017.csv'
    df_index_comp = pd.read_csv(os.path.join(path, index_comp_fn),
                                date_parser=lambda x: None if pd.isnull(x) else pd.datetime.strptime(x, '%Y/%m/%d'),
                                parse_dates=['from', 'thru'],
                                verbose=True,
                                index_col=0)
    df_index_comp['thru'] = pd.to_datetime(df_index_comp['thru'])
    print(df_index_comp.loc[:, ('from', 'thru', 'co_tic', 'co_cusip', 'co_cik')].head(10))

    file_name = 'WRDS_fin_ratios_spx500_constituents_2004_01_2015_12.csv'
    df_fin_ratios = pd.read_csv(os.path.join(path, file_name),
                                date_parser=lambda x: None if pd.isnull(x) else pd.datetime.strptime(x, '%Y/%m/%d'),
                                parse_dates=['adate', 'qdate', 'public_date'],
                                verbose=True,
                                index_col=0)
    print(df_fin_ratios.loc[:, ('pe_op_basic', 'adate', 'qdate', 'public_date')].head(10))
    print(df_fin_ratios.loc[:, ('pe_op_basic', 'adate', 'qdate', 'public_date')].tail(10))

    # no longer needed since the data set has already been truncated
    # df_fin_ratios = df_fin_ratios[df_fin_ratios['public_date'].dt.year >= 2005]
    missing_adate = df_fin_ratios[pd.isnull(df_fin_ratios['adate'])]
    if len(missing_adate) > 0:
        print('Found %d  rows with missing date' % len(missing_adate))

    df_fin_ratios['adate'] = pd.to_datetime(df_fin_ratios['adate'])

    # join by gvkey the constituents with financial ratios data frame
    result = df_fin_ratios.join(df_index_comp, how='inner')

    # missing date:
    # result.loc[pd.isnull(result['adate']), ('pe_op_basic', 'adate', 'qdate', 'public_date', 'from', 'thru', 'co_tic')]
    active_comp_fin_data = result.loc[pd.isnull(result['thru']) | (result['thru'] >= result['public_date'])]
    active_comp_fin_data.loc[(active_comp_fin_data['co_tic'] == 'AAPL'),
                             ('CAPEI', 'de_ratio', 'co_tic', 'public_date', 'from', 'thru', 'bm', 'cash_lt')]

    # gpm - Gross Profit Margin
    # dpr - Dividend Payout Ratio
    # ptb - Price to Book ratio
    # pretret_noa - Pre-tax return on Net Operating Assets
    file_name = 'WRDS_fin_ratios_spx500_constituents_2005_01_2015_12.csv'
    result.to_csv(os.path.join(path, file_name), date_format='%Y/%m/%d')

    #sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=6, aspect=0.7)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def predict(model, inputs):
    # the last value is always 1 because we have B_ij + B_0j
    # hence, to avoid storing b_0j in a separate variable we clamp B_0j into input vector eps
    hidden = np.dot(model['W1'], inputs)
    out = np.dot(model['W2'], sigmoid(hidden))
    return out


def train_neural_network(eps_intc):
    ###################  Neural Network with one hidden layer ###################
    num_hidden = 26  # number of neurons in the hidden layer
    num_inp = 4  # number of inputs to neuron

    # pre-process data
    x_data = eps_intc - eps_intc.mean()
    n_samples = len(x_data.index)

    # we have options how to initialize model weights
    # the last columns in W1 containes biases. it is common to initialize biases to 0. or 0.01
    # http://cs231n.github.io/neural-networks-2/
    model = {}
    # Initialize the weights by drawing them from a gaussian distribution with standard deviation
    # of sqrt( 2/ n) where nn is the number of inputs to the neuron
    model['W1'] = np.random.randn(num_hidden, num_inp + 1) / np.sqrt(num_inp / 2)  # He et al. 2015 , (note an additional /2)
    model['W1'] = np.random.randn(num_hidden, num_inp + 1) / np.sqrt(num_inp)  # "Xavier" initialization
    model['W2'] = np.random.randn(num_hidden) / np.sqrt(num_inp)

    # Batch normalization - make it roughly
    # Insert batch normalization layers into the network Ioffe Szegedy, 2015
    # take input x - make sure that every feature dimension
    # N things in mini-batch and D features - inputs
    # Batch normalization applies
    # Layers with BN: tanh <- BN <-FC <- tanh <- BN <- FC
    # (BN layers are usually inserted after Fully Connected or Convolutional layers)
    # Normalize: x_hat^(k) = [ x^(k)  - E[x^(k)] ] / sqrt(Var[ x^(k) ])
    # And then allow the network to squash the range if it wants to:
    # y^(k) = \gamma^(k) x_hat^(k) + \beta^(k)


    # Stochastic gradient descent
    step_size = 0.01
    num_iter = 10
    for _ in range(num_iter):
        # forward propagation

        # randomly sample 4-tuple of EPS[t-1], EPS[t-2], ...., EPS[t-4]
        eps = np.array([3., 2., 2.56, 4., 1])
        # we are predicting EPS[t]
        out = predict(model, eps)

        # output of the forward prop

        # calculate the gradient of all nodes in the network
        # need to have regularizaton

        # update model parameters W1, and W2

    # done gradient descent

    ############### Variations of Stochstic Gradient Descent ######################
    # gradient descent update: x+= -learning_rate * dx

    # Variation #1: momentum update:
    # v = mu * v -learning_rate * dx # integrate velocity
    # x += v # integrate position
    # mu * v has an interpretation of friction which creates a loss of energy
    # mu is a coefficient, and usually is 0.5, 0.9, 0.99
    # initialize v with zero, v is velocity


    # Variation #2: Nesterov Momentum Update, also known as Nesterov Accelerated Gradient (nag)
    # go in the direction of momentum first and then calculate the gradient at that point
    # v_t = \mu v_{t-1} - \epsilon \grad f(\theta_{t-1} + \mu v_{t-1})
    # \theta_t = \theta_{t-1} + v_t

    # Nesterov Accelerated Gradient (nag)
    # To make Nesterov Momentum Update look like a regular descent we change variable
    # usually we have \theta_{t-1}, \grad f(\theta_{t-1})
    # \phi_t = theta_{t-1} +\mu v_{t-1}
    # v_t = \mu v_{t-1} - \epsilon \grad f(\phi_{t-1})
    # \phi_t = \phi_{t-1} - \mu v_{t-1} + (1 + \mu) * v_t
    # we do updates on \phi
    # in code this looks like:

    # v_prev = v
    # v = mu * v - learning_rate * dx
    # x += -mu * v_prev + (1 + mu) * v

    # Variation #3: AdaGrad  update
    # Also called per parameter adaptive learning rate method
    # cache  +=  dx **2    # this is un-centered second moment
    # x += -learning_rate * dx / (np.sqrt(cache) + 1e-7)
    # cache is historical sum of squares in each dimension
    # cache is a giant vector and it is of the same size as your parameter vector
    # what happens to the step size over time ?  learning rate decays towards zero
    # hence, AdaGrad needs some adjustment because NN needs non-zero learning rate at all times

    # Variation #4: RMSprop update (Tielman and Hinton)
    # Jeff Hinton proposal to adjust AdaGrad:
    # instead of keeping a sum of squares in each dimension, we make a leaky counter
    # This is called RMSprop update :
    # cache  = decay_rate * cache + (1 - decay_rate) * dx **2    # this is un-centered second moment
    # x += -learning_rate * dx / (np.sqrt(cache) + 1e-7)

    # Variation #5: Adam update: Combine AdaGrad with momentum , Kingma and Ba (2014)
    # m = beta1 * m + (1 - beta1) * dx # update the first moment
    # v = beta2 * v + (1 - beta2) * (dx **2) # update the second moment
    # x += -learning_rate * m / (np.sqrt(v) + 1e-7)
    # beta1 = 0.9 and beta2 = 0.995 requires cross-validation at times

    # t has to start at 1
    # bias correction does not modify m, v, in place but creates new mb, vb (bias corrected versions)
    # which are used in the update
    # Adam
    # m, v =  ... initialize caches to zero
    # for t  in range(0, big_number):
    #   m = beta1 * m + (1 - beta1) * dx # update the first moment
    #   v = beta2 * v + (1 - beta2) * (dx **2) # update the second moment
    #   mb /= 1 - beta1**2 # correct bias
    #   mv /= 1 - beta2**2 # correct bias
    #   x += -learning_rate * mb / (np.sqrt(vb) + 1e-7)
    # x is a parameter vector

    # Learning rate - must decay over-time
    # Step-decay: e.g. decay learning rate by half every few epochs
    # Exponential decay: \alpha = \alpha_0 e^{- \kappa t}
    # 1/t decay: \alpha = \alpha_0 / (1 + \kappa t)

    # Epoch - time after which we have seen every single example at least one time

    # Those variations are so called first-order methods because they use only gradient
    # information at your loss function. We know the gradient in every single direction

    # Second order methods:
    # Approximated by Hessian to see how the surface is curving
    # Faster conversion, no hyper parameters, no learning rate
    # Hessian matrix needs to be inverted, a problem
    # BFGS builds up Hessian (need to be stored in matrix)
    # L-BFGS - does not store full Hessian in memory
    # But we work on large data sets , i.e. in mini-batches because all data does not fit into
    # memory and LBFGS works on mini-batches. If can afford full batch use L-BFGS
    # (while using L-BFGS don't forget to disable all sources of noise)


    # Model Ensembles
    # Train multiple independent models and average
    # Enjoy 2% extra performance


    # Regularization (dropout)
    # Randomly set some neurons to zero as we doing forward pass of the data x

    def train_step(x):
        '''
        :param x contains data
        x has dimension N x D where N number of samples and D number of features / attributes
        '''

        # prob_dropout, wgts1, wgts2, wgts3, b1, b2, b3 are initialized outsize of train_step
        prob_dropout = 0.5
        wgts1 = b1 = wgts2 = b2 = wgts3 = b3 = None

        # forward pass for example 3-layer neural network
        h1 = np.maximum(0, np.dot(wgts1, x) + b1)
        u1 = np.random.rand(*h1.shape) < prob_dropout  # first dropout mask
        h1 *= u1  # drop some neurons

        h2 = np.maximum(0, np.dot(wgts2, x) + b2)
        u2 = np.random.rand(*h2.shape) < prob_dropout  # second dropout mask
        h2 *= u2  # drop some neurons

        out = np.dot(wgts3, h2) + b3

        # backward pass: compute gradients ...  (not shown) has to be adjusted
        # drops have to be applied in back prop as well
        # perform parameter update (not shown)


def multi_variable_eps():
    '''
    https://wrds-web.wharton.upenn.edu/wrds/ds/crsp/ccm_a/fundq/index.cfm?navId=120
    Y = Quarterly EPS;
    INV = Inventory; = invtq
    AR = Accounts receivables; AR = rectq
    CAPX = Capital expenditure per Schedule V (Schedule V contains disclosures
    of property, plant, and equipment under the Securities and Exchange Commission [SEC] Regulation S-X.);
    since Compustat reports only annual capital expenditure, we divide the annual
    amount by four to approximate to the quarterly capital expenditure;
    GM = Gross margin, defined as sales less cost of goods sold; GM =  saleq - cogsq
    Gross Margin = (revenue - cogs) / cogs

    SA = Selling and administrative expenses;
    ETR = Effective tax rate, defined as income taxes divided by pretax income; ETR = txty / piq
    LF = Labor force, defined as sales divided by the number of employees;
    since Compustat reports the number of employees on a yearly basis, we take this number as the quarterly,
    assuming that the number of employed remains constant across the four quarters.

    Columns in the dataset:
    cogsq	cost of good sold
    cshoq	common shares outstanding
    cshpry  common shares used for calculating EPS
    cshprq common shares used for calculating EPS
    epspiq	earnings per share (dilluted) including extraordinary items
    epspxq	earnings per share (dilluted) excluding extraordinary items
    invtq	inventories
    rectq	account receivables
    saleq	net sales - Sales/Turnover (Net)
    Turnover is the net sales generated by a business, while profit is the residual earnings
    of a business after all expenses have been charged against net sales.
    thus, turnover and profit are essentially the beginning and ending points
    of the income statement - the top-line revenues and the bottom-line results.
    xsgaq	administrative expense
    piq 	Pretax Income
    txtq	Income taxes total
    txpdy	Income taxes paid
    prccq	Close price for the quarter
    capxy	capital expenditures
    revtq   Revenue Total
    :return:
    '''

    import scipy.stats

    # Using merged CRSP / Compustat data

    print(os.getcwd())
    df = get_eps_data('MSFT', path='../../data/finratio')
    # M2.1 by Lorek and Willinger (1996)
    # E(EPS[t]) = alpha + beta_1 *EPS[t-1] + beta_2*EPS[t-4] +
    # beta_3 * INV[t-1] + beta_4 * AR[t-1] + beta_5 * CAPX[t-1] + beta_6 *GM[t-1] +
    # beta_7 * SA[t-1] + beta_8 * ETR[t-1] + beta9 * LF[t-1] + epsilon_t

    eps_df = pd.DataFrame(df['EPS'].values, columns=['EPS'], index=df.index)
    predictors_df = df.loc[:, ('EPS_t1', 'EPS_t4', 'INV_t1', 'AR_t1', 'CAPX_t1', 'GM_t1', 'SA_t1', 'ETR_t1')]

    tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=2)
    for train_index, test_index in tscv.split(predictors_df):
        predict_train, predict_test = predictors_df.loc[predictors_df.index[train_index]], \
                                      predictors_df.loc[predictors_df.index[test_index]]
        eps_train, eps_test = eps_df.loc[eps_df.index[train_index]], \
                              eps_df.loc[eps_df.index[test_index]]

        lm_lorek = sklearn.linear_model.LinearRegression().fit(predict_train, eps_train)
        print("M2.1 Lorek and Willinger (1996) via scikit-learn")
        print("model coefficients: %s" % lm_lorek.coef_)
        print("model intercept: %.4f" % lm_lorek.intercept_[0])
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(eps_train, lm_lorek.predict(predict_train)))
        print("scikit-learn in-sample RMSE = %.4f" % rmse)

        eps_forecast = lm_lorek.predict(predict_test)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(eps_test, eps_forecast))
        print("scikit-learn out-of-sample RMSE = %.4f" % rmse)

        eps_forecast = pd.DataFrame(eps_forecast, index=eps_test.index, columns=('EPS Forecast',))
        comb = pd.concat([eps_forecast, eps_test], axis=1, join='inner')
        comb.rename(columns={0: 'EPS Forecast'}, inplace=True)
        comb.describe()
        comb.plot(title='sklearn.linear_model.LinearRegression')

        # fit stats models multi-variable linear regression model
        model_lorek = sm.OLS(eps_train, predict_train).fit()
        print("M2.1 Lorek and Willinger (1996) via stats models")
        print(model_lorek.summary())

        # make prediction
        eps_forecast = model_lorek.predict(predict_test)

        # compute root mean squared error of the forecasts
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(eps_test, eps_forecast))
        print("stats models out-of-sample RMSE = %.4f" % rmse)

        comb = pd.concat([eps_forecast, eps_test], axis=1, join='inner')
        comb.rename(columns={0: 'EPS Forecast'}, inplace=True)
        comb.describe()
        comb.plot(title='stats models OLS')

    # based on summary from stats models we see we can reject the null hypothesis for
    # INV_t1        -0.1030      0.025     -4.095      0.000      -0.153      -0.053
    # AR_t1         -0.0721      0.031     -2.363      0.020      -0.133      -0.012
    # CAPX_t1       -0.0857      0.040     -2.147      0.034      -0.165      -0.007
    # GM_t1          0.2673      0.078      3.411      0.001       0.112       0.422
    # SA_t1          0.6459      0.181      3.560      0.001       0.287       1.005
    # Hence, we construct  a new model using only those variable that have predictive power:
    tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=2)
    predictors_df = df.loc[:, ('INV_t1', 'AR_t1', 'CAPX_t1', 'GM_t1', 'SA_t1')]
    for train_index, test_index in tscv.split(predictors_df):
        predict_train, predict_test = predictors_df.loc[predictors_df.index[train_index]], \
                                      predictors_df.loc[predictors_df.index[test_index]]
        eps_train, esp_test = eps_df.loc[eps_df.index[train_index]], \
                              eps_df.loc[eps_df.index[test_index]]

        model_lorek = sm.OLS(eps_train, predict_train).fit()
        print(model_lorek.summary())
        eps_forecast = model_lorek.predict(predict_test)
        # compute root mean squared error
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(esp_test, eps_forecast))
        print("stats models out-of-sample RMSE = %.4f" % rmse)

    # plot EPS vs INV[t-1],  EPS vs AR[t-1], EPS vs CAPX[t-1]
    sns.pairplot(df, x_vars=['INV_t1', 'AR_t1', 'CAPX_t1'], y_vars='EPS', size=7, aspect=0.7, kind='reg')


    # M2.2 Abarbanell and Bushee (1997)
    # E(EPS[t]) = alpha + beta_1 * EPS[t - 1] + beta_2 * EPS[t-4] +
    # beta_3 * INV[t-4] + beta_4 * AR[t-4] + beta_5 * CAPX[t-4] + beta_6 *GM[t-4] +
    # beta_7 * SA[t-4] + beta_8 * ETR[t-4] + beta9 * LF[t-4] + epsilon_t
    # prepare data for M2.2 model
    df['INV_t4'] = df['inv'].shift(4)
    df['AR_t4'] = df['ar'].shift(4)
    df['SA_t4'] = df['sa'].shift(4)
    df['GM_t4'] = df['gm'].shift(4)
    df['CAPX_t4'] = df['capx'].shift(4)
    df['ETR_t4'] = df['etr'].shift(4)
    df = df.dropna()
    predictors_df = df.loc[:, ('EPS_t1', 'EPS_t4', 'INV_t4', 'AR_t4', 'CAPX_t4', 'GM_t4', 'SA_t4', 'ETR_t4')]
    for train_index, test_index in tscv.split(predictors_df):
        predict_train, predict_test = predictors_df.loc[predictors_df.index[train_index]], \
                                      predictors_df.loc[predictors_df.index[test_index]]
        eps_train, eps_test = eps_df.loc[eps_df.index[train_index]], \
                              eps_df.loc[eps_df.index[test_index]]

        lm_abarbanell = sklearn.linear_model.LinearRegression().fit(predict_train, eps_train)
        print("M2.2 Abarbanell and Bushee (1997) via scikit-learn")
        print("model coefficients: %s" % lm_abarbanell.coef_)
        print("model intercept: %.4f" % lm_abarbanell.intercept_[0])
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(eps_train, lm_abarbanell.predict(predict_train)))
        print("scikit-learn in-sample RMSE = %.4f" % rmse)

        eps_forecast = lm_abarbanell.predict(predict_test)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(eps_test, eps_forecast))
        print("scikit-learn out-of-sample RMSE = %.4f" % rmse)

        eps_forecast = pd.DataFrame(eps_forecast, index=eps_test.index, columns=('EPS Forecast',))
        comb = pd.concat([eps_forecast, eps_test], axis=1, join='inner')
        comb.rename(columns={0: 'EPS Forecast'}, inplace=True)
        comb.describe()
        comb.plot(title='sklearn.linear_model.LinearRegression')


def arma_eps():
    '''
    From IBES Guideance July 2009
    CPX Capital Expenditure
    CPXPAR  Capital Expenditure –Parent
    DPS Dividends per Share
    EBS EBITDA per Share
    EBSPAR EBITDA per Share –Parent
    EBT EBITDA
    EBTPAR  EBITDA –Parent
    EPS Earnings Per Share
    EPSPAR  Earnings Per Share –Parent
    FFO Funds From Operations Per Share
    FFOPAR  Funds From Operations Per Share –Parent
    GPS Fully Reported Earnings Per Share
    GPSPAR Fully Reported Earnings Per Share –Parent
    GRM Gross Margin
    GRMPAR Gross Margin –Parent
    NET Net Income
    NETPAR  Net Income –Parent
    OPR Operating Profit
    OPRPAR  Operating Profit –Parent
    PRE Pretax Income
    PREPAR  Pretax Income –Parent
    ROA Return on Assets (%)
    ROAPAR  Return on Assets (%)-Parent
    ROE Return on Equity (%)
    ROEPAR  Return on Equity (%)-Parent
    SAL Sales
    SALPAR Sales –Parent
    :return:
    '''
    # I/B/E/S Thompson Reuters Data Set
    print("============================ I/B/E/S Thompson Reuters Data Set ================================")
    print("============================         Univariate EPS models     ================================")
    print(os.getcwd())
    path = '../WRDS_data/IBES'
    load_all_tickers = False
    df_ibes = None
    if load_all_tickers:
        # VC: this is too large of a file, load per ticker
        file_name = 'IBES_Actuals_FinRatios_csv_01Jan1990 - 31Mar2017.csv'
        df_ibes = pd.read_csv(os.path.join(path, file_name),
                              date_parser=lambda dt, tm:
                              pd.to_datetime(dt + ' ' + tm, format='%Y/%m/%d %H:%M:%S'),
                              parse_dates=[['ACTDATS', 'ACTTIMS'], ['ANNDATS', 'ANNTIMS']],
                              index_col=0)

        df_ibes.set_index(['TICKER', 'ACTDATS', 'ACTTIMS'], inplace=True)

        all_intc_ibes = df_ibes[df_ibes['TICKER'] == 'INTC']
    else:
        file_name = 'IBES_Actuals_INTC_01Jan1990 - 31Mar2017.csv'
        all_intc_ibes = pd.read_csv(os.path.join(path, file_name),
                                   date_parser=lambda dt, tm:
                                        pd.to_datetime(dt + ' ' + tm, format='%Y/%m/%d %H:%M:%S'),
                                   parse_dates=[['ACTDATS', 'ACTTIMS'], ['ANNDATS',	'ANNTIMS']],
                                   index_col=0)

    # Earnings per Share (EPS) for Intel Corporation (INTC)
    eps_intc = all_intc_ibes[all_intc_ibes['MEASURE'] == 'EPS']
    intc_eps_summary = eps_intc.groupby(['TICKER', eps_intc.index]).agg(
        {'VALUE': 'mean', 'ANNDATS_ANNTIMS': ['min', 'max', 'count']})

    intc_eps_summary.head(10)

    # pandas multi-index
    # https://pandas.pydata.org/pandas-docs/stable/cookbook.html#levels
    eps_intc.reset_index(level=0, inplace=True)
    eps_intc = eps_intc.drop(['OFTIC', 'TICKER', 'CUSIP',
                              'PENDS', 'MEASURE', 'PDICITY', 'ANNDATS_ANNTIMS', 'CURR_ACT'], axis=1)

    # set index so that eps_intc can be used with statespace.SARIMAX
    eps_intc.set_index(['ACTDATS_ACTTIMS'], inplace=True)

    eps_tminus1 = eps_intc.shift(1)
    eps_tminus4 = eps_intc.shift(4)
    eps_tminus5 = eps_intc.shift(5)
    eps_tminus1.rename(columns={'VALUE': 'EPS_t1'}, inplace=True)
    eps_tminus4.rename(columns={'VALUE': 'EPS_t4'}, inplace=True)
    eps_tminus5.rename(columns={'VALUE': 'EPS_t5'}, inplace=True)
    result = pd.concat([eps_tminus1, eps_tminus4, eps_tminus5, eps_intc], axis=1, join='inner')
    result.rename(columns={'VALUE': 'EPS'}, inplace=True)

    result['EPS_t1_minus_EPS_t5'] = result['EPS_t1'] - result['EPS_t5']

    result = result.dropna()    # remove rows with NAs so that result could be used with OLS
    # the result can now be used to fit various models

    # model M1.3 Foster (1977)
    # E[Y(t)] = Y(t-4) + \phi_1 * [Y(t-1) -Y(t-5)] + \delta
    ols_foster_result = sm.ols(formula="EPS ~ EPS_t4 + EPS_t1_minus_EPS_t5", data=result).fit()
    print(ols_foster_result.summary())

    # analyzing the residuals
    import scipy.stats
    print("Normal residuals?")
    print(scipy.stats.normaltest(ols_foster_result.resid))

    # p - is the auto-regressive part of the model. Allows to incorporate the effect of past values into the model.
    # Intuitively, this would be similar to stating that it is likely to be warm tomorrow
    # if it has been warm the past 3 days.

    # d - is the integrated part of the model. This includes terms in the model that incorporate the amount
    # of differencing, (i.e. the number of past time points to subtract from the current value).
    # Intuitively, this would be similar to stating that it is likely to be same temperature tomorrow
    # if the difference in temperature in the last three days has been very small.
    # d - is the number of nonseasonal terms needed for stationarity

    # q - is the moving average part of the model. This allows us to set the error of our model
    # as a linear combination of the error values observed at previous time points in the past.
    # q is the number of lagged forecast errors in the prediction equation.

    # model M1.1 Brown & Rozeff (1979)
    # E[Y] = Y(t-4) + \phi_1 * [Y(t-1) -Y(t-5)] + \Theta_1 * \alpha_{t-4} + \delta
    # \Theta_1 - seasonal  moving average parameter
    # \theta - moving average parameter
    # \delta - constant term in ARIMA models
    # M1.1 is (1,0,0) x (0, 1, 1)_4 specification, i.e. (p, d, q) x (P,D,Q)_s where s is the periodicity of the time series

    eps_intc.rename(columns={'VALUE': 'EPS'}, inplace=True)

    eps_intc.plot(figsize=(12, 8),
                  title="Intel Corporation (INTC) Earnings Per Share (EPS)")

    # check for auto-correlation
    # Another popular test for serial correlation is the Durbin-Watson statistic.
    # The DW statistic will lie in the 0-4 range, with a value near two indicating
    # no first-order serial correlation. Positive serial correlation is associated
    # with DW values below 2 and negative serial correlation with DW values above 2.
    print("Durbin-Watson test for original time series of Intel Corporation EPS")
    print(sm_api.stats.durbin_watson(eps_intc))
    # The value of Durbin-Watson statistic is close to 2 if the errors are uncorrelated.
    # In our example, it is 0.06935. That means that there is a strong evidence that
    # the variable EPS has high auto-correlation.

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm_api.graphics.tsa.plot_acf(eps_intc.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm_api.graphics.tsa.plot_pacf(eps_intc, lags=40, ax=ax2)

    brown_rozeff_model = sm_api.tsa.statespace.SARIMAX(eps_intc,
                                                       order=(1, 0, 0),
                                                       seasonal_order=(0, 1, 1, 4),
                                                       enforce_stationarity=False,
                                                       enforce_invertibility=False).fit()

    print("M1.1 Brown & Rozeff (1979)")
    print(brown_rozeff_model.summary())
    print("Normal residuals?")
    print(scipy.stats.normaltest(brown_rozeff_model.resid))

    # visualize fit quality of SARIMAX model
    brown_rozeff_model.plot_diagnostics(figsize=(15, 12))
    plt.show()

    # Does our model obey the theory? We will use the Durbin-Watson test for auto-correlation.
    # The Durbin-Watson statistic ranges in value from 0 to 4.
    # A value near 2 indicates non-auto-correlation;
    # a value toward 0 indicates positive auto-correlation;
    # a value toward 4 indicates negative auto-correlation.
    # Durbin-Watson test for residuals

    if False:
        # train neural network
        train_neural_network(eps_intc)

    # model M1.4 Lawrence and Rozeff (1979)
    # E[Y(t)] = \alpha  + \beta_1 *Y(t - 1) + \beta_2 * Y(t - 4) + \epsilon_t

    # Time Series Analysis
    # https://bicorner.com/2015/11/16/time-series-analysis-using-ipython/

    # Guide to forecasting with ARIMA
    # https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
    # SARIMAX - Seasonal AutoRegressive Integrated Moving Averages with exogenous regressors


    # capital expenditures time series for Intel Corporation
    cpx_intc = all_intc_ibes[all_intc_ibes['MEASURE'] == 'CPX']
    print("Cap-Ex Intel Corporation")
    print(cpx_intc.head())

    # an alternative to the above is:
    # df_ibes = df_ibes.set_index(['TICKER','ANNDATS_ACT'])

    # ACTDATS_ACT	ACTTIMS_ACT - actual date and time of annoucement
    # ACTUAL - actual reported value
    # ANALYS -  The name of the individual or department at the research organization
    # providing forecast data to Thomson Financial.
    # ACTDATS - Indicates the date an estimate was saved to the Thomson Financial database

    # Announce Date (ANNDATS, ANNTIMS) - Indicates the effective date of an estimate as determined by the contributor.
    # Contributor - The name of the research organization providing data to Thomson Financial.
    # Announcement Date (ACTDATS_ACT, ACTTIMS_ACT) The date on which the company reported their earnings for the fiscal period indicated.
    if not(df_ibes is None):
        aapl_ibes = df_ibes.loc[(df_ibes['TICKER'] == 'AAPLE') &
                           (df_ibes['MEASURE'] == 'EPS') &
                           (df_ibes['PDICITY'] == 'QTR')]

        aapl_eps = aapl_ibes['MEASURE']
        print(aapl_eps.head())


    # will not use this data set because market cap data is not provided
    # Also, WRDS Financial Ratios does not have required data as per paper we are working off
    # Namely, AR, INV, SA, are missing
    # Keep for now in case we come back to it
    use_wrds_fin_ratios = False
    if use_wrds_fin_ratios:
        path = '../WRDS_data/Financial_Ratios/'
        file_name = 'WRDS_fin_ratios_spx500_constituents_2005_01_2015_12.csv'

        # invt_act - Inventories as a fraction of Current Assets
        # rect_act - Accounts Receivables as a fraction of Current Assets
        # efftax -  Income Tax as a fraction of Pretax Income
        # GProf - Gross Profitability as a fraction of Total Assets
        # gpm - Gross Profit as a fraction of Sales
        # CAPEI - Multiple of Market Value of Equity to 5-year moving average of Net Income
        # pe_exi - Price-to-Earnings, excl. Extraordinary Items (diluted)
        # bm - Book Value of Equity as a fraction of Market Value of Equity
        # cash_lt - Cash Balance as a fraction of Total Liabilities
        # de_ratio - Total Liabilities to Shareholders’ Equity (common and preferred),
        # de_ratio  - ratio of Total Debt / Equity, i.e. solvency
        df_finratios = pd.read_csv(os.path.join(path, file_name),
                                    date_parser=lambda x: None if pd.isnull(x) else pd.datetime.strptime(x, '%Y/%m/%d'),
                                    parse_dates=['adate', 'qdate', 'public_date'],
                                    index_col=0)

        aapl = df_finratios.loc[(df_finratios['co_tic'] == 'AAPL'),
                                 ('CAPEI', 'pe_exi', 'de_ratio', 'co_tic',
                                  'qdate', 'public_date', 'from', 'thru', 'bm', 'cash_lt')]

        aapl = aapl.set_index(['public_date'])
        aapl_pe_exi = aapl['pe_exi']
        rollmean = aapl_pe_exi.rolling(window=4, center=False).mean()
        plt.plot(rollmean)
    print('Done')

def testLoadIntradayLarge():
    '''
    Ignore this for now
    Does not work, the file turned out to be too big
    Must untar first and then read file by file to assemble a larger data frame
    :return:
    '''
    tar_file_path = '/Users/volodymyrchernat/Documents/NYU/ML_IH/intrady_spx500_data/stockdata.tar.gz'
    dest_dir = '/Users/volodymyrchernat/Documents/NYU/ML_IH/intrady_spx500_data'
    with tarfile.open(tar_file_path, 'r') as tar_fd:
        # tar_fd.list()
        do_process = True
        if do_process:
            return tar_fd.extract('data/allstocks_20010628/table_aapl.csv')
            for tar_info in tar_fd.getmembers():
               return tar_fd.extract(tar_info)
    print('done')


def nn_eps_via_tensorflow():
    '''
    Train forward feeding neural network using TensorFlow
    :return:
    '''
    import tensorflow as tf

    ########################### common code to massage data #######################
    # I/B/E/S Thompson Reuters Data Set
    print(os.getcwd())
    path = '../WRDS_data/IBES'
    load_all_tickers = False
    df_ibes = None
    if load_all_tickers:
        # VC: this is too large of a file, load per ticker
        file_name = 'IBES_Actuals_FinRatios_csv_01Jan1990 - 31Mar2017.csv'
        df_ibes = pd.read_csv(os.path.join(path, file_name),
                              date_parser=lambda dt, tm:
                              pd.to_datetime(dt + ' ' + tm, format='%Y/%m/%d %H:%M:%S'),
                              parse_dates=[['ACTDATS', 'ACTTIMS'], ['ANNDATS', 'ANNTIMS']],
                              index_col=0)

        df_ibes.set_index(['TICKER', 'ACTDATS', 'ACTTIMS'], inplace=True)

        all_intc_ibes = df_ibes[df_ibes['TICKER'] == 'INTC']
    else:
        file_name = 'IBES_Actuals_INTC_01Jan1990 - 31Mar2017.csv'
        all_intc_ibes = pd.read_csv(os.path.join(path, file_name),
                                    date_parser=lambda dt, tm:
                                    pd.to_datetime(dt + ' ' + tm, format='%Y/%m/%d %H:%M:%S'),
                                    parse_dates=[['ACTDATS', 'ACTTIMS'], ['ANNDATS', 'ANNTIMS']],
                                    index_col=0)

    # Earnings per Share (EPS) for Intel Corporation (INTC)
    eps_intc = all_intc_ibes[all_intc_ibes['MEASURE'] == 'EPS']
    intc_eps_summary = eps_intc.groupby(['TICKER', eps_intc.index]).agg(
        {'VALUE': 'mean', 'ANNDATS_ANNTIMS': ['min', 'max', 'count']})

    intc_eps_summary.head(10)

    # pandas multi-index
    # https://pandas.pydata.org/pandas-docs/stable/cookbook.html#levels
    eps_intc.reset_index(level=0, inplace=True)
    eps_intc = eps_intc.drop(['OFTIC', 'TICKER', 'CUSIP',
                              'PENDS', 'MEASURE', 'PDICITY', 'ANNDATS_ANNTIMS', 'CURR_ACT'], axis=1)

    # set index so that eps_intc can be used with statespace.SARIMAX
    eps_intc.set_index(['ACTDATS_ACTTIMS'], inplace=True)

    eps_tminus1 = eps_intc.shift(1)
    eps_tminus2 = eps_intc.shift(2)
    eps_tminus3 = eps_intc.shift(3)
    eps_tminus4 = eps_intc.shift(4)
    eps_tminus1.rename(columns={'VALUE': 'EPS_t1'}, inplace=True)
    eps_tminus2.rename(columns={'VALUE': 'EPS_t2'}, inplace=True)
    eps_tminus3.rename(columns={'VALUE': 'EPS_t3'}, inplace=True)
    eps_tminus4.rename(columns={'VALUE': 'EPS_t4'}, inplace=True)
    result = pd.concat([eps_tminus1, eps_tminus2, eps_tminus3, eps_tminus4, eps_intc], axis=1, join='inner')
    result.rename(columns={'VALUE': 'EPS'}, inplace=True)

    result = result.dropna()  # remove rows with NAs so that result could be used with OLS
    # the result can now be used to fit various models
    ###################### end of common code to massage data ###################

    # feature scaling is required for neural networks
    scaled_data = (result - result.mean(axis=0)) / result.std(axis=0)

    learning_rate = 0.01
    n_inputs = 4
    x_place_h = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y_place_h = tf.placeholder(tf.float32, shape=(None,), name="y")

    def neuron_layer(inp_layer, n_neurons, name,  activation_fun=None, add_bias=True):
        '''
        Create layer of neurons using TensorFlow API
        :param inp_layer: input layer of
        :param n_neurons: number of neurons in this layer
        :param name: name of this scope to which neurons from this layer belong to
        :param activation_fun: activation function
        :param add_bias: adds bias if True, and False otherwise
        :return:
        '''
        with tf.name_scope(name):
            n_inputs_this_layer = int(inp_layer.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs_this_layer)
            init_wgts = tf.truncated_normal((n_inputs_this_layer, n_neurons), stddev=stddev)
            wgts = tf.Variable(init_wgts, name="kernel")
            if add_bias:
                b = tf.Variable(tf.zeros([n_neurons]), name="bias")
                z = tf.matmul(inp_layer, wgts) + b
            else:
                z = tf.matmul(inp_layer, wgts)
            if activation_fun is not None:
                return activation_fun(z)
            else:
                return z

    gradient_descent = False
    n_hidden1 = 26
    n_outputs = 1
    with tf.name_scope("feed_fwd_nn"):
        hl1 = neuron_layer(x_place_h, n_hidden1, name="hidden", activation_fun=tf.nn.sigmoid)
        hl1_log = tf.log(hl1)
        outputs = neuron_layer(hl1_log, n_outputs, name="output", activation_fun=None, add_bias=False)
        # compute mean cross entropy over all instances
        loss = tf.reduce_mean(tf.square(outputs - y_place_h))
        if gradient_descent:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # now onto execution phase
    execute_one_at_time = False
    if execute_one_at_time:
        n_epochs = 50
        n_iterations = len(scaled_data.index)
        with tf.Session() as sess:
            init.run()  # initialize all TensorFlow variables
            for epoch in range(n_epochs):
                for iteration in range(n_iterations):
                    one_batch = scaled_data.loc[scaled_data.index[iteration], :].values
                    x_batch, y_batch = np.array([one_batch[0:len(one_batch)-1]]), np.array([one_batch[-1]])
                    sess.run(training_op, feed_dict={x_place_h: x_batch, y_place_h: y_batch})
                if epoch % 10 == 0:
                    mse = loss.eval(feed_dict={x_place_h: x_batch, y_place_h: y_batch})
                    print('epoch  %d, loss = %.6f' % (epoch, mse))
            save_path = saver.save(sess, './feed_fwd_nn_eps.ckpt')
            print('Saved session to %s' % save_path)
    else:
        # train in batches
        n_epochs = 100
        batch_size = 21
        n_batches = int(np.ceil(len(scaled_data.index) / batch_size))
        with tf.Session() as sess:
            init.run()  # initialize all TensorFlow variables
            for epoch in range(n_epochs):
                for batch_idx in range(n_batches):
                    one_batch = scaled_data.loc[scaled_data.index[batch_idx:(batch_idx+1)*batch_size], :].values
                    x_batch, y_batch = np.array(one_batch[:, 0:n_inputs]), one_batch[:, -1]
                    sess.run(training_op, feed_dict={x_place_h: x_batch, y_place_h: y_batch})
                if epoch % 10 == 0:
                    mse = loss.eval(feed_dict={x_place_h: x_batch, y_place_h: y_batch})
                    print('epoch  %d, loss = %.6f' % (epoch, mse))
            save_path = saver.save(sess, './feed_fwd_nn_eps.ckpt')
            print('Saved session to %s' % save_path)


if __name__ == '__main__':
    #combineFinRatios()
    arma_eps()
    #multi_variable_eps()
    #nn_eps_via_tensorflow()