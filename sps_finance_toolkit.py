# Various Import Statements
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import scipy.stats as sp
from scipy.optimize import Bounds, minimize
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State
import statsmodels.api as sm
import statsmodels.stats.moment_helpers as mh
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.distributions.empirical_distribution import ECDF
from numpy.linalg import inv
import math
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, roc_curve, auc
import cvxpy as cp
from sklearn.preprocessing import StandardScaler

import config


def get_df_columns(filename):
    df = pd.read_csv(filename, na_values=-99.99, index_col=0, parse_dates=[0])
    df.dropna(how='all', inplace=True, axis=1)
    df.columns = df.columns.str.strip()
    return df.columns


def get_df(filename, start_period=None, end_period=None, format='%Y%m', reqd_strategies=None, mode='return',
           to_per=False):
    """

    :param filename:
    :param start_period:None if NA
    :param end_period:None if NA
    :param format:
    :param reqd_strategies: None if NA
    :param mode: return or nos or size
    :return:
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=[0], na_values=-99.99)
    if mode == 'return':
        df = df / 100
    df.dropna(how='all', inplace=True, axis=1)
    df.columns = df.columns.str.strip()
    if reqd_strategies is not None:
        df = df[reqd_strategies]
    if to_per:
        df.index = pd.to_datetime(df.index, format=format).to_period('M')
    else:
        df.index = pd.to_datetime(df.index, format=format)
    # if start_period and end_period is not None:
    #     return df[start_period:end_period]
    # else:
    return df[start_period:end_period]


def get_ann_vol(df):
    ann_vol = df.std() * np.sqrt(12)
    return ann_vol


def get_ann_return(df, periodicity=12, expm1=True):
    if isinstance(df, (np.ndarray, np.generic)):
        ann_factor = periodicity / df.shape[0]
    else:
        ann_factor = periodicity / len(df.index)
    ann_ret_np = ann_factor * (np.log1p(df).sum()) # using log method for eff computation
    if expm1:
        ann_ret_np = np.expm1(ann_ret_np)
    else:
        ann_ret_np = np.exp(ann_ret_np)
    return ann_ret_np


def get_sharpe_ratio(ann_ret, ann_vol, rf=0.10):
    return (ann_ret - rf) / ann_vol


def get_semi_std(df):
    semi_std = df[df < 0].std(ddof=0)
    return semi_std


def hist_var(col_series, alpha):
    return np.percentile(col_series, alpha * 100)


def para_var(col_series, alpha):
    z = sp.norm.ppf(alpha)
    return col_series.mean() + z * col_series.std(ddof=0)


def corn_var(col_series, alpha):
    z = sp.norm.ppf(alpha)
    kurtosis = sp.kurtosis(col_series, fisher=True)
    skew = sp.skew(col_series)
    z = (z +
         (z ** 2 - 1) * skew / 6 +
         (z ** 3 - 3 * z) * (kurtosis - 3) / 24 -
         (2 * z ** 3 - 5 * z) * (skew ** 2) / 36)
    return col_series.mean() + z * col_series.std(ddof=0)


def get_VaR(df: pd.DataFrame, var_method, alpha):
    if var_method == 'historic':
        return df.aggregate(hist_var, alpha=alpha)
    elif var_method == 'parametric':
        return df.aggregate(para_var, alpha=alpha)
    elif var_method == 'cornish':
        return df.aggregate(corn_var, alpha=alpha)


def get_CVaR(df, VaR):
    CVaR = pd.Series(
        {df.columns[i]: df[df[df.columns[i]] < VaR[i]][df.columns[i]].mean() for i in range(len(df.columns))})
    return CVaR


def add_1(ddf):
    return ddf + 1


def drawdown(df: pd.DataFrame, retrive_index=False, init_wealth=1000, is1p=True):
    if retrive_index:
        if is1p:
            factor = np.exp(np.cumsum(np.log(df)))  # using log instead of cumprod for efficiency
        else:
            factor = np.exp(np.cumsum(np.log1p(df)))  # using log instead of cumprod for efficiency
        wealth_index = init_wealth * factor
        return wealth_index
    factor = np.exp(np.cumsum(np.log1p(df)))  # using log instead of cumprod for efficiency
    wealth_index = init_wealth * factor
    prev_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - prev_peaks) / prev_peaks
    return [wealth_index, prev_peaks, drawdowns]


def risk_info(df, risk_plot=['ann_ret'], rf=0.03, alpha=0.05, var_method='cornish', only_sharpe=False):
    ann_vol = get_ann_vol(df)
    ann_ret = get_ann_return(df)
    sharpe_ratio = get_sharpe_ratio(ann_ret, ann_vol, rf)
    if only_sharpe:
        return sharpe_ratio
    semi_std = get_semi_std(df)
    kurtosis = sp.kurtosis(df, fisher=True)
    skew = sp.skew(df)
    VaR = get_VaR(df, var_method, alpha)
    CVaR = get_CVaR(df, VaR)
    drawdown_df = drawdown(df)[2]
    drawdown_df = drawdown_df.aggregate(lambda col_series: col_series.min())
    info = pd.DataFrame({'ann_ret': ann_ret,
                         'ann_vol': ann_vol,
                         'sharpe_ratio': sharpe_ratio,
                         'semi_dev': semi_std,
                         'Kurtosis': kurtosis,
                         'Skew': skew,
                         'VaR': VaR,
                         'CVaR': CVaR,
                         'Drawdown': drawdown_df})
    return info.sort_values(by=risk_plot, ascending=True).transpose()


def terminal_risk_stats(fv, floor_factor, wealth_index, aslst=False, strategy=None):
    floor_value = fv * floor_factor
    if isinstance(wealth_index, pd.DataFrame):
        terminal_wealth = wealth_index.iloc[-1]
    else:
        terminal_wealth = wealth_index  # The terminal row
    n_scenarios = terminal_wealth.shape[0]
    exp_wealth = np.mean(terminal_wealth)
    med_wealth = np.median(terminal_wealth)
    vol_wealth = np.std(terminal_wealth)
    failure_mask = np.less(terminal_wealth, floor_value)
    n_breaches = failure_mask.sum()
    p_breaches = n_breaches / n_scenarios
    # exp_loss_post_breach = np.mean(terminal_wealth[failure_mask]) if n_breaches > 0 else 0.0
    # exp_shortfall1 = floor_value - exp_loss_post_breach if n_breaches > 0 else 0.0
    exp_shortfall = np.dot(floor_value - terminal_wealth, failure_mask) / n_breaches if n_breaches > 0 else 0.0
    best_case = terminal_wealth.max()
    worst_case = terminal_wealth.min()
    if aslst:
        stats = [strategy, exp_wealth, vol_wealth, med_wealth, n_breaches, p_breaches, exp_shortfall]
        return stats
    else:
        return '''
                Mean: ${:.2f}\n
                Median: ${:.2f}\n
                Violations: {} ({:.2%})\n
                Exp Shortfall: ${:.2f}\n
                Diff in worst and best case scenario: {}\n
                Worst Case: {}
                '''.format(exp_wealth, med_wealth, n_breaches, p_breaches, exp_shortfall, best_case - worst_case,
                           worst_case)


def ren_df(df, rev_name, exis_name='index'):
    return df.reset_index().rename(columns={exis_name: rev_name})


def get_cov(df, periodicity=12):
    return df.cov()*periodicity


def get_pf_ret(wt_array, ret_array):
    return wt_array.T @ ret_array


def get_pf_vol(wt_array, cov_mat):
    return (wt_array.T @ cov_mat @ wt_array) ** 0.5


def annualize_pf_vol(pf_vol, periodicity):
    return pf_vol * np.sqrt(periodicity)


def format_perc(wts):
    return '{:.4%}'.format(wts)


def get_hover_info(n_assets, reqd_strategies, wts_list):
    hoverinfo = []
    pf_alloc_wts_str = [list(map(format_perc, wt_array)) for wt_array in wts_list]
    for pf_alloc in pf_alloc_wts_str:
        hovertext = ''
        for i in range(n_assets):
            hovertext += ('{}: {} \n'.format(reqd_strategies[i], pf_alloc[i]))
        hoverinfo.append(hovertext)
    return hoverinfo


def optimize_wts(ret_series, cov_mat, n_points):
    wts_list = []
    n_assets = ret_series.shape[0]
    ret_array = ret_series.to_numpy()
    init_guess = np.repeat(1 / n_assets, n_assets)
    bounds = Bounds(lb=0.0, ub=1.0)
    is_tgt_met = {
        'type': 'eq',
        'args': (ret_array,),
        'fun': lambda wt_array, ret_array: get_pf_ret(wt_array, ret_array) - tgt_ret
    }
    wts_sum_to_1 = {
        'type': 'eq',
        'fun': lambda wt_array: np.sum(wt_array) - 1
    }
    for tgt_ret in np.linspace(ret_series.min(), ret_series.max(), n_points):
        results = minimize(fun=get_pf_vol,
                           args=(cov_mat,),
                           method='SLSQP',
                           bounds=bounds,
                           constraints=[is_tgt_met, wts_sum_to_1],
                           x0=init_guess,
                           options={'disp': False})
        wts_list.append(results.x)
    return wts_list


def get_mean_var_pts(ret_series, cov_df, n_points, reqd_strategies):
    n_assets = ret_series.shape[0]
    ret_array = ret_series.to_numpy()
    cov_mat = cov_df.to_numpy()
    wts_list = optimize_wts(ret_series, cov_mat, n_points)
    pf_ret = [get_pf_ret(wt_array, ret_array) for wt_array in wts_list]
    pf_vol = [annualize_pf_vol(get_pf_vol(wt_array, cov_mat), 12) for wt_array in wts_list]
    hover_desc = get_hover_info(n_assets, reqd_strategies, wts_list)
    mean_var_df = pd.DataFrame({'Returns': pf_ret,
                                'Volatility': pf_vol,
                                'Hover Description': hover_desc})
    return mean_var_df


def get_msr(ret_series, cov_df, rf, reqd_strategies, gmv=False, onlywts=False):
    if onlywts:  # if called from bt_roll and gmv weighting schemes
        n_assets = len(ret_series.columns)
    else:
        n_assets = ret_series.shape[0]
    if gmv:
        ret_array = np.repeat(1.0,
                              n_assets)  # for gmv wts to be independent of E(R) and thus minimisation function tries to manipulate volatility to minimioze -ve SR
    else:
        ret_array = ret_series.to_numpy()
    cov_mat = cov_df.to_numpy()
    bounds = Bounds(lb=0.0, ub=1.0)
    init_guess = np.repeat(1 / n_assets, n_assets)
    sum_wts_to_1 = {
        'type': 'eq',
        'fun': lambda wt_array: np.sum(wt_array) - 1
    }

    def neg_msr(wt_array, ret_array, cov_mat, rf):
        return -(get_pf_ret(wt_array, ret_array) - rf) / get_pf_vol(wt_array, cov_mat)

    results = minimize(fun=neg_msr,
                       args=(ret_array, cov_mat, rf,),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=[sum_wts_to_1],
                       options={'disp': False},
                       x0=init_guess)
    msr_wt_array = results.x
    if onlywts:
        return msr_wt_array
    if gmv:
        ret_array = ret_series.to_numpy()  # ret_series restored for calculating mean_var pts using optimized weights (optimized independent of E(R))
    msr_ret = get_pf_ret(msr_wt_array, ret_array)
    msr_vol = annualize_pf_vol(get_pf_vol(msr_wt_array, cov_mat), 12)
    hover_desc = get_hover_info(n_assets, reqd_strategies, [msr_wt_array])[0]
    return [msr_vol, msr_ret, hover_desc, msr_wt_array]


def get_gmv(ret_series, cov_df, rf, reqd_strategies, onlywts=False):
    return get_msr(ret_series, cov_df, rf, reqd_strategies, gmv=True, onlywts=onlywts)


def get_eq(ret_series, cov_df, reqd_strategies):
    n_assets = ret_series.shape[0]
    ret_array = ret_series.to_numpy()
    cov_mat = cov_df.to_numpy()
    eq_wt_array = np.repeat(1 / n_assets, n_assets)
    eq_ret = get_pf_ret(eq_wt_array, ret_array)
    eq_vol = annualize_pf_vol(get_pf_vol(eq_wt_array, cov_mat), 12)
    hover_desc = get_hover_info(n_assets, reqd_strategies, [eq_wt_array])[0]
    return [eq_vol, eq_ret, hover_desc]


def get_corr_mat(df, window):
    """

    :param df:
    :return: -> gives correlation matrix for each block of window period and mean correlations
    """
    corr_mat = df.rolling(window=window).corr().dropna(how='all', axis=0)
    corr_mat.index.names = ['Date', 'Sector']
    corr_groupings = corr_mat.groupby(level='Date')
    corr_series = corr_groupings.apply(lambda corr_mat: corr_mat.values.mean())  # getting mean corr for corr_mat for each date (each date being groupedby)
    return [corr_mat, corr_series]


def cipp_algo(risky_ret_df: pd.DataFrame, multiplier, floor: float, reqd_strategies, poi, alpha, var_method, rf=0.03,
              max_draw_mode=False, plot=True, s0=1000, gbm=False):
    def repl_shape(new_df: pd.DataFrame, tgt_df: pd.DataFrame):
        return new_df.reindex_like(tgt_df)

    init_wealth = s0
    pf_value = init_wealth
    prev_peak = init_wealth
    floor_value = init_wealth * floor
    riskfree_df = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_ret_history = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_value_history = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_risky_wt_history = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_cushion_history = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_floor_history = repl_shape(pd.DataFrame(), risky_ret_df)
    riskfree_df[:] = rf / 12

    for dt_index in range(len(risky_ret_df.index)):
        if max_draw_mode:
            prev_peak = np.maximum(prev_peak, pf_value)
            floor_value = prev_peak * floor
        cushion = (pf_value - floor_value) / pf_value
        risky_wt = multiplier * cushion
        risky_wt = np.maximum(risky_wt, 0)
        risky_wt = np.minimum(risky_wt, 1)
        rf_wt = 1 - risky_wt
        cppi_pf_rt = (risky_wt * risky_ret_df.iloc[dt_index]) + (rf_wt * riskfree_df.iloc[dt_index])
        pf_value = (cppi_pf_rt + 1) * pf_value

        # create logs
        cppi_ret_history.iloc[dt_index] = cppi_pf_rt.transpose()
        cppi_value_history.iloc[dt_index] = pf_value.transpose()
        cppi_cushion_history.iloc[dt_index] = cushion
        cppi_floor_history.iloc[dt_index] = floor_value
        cppi_risky_wt_history.iloc[dt_index] = risky_wt

    if gbm:
        return cppi_value_history
    # plot wealth index, drawdowns cushions and weights
    app = dash.Dash()
    temp_risky_ret = drawdown(risky_ret_df)
    risky_wealth = temp_risky_ret[0]
    risky_drawdown = temp_risky_ret[2]
    cppi_drawdown = drawdown(cppi_ret_history)[2]
    cppi_wealth_plot = go.Scatter(x=cppi_value_history.index,
                                  y=cppi_value_history[poi],
                                  name='cppi_wealth_index',
                                  text=(cppi_risky_wt_history[poi] * 100).round(decimals=2))
    cppi_drawdown_plot = go.Scatter(x=cppi_drawdown.index,
                                    y=cppi_drawdown[poi],
                                    name='cppi_drawdown')
    cppi_wt_plot = go.Scatter(x=cppi_risky_wt_history.index,
                              y=cppi_risky_wt_history[poi],
                              name='cppi-risky-asset-alloc')
    risky_wealth_plot = go.Scatter(x=risky_wealth.index,
                                   y=risky_wealth[poi],
                                   name='risky_wealth_index')
    risky_drawdown_plot = go.Scatter(x=risky_drawdown.index,
                                     y=risky_drawdown[poi],
                                     name='risky_drawdown')
    floor_plot = go.Scatter(x=cppi_floor_history.index,
                            y=cppi_floor_history[poi],
                            mode='lines',
                            line=dict(dash='dashdot',
                                      width=3),
                            name='Floor')
    lowpt_cppi_drawdown = cppi_drawdown[poi].min()
    lowpt_cppi_drawdown_year = cppi_drawdown[poi].idxmin()
    lowpt_risky_drawdown = risky_drawdown[poi].min()
    lowpt_risky_drawdown_year = risky_drawdown[poi].idxmin()
    lowpts = [[lowpt_cppi_drawdown, lowpt_cppi_drawdown_year], [lowpt_risky_drawdown, lowpt_risky_drawdown_year]]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(cppi_wealth_plot, row=1, col=1)
    fig.add_trace(risky_wealth_plot, row=1, col=1)
    fig.add_trace(floor_plot, row=1, col=1)
    fig.add_trace(cppi_drawdown_plot, row=2, col=1)
    fig.add_trace(risky_drawdown_plot, row=2, col=1)
    fig.add_trace(cppi_wt_plot, row=3, col=1)
    fig.update_layout(height=750)
    annotations = [dict(x=i[1],
                        y=i[0],
                        ax=0,
                        ay=50,
                        xref='x',
                        yref='y2',
                        arrowhead=7,
                        showarrow=True,
                        text='Max DrawDown is {} and occurred at {}'.format(i[0], i[1])) for i in lowpts]
    fig.update_layout(annotations=annotations)
    app.layout = html.Div([dcc.Graph(id='drawdowns', figure=fig)])
    if __name__ != '__main__' and plot:
        app.run_server()

    # create a result dataframe
    backtest_results = {'cppi_wealth': cppi_value_history,
                        'cppi_return': cppi_ret_history,
                        'cppi_drawdown': cppi_drawdown,
                        'cppi_risky_wts': cppi_risky_wt_history,
                        'risky_wealth': risky_wealth,
                        'floor': floor_value,
                        'risky_drawdown': risky_drawdown}
    return backtest_results


def plot_eff_frontier(ret_series: pd.Series, cov_df: pd.DataFrame, n_points: int, reqd_strategies, rf, show_msr=True,
                      show_eq=False, show_gmv=True):
    mean_var_df = get_mean_var_pts(ret_series, cov_df, n_points, reqd_strategies)
    msr = get_msr(ret_series, cov_df, rf, reqd_strategies)
    eq = get_eq(ret_series, cov_df, reqd_strategies)
    gmv = get_gmv(ret_series, cov_df, rf, reqd_strategies)
    app = dash.Dash()
    data = [go.Scatter(x=mean_var_df['Volatility'],
                       y=mean_var_df['Returns'],
                       mode='markers+lines',
                       name='efficient_frontier',
                       text=mean_var_df['Hover Description'])]
    if show_msr:
        data.append(go.Scatter(x=[0, msr[0]],
                               y=[rf, msr[1]],
                               mode='markers+lines',
                               name='CML',
                               text=['RF - 100%', msr[2]]))
    if show_eq:
        data.append(go.Scatter(x=[eq[0]],
                               y=[eq[1]],
                               mode='markers',
                               name='EQ',
                               text=[eq[2]]))
    if show_gmv:
        data.append(go.Scatter(go.Scatter(x=[gmv[0]],
                                          y=[gmv[1]],
                                          mode='markers',
                                          name='GMV',
                                          text=[gmv[2]])))

    app.layout = html.Div([html.Div([dcc.Graph(id='eff_frontier', figure=dict(data=data,
                                                                              layout=go.Layout(
                                                                                  title='Efficient Frontier',
                                                                                  xaxis=dict(title='Variance'),
                                                                                  yaxis=dict(title='mean'),
                                                                                  hovermode='closest')))]),
                           html.Div([html.Pre(id='display_info')])])

    @app.callback(Output('display_info', 'children'),
                  [Input('eff_frontier', 'hoverData')])
    def upd_markdown(hover_data):
        hover_data = hover_data['points'][0]
        wts_data = hover_data['text']
        pf_vol_data = hover_data['x']
        pf_ret_data = hover_data['y']
        disp_data = '''
            The weights are \n{}
            PF - Volatility: {:.2%}
            PF - Return    : {:.2%}
        '''.format(wts_data, pf_vol_data, pf_ret_data)
        return disp_data

    if __name__ != '__main__':
        app.run_server()


def plot_corr_mktret(ind_ret_filename, n_firms_filename, size_filename, start_period, end_period, format,
                     reqd_strategies=None, window=36, retrieve_mcw=False, to_per=False, retrieve_mkt_cap_wts=False):
    app = dash.Dash()
    # Populate all reqd dataframes
    ind_ret_m_df = get_df(ind_ret_filename, start_period, end_period, format, reqd_strategies, mode='return',
                          to_per=to_per)
    ind_n_firms_df = get_df(n_firms_filename, start_period, end_period, format, reqd_strategies, mode='nos',
                            to_per=to_per)
    ind_size_df = get_df(size_filename, start_period, end_period, format, reqd_strategies, mode='size', to_per=to_per)

    # industry returns --> mkt cap returns for index constructions
    mkt_cap_df = ind_n_firms_df * ind_size_df
    total_mkt_cap_series = mkt_cap_df.sum(axis=1)
    mkt_wts_df = mkt_cap_df.divide(total_mkt_cap_series, axis=0)
    if retrieve_mkt_cap_wts:
        return mkt_wts_df
    mcw = ind_ret_m_df * mkt_wts_df
    mcw_ret_df = pd.DataFrame({'mkt_cap_wt_ret_monthly': mcw.sum(axis=1)})
    if retrieve_mcw:
        return mcw_ret_df

    # index_generation
    mcw_index = drawdown(mcw_ret_df)[0]
    # mcw_index_36MA = mcw_index.rolling(window=window).mean()

    # rolling returns
    mcw_rolling_returns = mcw_ret_df.rolling(window=window).aggregate(get_ann_return)

    # corr matrix
    corr_results = get_corr_mat(mcw, window=window)
    corr_series = corr_results[1]

    # plots
    # ret_data = go.Scatter(x=mcw_ret_df.index,
    #                       y=mcw_ret_df['mkt_cap_wt_ret_monthly'],
    #                       mode='lines',
    #                       name='mcw_returns')
    roll_ret_data = go.Scatter(x=mcw_rolling_returns.index,
                               y=mcw_rolling_returns['mkt_cap_wt_ret_monthly'],
                               mode='lines',
                               name='roll_returns')
    roll_corr_data = go.Scatter(x=corr_series.index,
                                y=corr_series,
                                mode='lines',
                                name='roll_corr',
                                yaxis='y2')
    # index_data = go.Scatter(x=mcw_index.index,
    #                         y=mcw_index['mkt_cap_wt_ret_monthly'],
    #                         mode='lines',
    #                         name='index')
    # ma_data = go.Scatter(x=mcw_index_36MA.index,
    #                      y=mcw_index_36MA['mkt_cap_wt_ret_monthly'],
    #                      mode='lines',
    #                      name='ma_index',
    #                      yaxis='y2')
    layout = go.Layout(yaxis=dict(title='roll_return'),
                       yaxis2=dict(side='right',
                                   overlaying='y1',
                                   title='roll_corr'),
                       hovermode='closest')
    app.layout = html.Div([dcc.Graph(id='corr', figure=dict(data=[roll_ret_data, roll_corr_data], layout=layout))])

    if __name__ != '__main__':
        app.run_server()


def plot(df, mode, reqd_strategies: list, risk_plot: list, poi, var_method, alpha, rf):
    alpha = alpha / 100
    app = dash.Dash()
    infodf = risk_info(df, risk_plot=risk_plot, rf=rf, alpha=alpha, var_method=var_method)
    idx = reqd_strategies.index(poi)
    if mode == 'returns' or mode == 'downside':
        hist_plot = [df[col] for col in df.columns]
        group_labels = df.columns
        fig = ff.create_distplot(hist_plot, group_labels, show_hist=False)
        if mode == 'downside':
            var_annotation_x = infodf.loc['VaR'][idx]
            cvar_annotation_x = infodf.loc['CVaR'][idx]
            annotations = [dict(x=var_annotation_x,
                                y=0,
                                ax=0,
                                ay=-200,
                                showarrow=True,
                                arrowhead=7,
                                text='Min {} probability for {} % loss'.format(alpha, -(var_annotation_x * 100).round(
                                    decimals=4)),
                                xref='x',
                                yref='y'),
                           dict(x=cvar_annotation_x,
                                y=0,
                                ax=0,
                                ay=-100,
                                showarrow=True,
                                arrowhead=7,
                                text='Expected loss is {} %'.format(-(cvar_annotation_x * 100).round(decimals=4)),
                                xref='x',
                                yref='y')
                           ]
            fig.update_layout(annotations=annotations)
        app.layout = html.Div([dcc.Graph(id='returns', figure=fig)])

    elif mode == 'risk_stats':
        infodf = ren_df(infodf, 'risk_params', 'index')
        app.layout = dt.DataTable(id='risk-stats',
                                  columns=[{'name': col,
                                            'id': col} for col in infodf.columns],
                                  data=infodf.to_dict('records'))

    elif mode == 'risk_plot':
        data = [go.Bar(x=infodf.columns,
                       y=infodf.loc[risk_type],
                       name=risk_type) for risk_type in risk_plot]
        app.layout = html.Div([dcc.Graph(id='risk_plots', figure=dict(data=data))])

    elif mode == 'drawdowns':
        all_index = drawdown(df)
        ddf = all_index[2][reqd_strategies[idx]]
        wdf = all_index[0][reqd_strategies[idx]]
        pdf = all_index[1][reqd_strategies[idx]]
        wealth_plot = go.Scatter(x=df.index,
                                 y=wdf,
                                 name='wealth_index')
        peak_plot = go.Scatter(x=df.index,
                               y=pdf,
                               name='peak_index')
        draw_plot = go.Scatter(x=df.index,
                               y=ddf,
                               name='drawdown')
        lowpt_drawdown = ddf.min()
        lowpt_drawdown_year = ddf.idxmin()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(wealth_plot, row=1, col=1)
        fig.add_trace(peak_plot, row=1, col=1)
        fig.add_trace(draw_plot, row=2, col=1)
        annotations = [dict(x=lowpt_drawdown_year,
                            y=lowpt_drawdown,
                            ax=0,
                            ay=150,
                            xref='x',
                            yref='y2',
                            arrowhead=7,
                            showarrow=True,
                            text='Max DrawDown is {} and occurred at {}'.format(lowpt_drawdown, lowpt_drawdown_year))]
        fig.update_layout(annotations=annotations)
        app.layout = html.Div([dcc.Graph(id='drawdowns', figure=fig)])


    else:
        app.layout = html.Div([dcc.Markdown(id='test', children='Hi')])
    if __name__ != '__main__':
        app.run_server()


def gbm_stock(s0, n_scenarios, steps_per_yr, n_years, er, vol, floor, multiplier, rf, cppi, ret_series=False):
    floor_value = floor * s0
    dt = 1 / steps_per_yr
    total_time_steps = int(n_years * steps_per_yr) + 1

    # Using refined method
    dz = np.random.normal(loc=(1 + er) ** dt, scale=vol * np.sqrt(dt),
                          size=(total_time_steps, n_scenarios))
    # mu and sigma is annualized.
    # The drift and rw terms require mu and sigma for the infinitesimally small time.
    # Even better to use continuous comp ret.
    # eg dt = 0.25 and mu is 10% per year. So drift term for 1Qtr needs mu for such qtr viz (1.1)**0.25
    gbm_df = pd.DataFrame(dz)
    gbm_df.loc[0] = 1.0
    if cppi:
        gbm_df = gbm_df.apply(lambda gbm_rets: gbm_rets - 1)
        wealth_index = cipp_algo(gbm_df, multiplier=multiplier, floor=floor, reqd_strategies=[''], poi='', alpha='',
                                 var_method='', rf=rf, s0=s0, gbm=True)
    else:
        wealth_index = drawdown(gbm_df, retrive_index=True, init_wealth=s0)
    if ret_series:
        gbm_df.drop(0, inplace=True)
        return gbm_df
    return wealth_index


def plot_gbm(s0=100):
    # plot
    app = dash.Dash()
    app.layout = html.Div(
        [html.Div([html.Label(id='l_sce', children='N-Scenarios: '), dcc.Input(id='i_sce', type='number', value=10)]),
         html.Div([html.Label(id='l_st/yr', children='N-Steps per year: '),
                   dcc.Input(id='i_st/yr', type='number', value=12)]),
         html.Div([html.Label(id='l_yr', children='N-Years: '), dcc.Input(id='i_yr', type='number', value=10)]),
         html.Div([html.Label(id='l_er', children='Expected Return: '),
                   dcc.Input(id='i_er', type='number', value=0.07, step=0.005)]),
         html.Div([html.Label(id='l_vol', children='Expected Volatility: '),
                   dcc.Input(id='i_vol', type='number', value=0.15, step=0.005)]),
         html.Div([html.Label(id='l_floor', children='Floor: '),
                   dcc.Input(id='i_floor', type='number', value=0.8, step=0.1)]),
         html.Div([html.Label(id='l_multi', children='Multiplier: '), dcc.Input(id='i_multi', type='number', value=3)]),
         html.Div([html.Label(id='l_rf', children='Risk Free Rate: '),
                   dcc.Input(id='i_rf', type='number', value=0.03, step=0.005)]),
         html.Div([dcc.RadioItems(id='cppi', options=[{'label': 'CPPI?', 'value': 1}, {'label': 'Risky?', 'value': 0}],
                                  value=1)]),
         html.Div([html.Button(id='gen_gbm', children='Generate', n_clicks=0)]),
         html.Div([dcc.Graph(id='gbm_plot')]),
         html.Div([dcc.Markdown(id='gbm_stats')], style={'fontsize': '40em'})])

    @app.callback(Output('gbm_stats', 'children'),
                  [Input('gen_gbm', 'n_clicks')],
                  [State('i_sce', 'value'),
                   State('i_st/yr', 'value'),
                   State('i_yr', 'value'),
                   State('i_er', 'value'),
                   State('i_vol', 'value'),
                   State('i_floor', 'value'),
                   State('i_multi', 'value'),
                   State('i_rf', 'value'),
                   State('cppi', 'value')])
    def update_gbm(n_clicks, n_scenarios, steps_per_yr, n_years, er, vol, floor, multiplier, rf, cppi):
        wealth_index = gbm_stock(s0, n_scenarios, steps_per_yr, n_years, er, vol, floor, multiplier, rf, cppi)
        wealth_index.to_csv('tempfile.csv')
        result_stats = terminal_risk_stats(s0, floor, wealth_index)
        return result_stats

    @app.callback(Output('gbm_plot', 'figure'),
                  [Input('gbm_stats', 'children'),
                   Input('i_floor', 'value')])
    def upd_gbm_plot(gbm_stats, floor):
        floor_value = floor * s0
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
        wealth_index = pd.read_csv('tempfile.csv', index_col='Unnamed: 0')
        gbm_motion = wealth_index.aggregate(lambda scenario: go.Scatter(x=scenario.index, y=scenario))
        gbm_motion = gbm_motion.tolist()
        hist_plot = []
        for scenario in wealth_index.columns:
            hist_plot.append(wealth_index[scenario].tolist())
        length = len(hist_plot)
        for i in range(length - 1):
            hist_plot[0].extend(hist_plot[i + 1])
        hist_plot = go.Histogram(y=hist_plot[0],
                                 name='gbm_dist')
        for gbm_data in gbm_motion:
            fig.add_trace(gbm_data, row=1, col=1)
        fig.add_trace(hist_plot, row=1, col=2)  # Hist bin plot
        floor_threshold = [
            dict(type='line', xref='paper', yref='y1', x0=0, x1=1, y0=floor_value, y1=floor_value, name='floor',
                 line=dict(dash='dashdot', width=5))]
        fig.update_layout(showlegend=False,
                          hovermode='y',
                          height=500,
                          shapes=floor_threshold)
        return fig

    if __name__ != '__main__':
        app.run_server()


def get_macaulay_duration(pvf):  # Nota annualized. Make sure to annualize
    mac_dur = pvf.apply(lambda pvf: np.average(pvf.index + 1, weights=pvf))
    return mac_dur


def get_discount_factor(disc_rate: pd.DataFrame, period):
    disc_factors_df = disc_rate.apply(lambda r: np.power((1 + r), -period))
    return disc_factors_df


def get_present_value(cash_flow: pd.Series, disc_rate: pd.DataFrame):
    if not isinstance(disc_rate, pd.DataFrame):
        cash_flow.index -= 1  # To correct for cash_flow.index+1 when called from cir()
        disc_rate = pd.DataFrame(data=[disc_rate for t in cash_flow.index], index=cash_flow.index)
        get_present_value(cash_flow, disc_rate)
    if not len(disc_rate.index) == len(cash_flow.index):
        dr_steps = disc_rate.shape[0]
        cf_steps = cash_flow.shape[0]
        shortfall = cf_steps - dr_steps
        dr_last = disc_rate.iloc[-1]
        append_rate_df = pd.DataFrame(
            data=np.asarray(pd.concat([dr_last] * shortfall, axis=0)).reshape(shortfall, disc_rate.shape[1]),
            index=range(dr_steps, cf_steps, 1))
        disc_rate = disc_rate.append(append_rate_df)
    disc_factors = get_discount_factor(disc_rate, cash_flow.index + 1)
    present_value_factors = disc_factors.apply(lambda disc_factor: disc_factor * cash_flow)
    present_value = present_value_factors.sum()
    mac_dur = get_macaulay_duration(present_value_factors)
    return np.asarray(present_value), mac_dur


def gen_bond_cash_flows(tenor, steps_per_year, cr, fv):
    dt = 1 / steps_per_year
    total_time_steps = int(tenor * steps_per_year)
    periodicity_adj_cr = cr * dt
    coupon_cf = fv * periodicity_adj_cr
    bond_cf = pd.Series([coupon_cf for i in range(0, total_time_steps)])
    bond_cf.iloc[-1] += fv
    return bond_cf


def get_bond_prices(n_years, tenor, steps_per_year, disc_rate, cr=0.03, fv=100):
    dt = 1 / steps_per_year
    if isinstance(disc_rate, pd.DataFrame):
        periodicity_adj_disc_rate = disc_rate * dt
        bond_cf = gen_bond_cash_flows(tenor, steps_per_year, cr, fv)
        bond_prices, mac_dur = get_present_value(bond_cf, periodicity_adj_disc_rate)
        return bond_prices, mac_dur, bond_cf
    else:
        total_time_steps = int(n_years * steps_per_year)
        disc_rate = pd.DataFrame(data=np.repeat(disc_rate, total_time_steps).reshape(total_time_steps, 1))
        return get_bond_prices(n_years, tenor, steps_per_year, disc_rate, cr, fv)


def get_funding_ratio(pv_liabilities, pv_assets):
    return np.divide(pv_assets, pv_liabilities)


def get_terminal_wealth(rets):
    return np.exp(np.log1p(rets).sum())


def cumulate(rets):
    return np.expm1(np.log1p(rets).sum())


def get_optimal_wts(md_liab, ldb, sdb, av, disc_rate, dt):
    x0 = np.repeat(0.5, 2)
    bounds = Bounds(lb=0.00, ub=1.00)

    def core_check_algo(wts, ldb, sdb, av, disc_rate, dt):
        wt_l = wts[0]
        alloc_long_dur_bond = av * wt_l
        alloc_short_dur_bond = av * (1 - wt_l)
        n_long_dur_bond_match = alloc_long_dur_bond / ldb[0]
        n_short_dur_bond_match = alloc_short_dur_bond / sdb[0]
        dur_matched_bond_cf = pd.DataFrame(
            data=pd.concat([ldb[2] * n_long_dur_bond_match, sdb[2] * n_short_dur_bond_match]), columns=['cf'])
        dur_matched_bond_cf = dur_matched_bond_cf.groupby(dur_matched_bond_cf.index)['cf'].sum()
        dur_matched_bond_cf.index += 1
        disc_rate = disc_rate * dt
        pv_pf, mac_dur_pf = get_present_value(dur_matched_bond_cf, disc_rate)
        mac_dur_pf = mac_dur_pf[0]
        return mac_dur_pf * dt

    def check_dur_match(wts, md_liab, ldb, sdb, av, disc_rate, dt):
        mac_dur_pf = core_check_algo(wts, ldb, sdb, av, disc_rate, dt)
        return mac_dur_pf - md_liab

    sum_wts_to_1 = {
        'type': 'eq',
        'fun': lambda wts: np.sum(wts) - 1
    }

    is_diff_zero = {
        'type': 'eq',
        'args': (md_liab, ldb, sdb, av, disc_rate, dt),
        'fun': check_dur_match
    }

    result = minimize(fun=check_dur_match,
                      args=(md_liab, ldb, sdb, av, disc_rate, dt),
                      x0=x0,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=[sum_wts_to_1, is_diff_zero],
                      options={'disp': False}
                      )
    wts = result.x
    return wts


# # NEED TO ADAPT FOR BONDS WITH VARYING COUPON PERIODS
def get_duration_matched_pf(liabilities: pd.Series, n_years: list, steps_per_year: list, disc_rate, cr: list, fv: list,
                            av, fr_change_sim=False):
    pv_liabilities, mac_dur_liabilities = get_present_value(liabilities, disc_rate)
    pv_bond_1, mac_dur_bond_1, bond_cf_1 = get_bond_prices(n_years[0], n_years[0], steps_per_year[0], disc_rate, cr[0],
                                                           fv[0])
    pv_bond_2, mac_dur_bond_2, bond_cf_2 = get_bond_prices(n_years[1], n_years[1], steps_per_year[1], disc_rate, cr[1],
                                                           fv[1])
    # bond_cf_1.index += 1
    # bond_cf_2.index += 1
    mac_dur_bond_1 = mac_dur_bond_1.loc[0] / steps_per_year[0]
    mac_dur_bond_2 = mac_dur_bond_2.loc[0] / steps_per_year[1]
    mac_dur_liabilities = mac_dur_liabilities.loc[0]
    pv_bond_1 = pv_bond_1[0]
    pv_bond_2 = pv_bond_2[0]
    pv_liabilities = pv_liabilities[0]
    if mac_dur_bond_1 > mac_dur_bond_2:
        long_dur_bond = [pv_bond_1, mac_dur_bond_1, bond_cf_1, steps_per_year[0]]
        short_dur_bond = [pv_bond_2, mac_dur_bond_2, bond_cf_2, steps_per_year[1]]
    else:
        long_dur_bond = [pv_bond_2, mac_dur_bond_2, bond_cf_2, steps_per_year[1]]
        short_dur_bond = [pv_bond_1, mac_dur_bond_1, bond_cf_1, steps_per_year[0]]
    tts_for_pf = steps_per_year[0] if len(bond_cf_1.index) > len(bond_cf_2.index) else steps_per_year[
        1]  # To adj disc_rate periodicity for dur_match pf
    dt = 1 / tts_for_pf

    # computes duration match wtss

    wt_array = get_optimal_wts(mac_dur_liabilities, long_dur_bond, short_dur_bond, av, disc_rate, dt)
    wt_long_dur_bond = wt_array[0]
    wt_short_dur_bond = wt_array[1]
    # wt_short_dur_bond = (long_dur_bond[1] - mac_dur_liabilities) / (long_dur_bond[1] - short_dur_bond[1])
    # wt_long_dur_bond = 1-wt_short_dur_bond
    # wt_long_dur_bond = 1.0
    # wt_short_dur_bond = 1-wt_long_dur_bond
    alloc_long_dur_bond = av * wt_long_dur_bond
    alloc_short_dur_bond = av * wt_short_dur_bond
    n_long_dur_bond_match = alloc_long_dur_bond / long_dur_bond[0]
    n_short_dur_bond_match = alloc_short_dur_bond / short_dur_bond[0]
    n_long_bond_full = av / long_dur_bond[0]
    n_short_bond_full = av / short_dur_bond[0]
    dur_matched_bond_cf = pd.DataFrame(
        data=pd.concat([long_dur_bond[2] * n_long_dur_bond_match, short_dur_bond[2] * n_short_dur_bond_match]),
        columns=['cf'])
    dur_matched_bond_cf = dur_matched_bond_cf.groupby(dur_matched_bond_cf.index)['cf'].sum()
    dur_matched_bond_cf.index += 1
    long_bond_cf_full = long_dur_bond[2] * n_long_bond_full
    short_bond_cf_full = short_dur_bond[2] * n_short_bond_full
    disc_rate = disc_rate * dt
    pv_pf, mac_dur_pf = get_present_value(dur_matched_bond_cf, disc_rate)
    pv_pf = pv_pf[0]
    mac_dur_pf = mac_dur_pf[0] * dt
    if fr_change_sim:
        disc_rates = np.linspace(0, 0.1, 50)
        fr_long = []
        fr_short = []
        fr_match = []
        dr_list = []
        for dr in disc_rates:
            dr_list.append(dr)
            liab, dur_li = get_present_value(liabilities, dr)
            l_bond, dur_l = get_present_value(long_bond_cf_full, dr)
            s_bond, dur_s = get_present_value(short_bond_cf_full, dr)
            m_bond, dur_m = get_present_value(dur_matched_bond_cf, dr)
            fr_long.append(get_funding_ratio(liab[0], l_bond[0]))
            fr_short.append(get_funding_ratio(liab[0], s_bond[0]))
            fr_match.append(get_funding_ratio(liab[0], m_bond[0]))
        fr = pd.DataFrame({
            'dr': dr_list,
            'fr_long': fr_long,
            'fr_short': fr_short,
            'fr_match': fr_match,
        }).set_index(keys='dr')
        app = dash.Dash()
        data = [go.Scatter(x=fr.index,
                           y=fr[col],
                           mode='lines',
                           name=col) for col in fr.columns]
        app.layout = html.Div([dcc.Graph(id='cfr', figure=dict(data=data))])
        app.run_server()
    return [wt_long_dur_bond, wt_short_dur_bond, mac_dur_pf, mac_dur_liabilities, long_dur_bond[1], short_dur_bond[1]]


def conv_to_short_rate(r):
    """
    price relative = exp(t*sr) => ln(1+r)/t = sr (assumes t = 1)
    :param r: annualised interest rate
    :return: short rates
    """
    return np.log1p(r)


def conv_to_annualised_rate(sr):
    """
    exp(t*sr) - 1 = r (assumes t = 1)
    :param sr: short rate
    :return: annualised rate for a given short rate
    """
    return np.expm1(sr)


def get_rates_gbm(rf, n_years, steps_per_yr, n_scenarios, volatility, a, b):
    dt = 1 / steps_per_yr
    b = conv_to_short_rate(b)  # Since short rates are being modelled
    sr = conv_to_short_rate(rf)
    total_time_steps = int(n_years * steps_per_yr) + 1
    shock = np.random.normal(loc=0, scale=volatility * np.sqrt(dt), size=(total_time_steps, n_scenarios))
    rates = np.empty_like(shock)
    # For ZCB price generation
    # Formula - please refer cir1.png
    h = math.sqrt(a ** 2 + 2 * volatility ** 2)
    zcb = np.empty_like(shock)

    def price(ttm, rf):
        _A = ((2 * h * math.exp((h + a) * ttm / 2)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))) ** (
                2 * a * b / volatility ** 2)
        _B = (2 * (math.exp(h * ttm) - 1)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
        _P = _A * np.exp(-_B * rf)
        return _P

    zcb[0] = price(n_years, rf)

    rates[0] = sr
    for steps in range(1, total_time_steps):
        prev_rate = rates[steps - 1]
        drift = a * (b - prev_rate) * dt
        shock[steps] = shock[steps] * np.sqrt(prev_rate)
        dr = drift + shock[steps]
        rates[steps] = abs(prev_rate + dr)
        zcb[steps] = price(n_years - steps * dt, rates[steps])
    rates_gbm_df = pd.DataFrame(data=conv_to_annualised_rate(rates), index=range(total_time_steps))
    zcb_gbm_df = pd.DataFrame(data=zcb, index=range(total_time_steps))
    zcb_rets = zcb_gbm_df.pct_change().dropna()
    return rates_gbm_df, zcb_gbm_df, zcb_rets


def get_btr(rates_gbm_df, n_years, steps_per_yr, tenor, cr, fv, n_scenarios):
    cb_df, mac_dur_df, bond_cf = get_bond_gbm(rates_gbm_df, n_years, steps_per_yr, tenor, cr, fv)
    mac_dur_df = mac_dur_df / steps_per_yr
    bond_ann_ret = get_bond_tr(cb_df, bond_cf, n_scenarios)
    return bond_ann_ret, cb_df, mac_dur_df


def reshape_disc_rate(n_years, steps_per_year, n_scenarios, disc_rate):
    rates_df = pd.DataFrame(data=disc_rate, index=range(0, (n_years * steps_per_year + 1)),
                            columns=range(0, n_scenarios))
    return rates_df


def get_bond_gbm(rates_gbm_df: pd.DataFrame, n_years, steps_per_yr, tenor=0, cr=0.05, fv=100):
    bond_cf = 0
    dt = 1 / steps_per_yr
    total_time_steps = int(n_years * steps_per_yr)
    n_scenarios = len(rates_gbm_df.columns)
    cb = np.repeat(0.0, (total_time_steps) * n_scenarios).reshape(total_time_steps, n_scenarios)
    mac_dur = np.empty_like(cb)
    # CB prices
    for step in range(0, total_time_steps):
        ttm = total_time_steps - step
        disc_rate = rates_gbm_df.loc[step]
        disc_rate = pd.DataFrame(np.asarray(pd.concat([disc_rate] * ttm, axis=0)).reshape(ttm, n_scenarios))
        cb[step], mac_dur[step], temp = get_bond_prices(n_years - step * dt, tenor - step * dt, steps_per_yr, disc_rate,
                                                        cr, fv)
        if step == 0:
            bond_cf = temp
    cb_df = pd.DataFrame(cb)
    mac_dur_df = pd.DataFrame(mac_dur)
    cb_df = cb_df.append(cb_df.iloc[-1] * (rates_gbm_df.iloc[-2] * dt + 1), ignore_index=True)
    return cb_df, mac_dur_df, bond_cf


def get_bond_tr(cb_df, bond_cf, n_scenarios):
    print()
    if not len(cb_df.index) - len(bond_cf) == 1:
        dr_steps = cb_df.shape[0]
        cf_steps = bond_cf.shape[0]
        shortfall = cf_steps - dr_steps + 2
        bond_cf.drop(bond_cf.tail(shortfall).index, inplace=True)
    else:
        bond_cf.drop(bond_cf.tail(1).index, inplace=True)
    bond_cf.index += 1
    concat_cf = pd.concat([bond_cf] * n_scenarios, axis=1)
    concat_cf.loc[0] = 0
    concat_cf.loc[len(concat_cf.index)] = 0
    tcf_df = (cb_df + concat_cf)
    tr_df = (np.divide(tcf_df, cb_df.shift()) - 1).dropna()
    # bond_ann_ret = get_ann_return(tr_df)
    return tr_df


def plot_cir():
    app = dash.Dash()

    def upd_label(out_id, inp_id):
        @app.callback(Output(out_id, 'children'),
                      [Input(inp_id, 'value')])
        def upd_(value):
            return value

    app.layout = html.Div([html.Div([html.Div([html.Label(children='Select Initial asset value: '),
                                               dcc.Slider(id='sl_av', min=0.10, max=1.5, step=0.05, value=0.75),
                                               html.Label(id='out_av')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select Initial rf annualized: '),
                                               dcc.Slider(id='sl_rf', min=0.01, max=0.10, step=0.005, value=0.03),
                                               html.Label(id='out_rf')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select expected LT RF: '),
                                               dcc.Slider(id='sl_ltrf', min=0.01, max=0.10, step=0.005, value=0.03),
                                               html.Label(id='out_ltrf')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select speed of MR: '),
                                               dcc.Slider(id='sl_speed', min=0.2, max=1, step=0.05, value=0.5),
                                               html.Label(id='out_speed')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select volatility: '),
                                               dcc.Slider(id='sl_vola', min=0, max=1, step=0.05, value=0.15),
                                               html.Label(id='out_vola')], style={'display': 'inline-block'}),
                                     html.Button(id='sub_cir', children='SUBMIT', n_clicks=0,
                                                 style={'display': 'inline-block'})],
                                    style={'display': 'flex', 'justify-content': 'space-evenly'}),
                           html.Div([html.Div([html.Label(children='Select N-Periods: '),
                                               dcc.Slider(id='sl_periods', min=1, max=20, step=1, value=10),
                                               html.Label(id='out_periods')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select steps_per_yr: '),
                                               dcc.Slider(id='sl_stperyr', min=1, max=10000, step=1, value=12),
                                               html.Label(id='out_stperyr')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select N-Scenarios: '),
                                               dcc.Slider(id='sl_scenarios', min=2, max=250, step=1, value=10),
                                               html.Label(id='out_scenarios')], style={'display': 'inline-block'})
                                     ], style={'display': 'flex', 'justify-content': 'space-evenly',
                                               'padding-top': '25px'}),
                           html.Div([dcc.Graph(id='cir')]),
                           html.Div([dcc.Graph(id='hist_tfr')])])

    upd_label('out_av', 'sl_av')
    upd_label('out_rf', 'sl_rf')
    upd_label('out_ltrf', 'sl_ltrf')
    upd_label('out_speed', 'sl_speed')
    upd_label('out_vola', 'sl_vola')
    upd_label('out_periods', 'sl_periods')
    upd_label('out_stperyr', 'sl_stperyr')
    upd_label('out_scenarios', 'sl_scenarios')

    @app.callback([Output('cir', 'figure'),
                   Output('hist_tfr', 'figure')],
                  [Input('sub_cir', 'n_clicks')],
                  [State('sl_rf', 'value'),
                   State('sl_periods', 'value'),
                   State('sl_stperyr', 'value'),
                   State('sl_scenarios', 'value'),
                   State('sl_vola', 'value'),
                   State('sl_speed', 'value'),
                   State('sl_ltrf', 'value'),
                   State('sl_av', 'value')])
    def upd_cir(n_clicks, rf, n_years, steps_per_yr, n_scenarios, volatility, a, b, av):
        def get_scatter_points(df: pd.DataFrame):
            return df.aggregate(lambda scenario: go.Scatter(x=scenario.index, y=scenario)).tolist()

        tenor = n_years
        rates_gbm_df, zcb_gbm_df, zcb_rets = get_rates_gbm(rf, n_years, steps_per_yr, n_scenarios, volatility, a, b)
        liabilities = zcb_gbm_df  # Assuming same liab as that of ZCB
        cb_df, mac_dur_df, bond_cf = get_bond_gbm(rates_gbm_df, n_years=n_years, steps_per_yr=steps_per_yr, tenor=tenor)

        # Investments in ZCB at T0
        n_bonds = av / zcb_gbm_df.loc[0, 0]
        av_zcb_df = n_bonds * zcb_gbm_df
        # fr_zcb = (av_zcb_df/liabilities).round(decimals=6)
        fr_zcb = get_funding_ratio(liabilities, av_zcb_df).round(decimals=6)
        fr_zcb_df = fr_zcb.pct_change().dropna()

        # Cash investments cumprod
        fd_rates = rates_gbm_df.apply(lambda x: x / steps_per_yr)
        av_cash_df = drawdown(fd_rates, retrive_index=True, init_wealth=av, is1p=False)
        # fr_cash = av_cash_df/liabilities
        fr_cash = get_funding_ratio(liabilities, av_cash_df)
        fr_cash_df = fr_cash.pct_change().dropna()

        fig = make_subplots(rows=4, cols=2, shared_xaxes=True, specs=[[{}, {}],
                                                                      [{}, {}],
                                                                      [{}, {}],
                                                                      [{}, {}]], subplot_titles=(
            "CIR model of Interest rates", "ZCB Prices based on CIR", "CB Prices based on CIR",
            "CB Mac Dur", "Cash invested in FD with rolling maturity",
            " {:.4f} ZCB investments at T=0".format(n_bonds), "Funding Ratio %ch-Cash",
            "Funding Ratio %ch-ZCB"))
        rates_gbm = get_scatter_points(rates_gbm_df)
        zcb_gbm = get_scatter_points(zcb_gbm_df)
        cb_gbm = get_scatter_points(cb_df)
        cb_mac_dur_gbm = get_scatter_points(mac_dur_df)
        av_zcb_gbm = get_scatter_points(av_zcb_df)
        av_cash_gbm = get_scatter_points(av_cash_df)
        fr_cash_gbm = get_scatter_points(fr_cash_df)
        fr_zcb_gbm = get_scatter_points(fr_zcb_df)
        tfr_cash_hist = fr_cash.iloc[-1].tolist()
        tfr_zcb_hist = fr_zcb.iloc[-1].loc[0]  # since all are same

        for rates_data in rates_gbm:
            fig.add_trace(rates_data, row=1, col=1)
        for zcb_price in zcb_gbm:
            fig.add_trace(zcb_price, row=1, col=2)
        for cb_price in cb_gbm:
            fig.add_trace(cb_price, row=2, col=1)
        for cb_mac_dur in cb_mac_dur_gbm:
            fig.add_trace(cb_mac_dur, row=2, col=2)
        for av_cash in av_cash_gbm:
            fig.add_trace(av_cash, row=3, col=1)
        for av_zcb in av_zcb_gbm:
            fig.add_trace(av_zcb, row=3, col=2)
        for fr_cash in fr_cash_gbm:
            fig.add_trace(fr_cash, row=4, col=1)
        for fr_zcb in fr_zcb_gbm:
            fig.add_trace(fr_zcb, row=4, col=2)

        b = conv_to_annualised_rate(b)
        mrl = [dict(type='line', xref='x1', yref='y1', x0=0, x1=n_years * steps_per_yr, y0=b, y1=b,
                    name='Mean Reverting Level',
                    line=dict(dash='dashdot', width=5))]
        fig.update_xaxes(matches='x')
        fig.update_layout(showlegend=False,
                          height=1000,
                          hovermode='closest',
                          shapes=mrl)
        tfr_zcb = [
            dict(type='line', xref='x1', yref='paper', y0=0, y1=1, x0=tfr_zcb_hist, x1=tfr_zcb_hist, name='tfr-zcb',
                 line=dict(dash='dashdot', width=5))]
        tfr_cash_distplot = ff.create_distplot(hist_data=[tfr_cash_hist], group_labels=["tfr_cash"], show_hist=False)
        tfr_cash_distplot.update_layout(shapes=tfr_zcb, hovermode='closest')
        return fig, tfr_cash_distplot

    app.run_server()


def bt_mix(r1: pd.DataFrame, r2: pd.DataFrame, allocator, **kwargs):
    if not r1.shape == r2.shape:
        raise ValueError("Returns need to be of same shape")
    wt_r1 = allocator(r1, r2, **kwargs)
    if not wt_r1.shape == r1.shape:
        raise ValueError("Use a compatible allocator")
    return wt_r1 * r1 + (1 - wt_r1) * r2


def fixed_mix_allocator(r1: pd.DataFrame, r2: pd.DataFrame, wt_r1):
    return pd.DataFrame(data=wt_r1, index=r1.index, columns=r1.columns)


def glide_path_allocator(r1: pd.DataFrame, r2: pd.DataFrame, wt_start=0.8, wt_end=0.2):
    n_points = r1.shape[0]
    n_scenarios = r1.shape[1]
    wt_r1 = pd.Series(np.linspace(wt_start, wt_end, n_points))
    wt_r1 = pd.concat([wt_r1] * n_scenarios, axis=1)
    return wt_r1


def floor_allocator(r1: pd.DataFrame, r2: pd.DataFrame, floor, zcb_prices: pd.DataFrame, m=3, max_dd_mode=False):
    zcb_prices = zcb_prices.drop(index=0).reindex()
    if not r1.shape == r2.shape:
        raise ValueError("Non-Compatible rets dataframe")
    wt_r1 = pd.DataFrame().reindex_like(r1)
    total_time_steps, n_scenarios = r1.shape
    pf_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    for step in range(0, total_time_steps):
        if max_dd_mode:
            peak_value = np.maximum(peak_value, pf_value)
            floor_value = floor * peak_value
        else:
            floor_value = floor * zcb_prices.iloc[step]
        cushion = (pf_value - floor_value) / pf_value
        wt1 = (cushion * m).clip(0, 1)
        pf_ret = wt1 * r1.iloc[step] + (1 - wt1) * r2.iloc[step]
        pf_value = (1 + pf_ret) * pf_value
        wt_r1.iloc[step] = wt1
    return wt_r1


def distplot_terminal_paths(floor_factor, **kwargs):
    app = dash.Dash()
    terminal_paths = []
    pf_type = []
    stats = []
    for key, value in kwargs.items():
        pf_type.append(key)
        terminal_paths.append(value.tolist())
        stats.append(terminal_risk_stats(fv=1, floor_factor=floor_factor, wealth_index=value, aslst=True, strategy=key))
    stats = pd.DataFrame(data=stats, columns=["strategy", 'Exp_wealth', "Exp_Volatility", "Med_Wealth", "#_violations",
                                              "p_violations", "CVaR"])
    floor = [dict(type='line', xref='x1', yref='paper', y0=0, y1=1, x0=floor_factor, x1=floor_factor, name='floor',
                  line=dict(dash='dashdot', width=5))]
    fig = ff.create_distplot(hist_data=terminal_paths, group_labels=pf_type, show_hist=False)
    fig.update_layout(shapes=floor)
    app.layout = html.Div([html.Div([dcc.Graph(id='terminal_dist_plot', figure=fig)]),
                           html.Div([dt.DataTable(id='risk-stats',
                                                  columns=[{'name': col,
                                                            'id': col} for col in stats.columns],
                                                  data=stats.to_dict('records'))])
                           ])
    app.run_server()


def get_options_cv(elasticnet=False):
    if elasticnet:
        params = {
            'max_lamda': 0.25,
            'n_lamdas': 20,
            'max_l1_ratio': 0.99,
            'n_l1-ratio': 50,
            'k_folds': 10,
            'randomseed': 7777
        }
    else:
        params = {
            'max_lamda': 0.25,
            'n_lamdas': 100,
            'k_folds': 10,
            'randomseed': 7777
        }
    return params

def regress(dependent_var: pd.DataFrame, explanatory_var: pd.DataFrame, start_period=None, end_period=None,
            intercept=True, excess_mkt=True, rfcol='RF', method='ols', lamda=0.1, C=0.1, penalty='l1'):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
        returns an object of type statsmodel's RegressionResults on which you can call
           .summary() to print a full summary
           .params for the coefficients
           .tvalues and .pvalues for the significance levels
           .rsquared_adj and .rsquared for quality of fit
           NOTE: SKLearn calles Lambda Alpha.  Also, it uses a scaled version of LASSO argument, so here I scale when converting lambda to alpha
    """
    if isinstance(dependent_var, pd.Series):
        dependent_var = pd.DataFrame(dependent_var)
    dependent_var = dependent_var.loc[start_period:end_period]
    explanatory_var = explanatory_var.loc[start_period:end_period]
    if excess_mkt:
        dependent_var = dependent_var - explanatory_var.loc[:, [rfcol]].values
        explanatory_var = explanatory_var.drop([rfcol], axis=1)
    if method == 'ols':
        if intercept:
            explanatory_var['Alpha'] = 1
        regression_result = sm.OLS(dependent_var, explanatory_var).fit()
        return regression_result
    elif method == 'lasso':
        alpha = lamda / (2*dependent_var.shape[0])
        sk_lasso = Lasso(alpha=alpha, fit_intercept=intercept).fit(X=explanatory_var, y=dependent_var)
        print_sklearn_results(method=method, intercept=sk_lasso.intercept_, coeff=sk_lasso.coef_, explanatory_df=explanatory_var, dependent_df=dependent_var, alpha=alpha, lamda=lamda)
        return sk_lasso
    elif method == 'ridge':
        alpha = lamda
        sk_ridge = Ridge(alpha=alpha, fit_intercept=intercept).fit(X=explanatory_var, y=dependent_var)
        print_sklearn_results(method=method, intercept=sk_ridge.intercept_, coeff=sk_ridge.coef_, explanatory_df=explanatory_var, dependent_df=dependent_var, alpha=alpha, lamda=lamda)
        return sk_ridge
    elif method == 'cv_lasso':
        params = get_options_cv()
        max_alpha = params['max_lamda'] / (2*dependent_var.shape[0])
        alphas = np.linspace(1e-6, max_alpha, params['n_lamdas'])
        parameters = {'alpha': alphas}
        lasso = Lasso(fit_intercept=True, random_state=params['randomseed'])
        cv_lasso = GridSearchCV(lasso, parameters, cv=params['k_folds'], refit=True)
        cv_lasso = cv_lasso.fit(X=explanatory_var, y=dependent_var)
        lasso_best = cv_lasso.best_estimator_
        alpha_best = cv_lasso.best_params_['alpha']
        lamda_best = alpha_best * 2 * dependent_var.shape[0]
        print('Max_alpha is : {}'.format(max_alpha))
        print_sklearn_results(method=method, intercept=lasso_best.intercept_, coeff=lasso_best.coef_,
                              explanatory_df=explanatory_var, dependent_df=dependent_var, alpha=alpha_best, lamda=lamda_best)
        return cv_lasso
    elif method == 'cv_elasticnet':
        params = get_options_cv(elasticnet=True)
        max_alpha = params['max_lamda'] / (2 * dependent_var.shape[0])
        alphas = np.linspace(1e-6, max_alpha, params['n_lamdas'])
        max_l1_ratio = params['max_l1_ratio']
        l1_ratios = np.linspace(1e-6, max_l1_ratio, params['n_l1-ratio'])
        parameters = {'alpha': alphas, 'l1_ratio': l1_ratios}
        elasticnet = ElasticNet(fit_intercept=True, random_state=params['randomseed'])
        cv_elasticnet = GridSearchCV(elasticnet, parameters, cv=params['k_folds'], refit=True)
        cv_elasticnet = cv_elasticnet.fit(X=explanatory_var, y=dependent_var)
        elastic_best = cv_elasticnet.best_estimator_
        alpha_best = cv_elasticnet.best_params_['alpha']
        l1_ratio_best = cv_elasticnet.best_params_['l1_ratio']
        lasso_lamda_best = alpha_best * 2 * dependent_var.shape[0] * l1_ratio_best
        ridge_lambda_best = alpha_best * dependent_var.shape[0] * (1 - l1_ratio_best)
        msg = '''
            Best L1 ratio is : {}
            Best Lasso_Lambda is : {}            
            Best Ridge_Lambda is : {}
        '''.format(l1_ratio_best, lasso_lamda_best, ridge_lambda_best)
        print(msg)
        print_sklearn_results(method=method, intercept=elastic_best.intercept_, coeff=elastic_best.coef_,
                              explanatory_df=explanatory_var, dependent_df=dependent_var, alpha=alpha_best,
                              lamda=lasso_lamda_best)
        return cv_elasticnet
    elif method == 'cv_log_regression':
        dependent_var = dependent_var['Label'].to_numpy()
        scoring = "roc_auc"
        kfolds = TimeSeriesSplit(n_splits=3)
        # Create regularization hyperparameter space - lower values strong regularisation
        C = np.reciprocal([0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005,
                           0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000])
        hyperparameters = dict(C=C)
        lr_l1 = LogisticRegression(max_iter=10000, penalty=penalty, solver='saga')
        log_regression_l1_best = GridSearchCV(estimator=lr_l1, param_grid=hyperparameters, cv=kfolds, scoring=scoring).fit(X=explanatory_var, y=dependent_var).best_estimator_
        return log_regression_l1_best
    elif method == 'log_regression':
        lr_l1 = LogisticRegression(max_iter=10000, C=C, penalty=penalty, solver='saga').fit(X=explanatory_var, y=dependent_var)
        return lr_l1
    return None


def print_sklearn_results(method, intercept, coeff, explanatory_df, dependent_df, alpha, lamda):
    period = str(explanatory_df.index[0])  + ' - ' + str(explanatory_df.index[-1])
    desc = '''
        Regression method is {}
        Time period is {}
        Alpha is {}
        Lambda is {}
    '''.format(method, period, alpha, lamda)
    print(desc)
    factor_names = ['intercept'] + list(explanatory_df.columns)
    loadings = np.insert(coeff, 0, intercept)
    loadings = pd.DataFrame(loadings, index=factor_names, columns=[method]).transpose()
    print(loadings)


def tracking_error(act_rets, exp_rets):
    act_rets.columns = [0]
    err = act_rets - exp_rets
    sqderr = (err ** 2).sum()
    return np.sqrt(sqderr)


def pf_tracking_error(weights, actual_rets, bm_rets):
    exp_rets = pd.DataFrame(data=(weights * bm_rets).sum(axis=1))
    return tracking_error(actual_rets, exp_rets)


def style_analyze(dependent_var: pd.DataFrame, explanatory_var: pd.DataFrame, start_period=None, end_period=None,
                  droprf=False, rfcol='RF'):
    if isinstance(dependent_var, pd.Series):
        dependent_var = pd.DataFrame(dependent_var)
    dependent_var = dependent_var.loc[start_period:end_period]
    explanatory_var = explanatory_var.loc[start_period:end_period]
    if droprf:
        explanatory_var = explanatory_var.drop([rfcol], axis=1)
    n_expl_var = explanatory_var.shape[1]
    init_guess = np.repeat(1 / n_expl_var, n_expl_var)
    bounds = Bounds(lb=0.0, ub=1.0)
    wts_sum_to_1 = {
        'type': 'eq',
        'fun': lambda wts: np.sum(wts) - 1
    }
    result = minimize(fun=pf_tracking_error,
                      args=(dependent_var, explanatory_var),
                      bounds=bounds,
                      constraints=[wts_sum_to_1],
                      method='SLSQP',
                      options={'disp': False},
                      x0=init_guess)
    weights = pd.Series(data=result.x, index=explanatory_var.columns)
    return weights


# WEIGHTING OPTIMIZERS - BACKTEST WEIGHTING SCHEMES

def weight_ew(r, cap_wts=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    n_components = len(r.columns)
    weights = np.repeat(1 / n_components, n_components)
    weights = pd.Series(weights, index=r.columns)
    if cap_wts is not None:
        cap_weights = cap_wts.loc[r.index[0]]  # Cap wts as at begining of a window
        if microcap_threshold is not None and microcap_threshold > 0:
            drop_microcap_mask = cap_weights < microcap_threshold
            weights[drop_microcap_mask] = 0
            weights = weights / weights.sum()  # Recomputes weights
        if max_cw_mult is not None and max_cw_mult > 0:
            weights = np.minimum(weights, cap_weights * max_cw_mult)
            weights = weights / weights.sum()  # Recomputes weights
    return weights


def weight_custom(r, cust_wts):
    return cust_wts


def weight_cw(r, cap_wts, **kwargs):
    cap_wts = cap_wts.loc[r.index[0]]  # Because for a rolling period, i would create PF using 1st available market weights for such window.
    return cap_wts


def sample_cov(r, **kwargs):
    return r.cov()


def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    Average of sample correlation is used to find const_cov
    """
    sample_corr = r.corr()
    n_assets = len(r.columns)
    avg_distinct_rho = (sample_corr.values.sum() - n_assets) / (
            n_assets * (n_assets - 1))  # Taking avg of off diagonal corr matrix on one side
    const_corr = np.full_like(sample_corr, avg_distinct_rho)
    np.fill_diagonal(const_corr, 1.)
    sd = r.std()
    # Convert to cov using statsmodel
    const_cov_sm = mh.corr2cov(const_corr, sd)
    # Convert to cov using formula and outer product - alternate way is to use sd @ sd.T instead of np.outer(sd, sd) -> yields matrix(mxm)
    const_cov = const_corr * np.outer(sd, sd)
    return pd.DataFrame(const_cov, columns=r.columns, index=r.columns)


def stat_shrinkage_cov(r, delta=0.5, **kwargs):
    s_cov = sample_cov(r, **kwargs)
    c_cov = cc_cov(r, **kwargs)
    stat_cov = delta * c_cov + (1 - delta) * s_cov
    return stat_cov


def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    cov_df = cov_estimator(r, **kwargs)
    wts = get_gmv(ret_series=r, cov_df=cov_df, rf=0.03, reqd_strategies=None, onlywts=True)
    return wts


def weight_erc(r, cov_estimator=sample_cov, **kwargs):
    cov_df = cov_estimator(r, **kwargs)
    wts = equal_risk_contrib(cov_df)
    return wts


def bt_roll(r, window, weighting_scheme, **kwargs):
    total_periods = len(r.index)
    windows = [(start, start + window) for start in range(total_periods - window)]
    weights = [weighting_scheme(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # Convert from list of weights to dataframe with sectors along columns and index begining after first rolling period, so that it aligns with returns df
    weights = pd.DataFrame(weights, columns=r.columns, index=r.iloc[window:].index)
    returns = (weights * r).sum(axis=1, min_count=1)
    return returns


def as_colvec(x):
    if np.ndim(x) == 2:
        return x
    else:
        return np.expand_dims(x, axis=1)


def rev_opt_implied_returns(delta, sigma_prior: pd.DataFrame, wts_prior: pd.Series):
    """
    Obtain the implied expected returns by reverse engineering the weights
    Inputs:
    delta: Risk Aversion Coefficient (scalar)
    sigma: Variance-Covariance Matrix (N x N) as DataFrame
        w: Portfolio weights (N x 1) as Series
    Returns an N x 1 vector of Returns as Series
    """
    pi = (delta * sigma_prior @ wts_prior).squeeze()  # @ may be used instead of dot, but some issues arise if a df is not passed
    pi.name = 'Implied Returns'
    return pi


def omega_proportional_prior(sigma_prior: pd.DataFrame, tau, p: pd.DataFrame):
    """
    As we noted previously, \cite{he1999intuition} suggest that if the investor does not have a specific way to explicitly
    quantify the uncertaintly associated with the view in the Ω matrix, one could make the simplifying assumption
    that Ω is proportional to the variance of the prior.

    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a K x K diagonal DataFrame, a Matrix representing Prior Uncertainties - Omega
    """
    scaled_sigma_prior = (tau * sigma_prior).to_numpy()
    helit_omega_matrix_kxk = p.to_numpy() @ scaled_sigma_prior @ p.T.to_numpy()
    helit_omega_diag_values = np.diag(helit_omega_matrix_kxk)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(helit_omega_diag_values), columns=p.index, index=p.index)


def black_litterman(wts_prior: pd.Series, sigma_prior: pd.DataFrame, p: pd.DataFrame, q: pd.Series, omega=None,
                    delta=2.5, tau=0.02):
    """

    :param wts_prior: N x 1 col vector
    :param sigma_prior: N x N cov matrix
    :param p: K x N view portfolio - associating views with assets
    :param q: K x 1 col vector representing views
    :param omega: Uncertainty around views. If none - omega - proportional prior
    :param delta: Risk aversion factor
    :param tau: Uncertainty factor scaling sigma_prior
    :return: posterior returns and cov based on black litterman formula
    """
    if omega is None:
        omega = omega_proportional_prior(sigma_prior, tau, p).to_numpy()
    p = p.to_numpy()
    q = q.to_numpy()
    n_assets = wts_prior.shape[0]
    k_views = q.shape[0]
    # Get implied pi
    pi = rev_opt_implied_returns(delta, sigma_prior, wts_prior).to_numpy()
    # Scaled sigma_prior
    sigma_prior = sigma_prior.to_numpy()
    scaled_sigma_prior = (tau * sigma_prior)
    common_factor = scaled_sigma_prior @ p.T @ inv(p @ scaled_sigma_prior @ p.T + omega)
    bl_mu = pi + common_factor @ (q - (p @ pi))
    bl_sigma = sigma_prior + scaled_sigma_prior - common_factor @ p @ scaled_sigma_prior
    return bl_mu, bl_sigma, omega


def get_inverse_df(df: pd.DataFrame):
    return pd.DataFrame(inv(df), index=df.columns, columns=df.index)


def w_msr_closed_form(sigma: pd.DataFrame, mu: pd.Series, scale=True):
    """

    :param sigma: N x N cov mat
    :param mu: N x 1 expected return col vector
    :param scale: to give % of wt and it assumes all wts are +ve
    :return: wts_msr
    """
    wts_msr = inv(sigma) @ mu
    if scale:
        wts_msr = wts_msr / wts_msr.sum()
    return wts_msr


def get_optimal_wts(sigma_prior: pd.DataFrame, bl_sigma: pd.DataFrame, pi: pd.Series, bl_mu: pd.Series, delta, tau, wts_he=True, scale=True):
    if wts_he:
        prior_wts_equil = ((inv(sigma_prior) @ pi) / delta) / (1 + tau)
        posterior_wts_optimal = (inv(bl_sigma) @ bl_mu) / delta
    else:
        prior_wts_equil = ((inv(sigma_prior) @ pi) / delta)
        posterior_wts_optimal = (inv(bl_sigma) @ bl_mu)
        if scale:
            posterior_wts_optimal = posterior_wts_optimal / posterior_wts_optimal.sum()
    wts_diff = posterior_wts_optimal - prior_wts_equil
    return prior_wts_equil, posterior_wts_optimal, wts_diff


def get_risk_contrib(weights, cov):
    marginal_contrib = cov @ weights
    pf_var = get_pf_vol(weights, cov) ** 2
    indiv_risk_contrib = (marginal_contrib * weights) / pf_var
    return indiv_risk_contrib


def target_risk_contrib(target_risk, cov):
    n_assets = cov.shape[0]
    init_guess = np.repeat(1/n_assets, n_assets)
    bounds = Bounds(lb=0.0, ub=1.0)
    wts_sum_to_1 = {
        'type': 'eq',
        'fun': lambda wts: np.sum(wts) - 1
    }

    def min_sq_deviation(weights, cov, target_risk):
        indiv_risk_contrib = get_risk_contrib(weights, cov)
        err = indiv_risk_contrib - target_risk
        return err.T @ err

    results = minimize(fun=min_sq_deviation,
                       args=(cov, target_risk),
                       x0=init_guess,
                       bounds=bounds,
                       method='SLSQP',
                       options={'disp':False},
                       constraints=[wts_sum_to_1])
    weights = results.x
    return weights


def equal_risk_contrib(cov):
    n_assets = cov.shape[0]
    target_risk = np.repeat(1/n_assets, n_assets)
    weights = target_risk_contrib(target_risk, cov)
    return weights


def get_wealth_index_risk(btr: dict):
    btr_df = pd.DataFrame(btr)
    btr_df.dropna(inplace=True)
    btr_wealth_index = pd.DataFrame({key+'_wealth_index': drawdown(btr[key], retrive_index=True, is1p=False) for key in btr_df.columns})
    wealth_data_plots=[go.Scatter(x=btr_wealth_index.index.to_timestamp(),
                     y=btr_wealth_index[str],
                     name=str) for str in btr_wealth_index.columns]
    return wealth_data_plots, risk_info(btr_df)


def split_regime_returns(wealth_data, reqd_assets, regime_select):
    wealth_data[reqd_assets] = wealth_data[reqd_assets].pct_change()
    wealth_data.dropna(how='any', inplace=True, axis=0)
    asset_returns = wealth_data[reqd_assets]
    growth_regime_mask = wealth_data[regime_select] == 1
    crash_regime_mask = wealth_data[regime_select] == -1
    asset_returns_growth_regime, asset_returns_crash_regime = wealth_data.loc[growth_regime_mask, reqd_assets], wealth_data.loc[crash_regime_mask, reqd_assets]
    return asset_returns, asset_returns_growth_regime, asset_returns_crash_regime


def QQ_plot(rets_data, reqd_assets:list):
    qq_scatter_data = [go.Scatter(x=qqplot(rets_data[asset], line='s').gca().lines[0].get_xdata(),
                                  y=qqplot(rets_data[asset], line='s').gca().lines[0].get_ydata(),
                                  mode='markers',
                                  name=asset) for asset in reqd_assets]
    qq_scatter_opt_data = [go.Scatter(x=qqplot(rets_data[asset], line='s').gca().lines[1].get_xdata(),
                                      y=qqplot(rets_data[asset], line='s').gca().lines[1].get_ydata(),
                                      mode='lines',
                                      name=asset) for asset in reqd_assets]
    qq_scatter_data.extend(qq_scatter_opt_data)
    qq_layout = go.Layout(title='QQ-Plots',
                          xaxis=dict(title='Idealised Quantiles'),
                          yaxis=dict(title='Actual Quantiles'),
                          showlegend=True)
    fig = go.Figure(data=qq_scatter_data, layout=qq_layout)
    # fig.add_trace(data=qq_scatter_opt_data)
    return fig


def ecdf_plot(rets_data, reqd_assets:list):
    ecdf_data = [go.Scatter(x=ECDF(rets_data[asset]).x,
                            y=ECDF(rets_data[asset]).y,
                            mode='lines',
                            name=asset) for asset in reqd_assets]
    ecdf_layout = go.Layout(title='ECDF Plot')
    fig = go.Figure(ecdf_data, ecdf_layout)
    return fig


def trend_filter(rets_data, lambda_value):
    """
    Strips and returns the drift term for identification of regime changes using ML algo - refer coursera notebook
    :param rets_data:
    :param lambda_value:
    :return:
    """
    #USING CVXPY convex optimiser
    n_periods = rets_data.shape[0]
    rets = rets_data.to_numpy()

    D_full = np.diag([1]*n_periods) - np.diag([1]*(n_periods-1), 1)
    D = D_full[0:n_periods-1,]
    beta = cp.Variable(n_periods)
    lambd = cp.Parameter(nonneg=True)
    lambd.value = lambda_value

    def lasso_min(betas, rets, lambd):
        return cp.norm(rets-betas, 2)**2 + lambd*cp.norm(cp.matmul(D, betas), 1)

    problem = cp.Problem(cp.Minimize(lasso_min(beta, rets, lambd)))
    problem.solve()

    # NOT WORKING
    # n_periods = rets_data.shape[0]
    # D_full = np.diag([1] * n_periods) - np.diag([1] * (n_periods - 1), 1)
    # D = D_full[0:n_periods - 1, ]
    # def lasso_min(betas, rets, D, lambda_value):
    #     return np.linalg.norm(rets-betas)**2 + lambda_value*np.linalg.norm(D@betas,1)
    #
    # init_guess = np.repeat(1/n_periods, n_periods)
    # bounds = Bounds(lb=0.0, ub=1.0)
    # results = minimize(fun=lasso_min,
    #                    args=(rets_data, D, lambda_value),
    #                    x0=init_guess,
    #                    bounds=bounds,
    #                    method='SLSQP',
    #                    options={'disp':False})
    # betas = pd.Series(results.x, index=rets_data.index)
    # return betas
    betas = pd.DataFrame(beta.value, index=rets_data.index.to_timestamp(), columns=['drift'])
    return betas


def get_regime_switches(betas:pd.DataFrame, threshold_value=1e-5):
    betas['crash_regime'] = False
    betas['crash_regime'] = betas['drift'] < threshold_value
    switches = betas[betas['crash_regime'].diff().fillna(False)]
    betas_switch = betas.loc[switches.index]
    crash_periods_start = list(betas_switch[betas_switch['crash_regime']].index)
    crash_periods_end = list(betas_switch[betas_switch['crash_regime'] == False].index)
    coordinate_list = list(zip(crash_periods_start, crash_periods_end))
    return coordinate_list


def trend_filter_plot(rets_data, lambda_value, threshold_value):
    betas = trend_filter(rets_data, lambda_value)
    regime_plot_data = [go.Scatter(x=betas.index,
                                   y=rets_data,
                                   mode='lines',
                                   name='Actual TS data'),
                        go.Scatter(x=betas.index,
                                   y=betas['drift'],
                                   mode='lines',
                                   name='Drift term')]
    fig = go.Figure(regime_plot_data, go.Layout(title='Trend Filter Plot'))
    regimes = get_regime_switches(betas, threshold_value)
    for coordinates in regimes:
        rect_shape = dict(type='rect', y0=0, y1=1, x0=coordinates[0], x1=coordinates[1],
                          yref='paper', xref='x1', fillcolor='rgba(255,24,86,0.5)')
        fig.add_shape(rect_shape)
    return fig


def transition_matrix(regime: pd.Series):
    n_unique = regime.value_counts()
    n_unique.index = ['normal', 'crash']
    switches = regime.diff().fillna(0.0)
    n_switches = switches.value_counts()
    n_switches.index = ['no_switch', 'cr_gr_switch', 'gr_cr_switch']
    p_matrix = pd.DataFrame({
        'normal': [(n_unique['normal'] - n_switches['gr_cr_switch']), n_switches['gr_cr_switch']] / n_unique['normal'],
        'crash': [n_switches['cr_gr_switch'], (n_unique['crash'] - n_switches['cr_gr_switch'])] / n_unique['crash']
    }, index=['normal', 'crash']).T
    return p_matrix


def check_ranks(coeff_matrix, b):
    aug_matrix = np.append(coeff_matrix, b.reshape(1, len(b)).T, axis=1)
    rank_coeff_matrix = np.linalg.matrix_rank(coeff_matrix)
    rank_aug_matrix = np.linalg.matrix_rank(aug_matrix)
    if rank_aug_matrix == rank_coeff_matrix:
        print('Unique stationary pi exist')
    else:
        print('Do Markov Simulation')
    return None


def get_markov_stationary_distr(p_matrix: pd.DataFrame):
    """
    Please refer https://towardsdatascience.com/markov-chain-analysis-and-simulation-using-python-4507cee0b06e
    Also see notes
    A.pi = b
    So, (AT.A).pi = (AT.b) # Visualise as reverse transforming b vector to land at pi
    Needed to implement markov simulation
    """
    n_states = p_matrix.shape[0]
    i = np.repeat(1, n_states)
    I = np.identity(n_states)
    P = p_matrix.to_numpy()
    A = np.append((P.T - I), [i], axis=0)
    b = np.repeat(0, n_states)
    b = np.append(b, 1)
    check_ranks(A, b)
    pi = pd.Series(np.linalg.solve(A.T @ A, A.T @ b), index=p_matrix.columns)
    return pi


def get_multivariate_sim_scenario_rets(asset_data, n_years, n_scenarios, regime_col='Regime-5'):
    """
    :return: 3 dimensional matrix asset returns simulated for time steps for each scenario scenario wise
    """
    np.random.seed(7)
    rets, ret_gr, ret_cr = split_regime_returns(asset_data, config.asset_categories, regime_col)  # mNot annualised
    regime = asset_data[regime_col]
    p_matrix = transition_matrix(regime)
    pi = get_markov_stationary_distr(p_matrix)
    time_steps = n_years * 12
    mvn_distr_gr = np.random.multivariate_normal(get_ann_return(ret_gr, periodicity=1, expm1=False),
                                                 get_cov(ret_gr, periodicity=1), (n_scenarios, time_steps))
    mvn_distr_cr = np.random.multivariate_normal(get_ann_return(ret_cr, periodicity=1, expm1=False),
                                                 get_cov(ret_cr, periodicity=1), (n_scenarios, time_steps))
    mvn_mixed = mvn_distr_gr * pi['normal'] + mvn_distr_cr * pi['crash']  # Long term stationary probability distribution representing fraction of time spent in each state
    # mvn_mixed = mvn_mixed.reshape(len(rets.columns), time_steps, n_scenarios)
    return mvn_mixed


def fix_mix(orig_wts, mvn_asset_rets, spending_rate, rebal_freq=None):
    """
    rebal_freq if none implies that at first time step, wt scheme which was used to form portfolio is allowed to run its course throughout
    rebal freq if in month (say 3) means at end of each 3rd month, wt scheme is reset to original wt scheme
    """
    n_scenarios, time_steps, n_assets = mvn_asset_rets.shape
    wealth_index = np.zeros((int(time_steps/12), n_scenarios))
    for scenario in range(n_scenarios):
        asset_rets = mvn_asset_rets[scenario]
        cum_pf_rets_component_wise = orig_wts  # Initial weight adopted for first time step
        if rebal_freq is None:
            for period in range(time_steps):
                cum_pf_rets_component_wise = cum_pf_rets_component_wise * asset_rets[period]
                if period % 12 == 0:
                    cum_pf_rets_component_wise = cum_pf_rets_component_wise * (1-spending_rate)
                    wealth_index[int(period/12), scenario] = np.sum(cum_pf_rets_component_wise)
        else:
            for period in range(time_steps):
                cum_pf_rets_component_wise = cum_pf_rets_component_wise * asset_rets[period]
                if period % rebal_freq == 0:
                    cum_pf_rets_component_wise = np.sum(
                        cum_pf_rets_component_wise) * orig_wts  # Rebalnce occurs at the end of the period
                if period % 12 == 0:
                    cum_pf_rets_component_wise = cum_pf_rets_component_wise * (1 - spending_rate)
                    wealth_index[int(period / 12), scenario] = np.sum(cum_pf_rets_component_wise)
    return wealth_index


def build_pf_ret(mvn_asset_rets, orig_wts, allocator=fix_mix, spending_rate=0.03, rebal_freq=None):
    return allocator(orig_wts, mvn_asset_rets, spending_rate, rebal_freq)


def retrieve_stationary_time_series(features_df: pd.DataFrame, threshold=0.1):
    """
    #Refer https://machinelearningmastery.com/time-series-data-stationary-python/
    # Check stationarity in time series data
    # We will perform adfuller test to check unit roots 3 times.
    # First time for non-stationary series we will take first order difference
    # Second time we will take second order difference
    # Third time if there are still remaining non-stationary columns we will drop them from feature set
    """
    def check_ad_fuller(col):
        result = adfuller(col)
        p_value = result[1]
        return p_value

    non_stationary_cols = []

    for order in range(3):
        for col in features_df.columns:
            p_value = check_ad_fuller(features_df[col])
            if p_value > threshold: #failing to reject h0 and thus original time series has unit root and is thus non stationary
                if order == 2:
                    non_stationary_cols.append(col)
                    features_df.drop(non_stationary_cols, axis=1)
                else:
                    features_df[col] = features_df[col].diff() #Taking differences to again check for stationarity
        features_df.dropna(axis=0, inplace=True)
    return features_df


def standardise_data(stationary_df:pd.DataFrame):
    """
    Does standardisation
    :param feature_df:
    :return: standardised dataframe
    """
    scalar = StandardScaler()
    scalar.fit(stationary_df)
    standardised_feature_df = pd.DataFrame(scalar.transform(stationary_df), columns=stationary_df.columns, index=stationary_df.index)
    return standardised_feature_df


def get_selected_features(dataset_features, model):
    """
    Selects those statistically significant features and transforms the original dataset to selected features. Mainly DIMENSIONALITY REDUCTION OF DATASET
    """
    model = SelectFromModel(model, prefit=True)
    feature_bool_mask = model.get_support()
    selected_features = dataset_features.columns[feature_bool_mask]
    transformed_dataset = pd.DataFrame(model.transform(dataset_features), columns=dataset_features.columns[feature_bool_mask], index=dataset_features.index)
    return selected_features, transformed_dataset
