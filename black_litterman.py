# import edhec_risk_kit as sps
import sps_finance_toolkit as sps
import numpy as np
import pandas as pd
from dash_extensions.callback import CallbackGrouper
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_table as dt

app = dash.Dash()
# cg = app
cg = CallbackGrouper()
checkoptions = [
    {'label': 'Weights', 'value': 0},
    {'label': 'Cov_Mat', 'value': 1}
]


def get_bl_results(wts_prior: pd.Series, sigma_prior: pd.DataFrame, delta, tau, p: pd.DataFrame = None,
                   q: pd.Series = None, omega=None, scale=True, wts_he=False):
    pi = sps.rev_opt_implied_returns(delta, sigma_prior, wts_prior).to_numpy()
    omega = omega.to_numpy()
    diag_el = np.diag(omega)
    if not diag_el.any():
        omega = None
    bl_mu, bl_sigma, omega = sps.black_litterman(wts_prior, sigma_prior, p, q, omega, delta, tau)
    prior_wts, posterior_wts, wts_diff = sps.get_optimal_wts(sigma_prior, bl_sigma, pi, bl_mu, delta, tau,
                                                             wts_he=wts_he, scale=scale)
    wts = [prior_wts, posterior_wts, wts_diff]
    return pi, bl_mu, bl_sigma, wts, omega


def percent_format(col: pd.Series):
    return col.map('{:.4%}'.format)


app.layout = html.Div(
    [html.Div([html.Div([dcc.Checklist(id='local_file', options=[{'label': 'Y', 'value': 1}], value=[],
                                       labelStyle={'display': 'inline-block'}),
                         dcc.Checklist(id='hard_coded', options=[{'label': 'H', 'value': 1}], value=[],
                                       labelStyle={'display': 'inline-block'})
                         ]),
               html.Label(children='Enter list of asset names: '),
               dcc.Input(id='asset_list', value='', type='text')], style={'display': 'inline-block'}),
     html.Div([dcc.Checklist(id='display_table', options=checkoptions, value=[],
                             labelStyle={'display': 'inline-block'})]),
     html.Div(id='dump_vw_df', style={'display': 'none'}),
     html.Div(id='dump_wt_df', style={'display': 'none'}),
     html.Div(id='dump_wt_hc_df', style={'display': 'none'}),
     html.Div(id='dump_sigma_hc_df', style={'display': 'none'}),
     html.Div(id='dump_no', style={'display': 'none'}),
     html.Button(id='proceed_to_views', children='PROCEED', n_clicks=0, style={'display': 'inline-block'}),
     html.Div(id='wts_table_container', children=[dt.DataTable(id='wts_table', editable=True)],
              style={'display': 'none'}),
     html.Div(id='rhos_table_container', children=[dt.DataTable(id='rhos_table', editable=True)],
              style={'display': 'none'}),
     html.Div(id='vol_table_container', children=[dt.DataTable(id='vol_table', editable=True)],
              style={'display': 'none'}),
     html.Div(id='cov_table_container', children=[dt.DataTable(id='cov_table', editable=True)],
              style={'display': 'none'}),
     html.Div(id='view_params', children=[html.Label(children='Enter no of views: '),
                                          dcc.Input(id='no_views', value='', type='number'),
                                          html.Label(children='Enter tau: '),
                                          dcc.Input(id='tau', value=0.02, type='number'),
                                          html.Label(children='Enter delta: '),
                                          dcc.Input(id='delta', value=2.5, type='number'),
                                          html.Button(id='update_views', children='UPDATE VIEWS', n_clicks=0)],
              style={'display': 'none'}),
     html.Div(id='view_table_container', children=[dt.DataTable(id='view_table', editable=True)],
              style={'display': 'none'}),
     html.Div(id='pick_table_container', children=[dt.DataTable(id='pick_table', editable=True)],
              style={'display': 'none'}),
     html.Div(id='omega_table_container', children=[dt.DataTable(id='omega_table', editable=True)],
              style={'display': 'none'}),
     html.Button(id='update', children='UPDATE', n_clicks=0, style={'display': 'none'}),
     html.Div(dt.DataTable(id='bl_results_table'),
              style={'display': 'flex', 'justify-content': 'space-evenly', 'padding-top': '25px'}),
     html.Div(html.Pre(id='disp_bl_cov', style={'display': 'block'}),
              style={'display': 'flex', 'justify-content': 'space-evenly', 'padding-top': '25px'}),
     html.Div(html.Pre(id='disp_omega', style={'display': 'block'}),
              style={'display': 'flex', 'justify-content': 'space-evenly', 'padding-top': '25px'})
     ])


@cg.callback([Output('asset_list', 'value'),
              Output('dump_no', 'children'),
              Output('dump_vw_df', 'children'),
              Output('dump_wt_df', 'children'),
              Output('display_table', 'options')],
             [Input('local_file', 'value')])
def upd_asset_list(localfile):
    if localfile:
        ind_vw_2014 = sps.get_df('data/ind49_m_vw_rets.csv', to_per=True, start_period='2013', reqd_strategies=['Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food'])
        ind_mkt_wts_2014 = sps.plot_corr_mktret(ind_ret_filename='data/ind49_m_vw_rets.csv',
                                                n_firms_filename='data/ind49_m_nfirms.csv',
                                                size_filename='data/ind49_m_size.csv',
                                                start_period='2014',
                                                end_period=None,
                                                to_per=True,
                                                retrieve_mkt_cap_wts=True,
                                                format='%Y%m',
                                                reqd_strategies=['Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food'])
        a = ind_vw_2014.columns
        asset_list = [' '.join(list(ind_vw_2014.columns))]
        json_vw = ind_vw_2014.to_json(date_format='iso', orient='table')
        json_wts = ind_mkt_wts_2014.to_json(date_format='iso', orient='table')
        checkoptions = []
        return asset_list, len(ind_vw_2014.columns), json_vw, json_wts, checkoptions


@cg.callback([Output('dump_wt_hc_df', 'children'),
              Output('dump_sigma_hc_df', 'children'),
              Output('dump_no', 'children'),
              Output('asset_list', 'value'),
              Output('display_table', 'options')],
             [Input('hard_coded', 'value')])
def hard_code_cov(hard_coded):
    if hard_coded:
        countries = ['AU', 'CA', 'FR', 'DE', 'JP', 'UK', 'US']
        asset_list = [' '.join(countries)]
        wts_prior = pd.Series([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615], index=countries, name='wts_prior')
        rhos_prior = pd.DataFrame([[1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
                                   [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
                                   [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
                                   [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
                                   [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
                                   [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
                                   [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]], index=countries,
                                  columns=countries)
        vol_prior = pd.DataFrame([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187], index=countries, columns=['vol'])
        sigma_prior = vol_prior @ vol_prior.T * rhos_prior
        checkoptions = []
        return wts_prior.to_json(orient='table'), sigma_prior.to_json(orient='table'), len(
            countries), asset_list, checkoptions


@app.callback([Output('no_views', 'max')],
              [Input('dump_no', 'children')])
def max_views(max):
    return [int(max)]


@cg.callback([Output('wts_table_container', 'style'),
              Output('rhos_table_container', 'style'),
              Output('vol_table_container', 'style'),
              Output('cov_table_container', 'style'),
              Output('wts_table', 'columns'),
              Output('wts_table', 'data'),
              Output('rhos_table', 'columns'),
              Output('rhos_table', 'data'),
              Output('vol_table', 'columns'),
              Output('vol_table', 'data'),
              Output('cov_table', 'columns'),
              Output('cov_table', 'data'),
              Output('view_params', 'style'),
              Output('dump_no', 'children')],
             [Input('proceed_to_views', 'n_clicks')],
             [State('display_table', 'value'),
              State('asset_list', 'value')])
def upd_visibility(n_clicks, value, asset_list):
    if isinstance(asset_list, list):
        asset_list = asset_list[0]
    asset_list = list(asset_list.split(' '))
    disp_wts = disp_rhos = disp_vol = 'block' if 0 in value else 'none'
    if 1 in value:
        disp_rhos = disp_vol = 'none'
    disp_cov = 'block' if 1 in value else 'none'
    col_wts = [{'id': 'asset_name_wts', 'name': 'Assets'}] + [{'id': 'wts', 'name': 'Prior_weights'}]
    data_wts = [dict(asset_name_wts=asset) for asset in asset_list]
    col_vol = [{'id': 'asset_name_wts', 'name': 'Assets'}] + [{'id': 'vol', 'name': 'Prior_volatility'}]
    data_vol = [dict(asset_name_wts=asset) for asset in asset_list]
    col_rhos = [{'id': 'rhos', 'name': 'Rhos'}] + [{'id': asset, 'name': asset} for asset in asset_list]
    data_rhos = [dict(rhos=asset) for asset in asset_list]
    col_cov = [{'id': 'cov', 'name': 'cov_mat'}] + [{'id': asset, 'name': asset} for asset in asset_list]
    data_cov = [dict(cov=asset) for asset in asset_list]
    if n_clicks and n_clicks > 0:
        return {'display': disp_wts}, {'display': disp_rhos}, {'display': disp_vol}, {'display': disp_cov}, \
               col_wts, data_wts, col_rhos, data_rhos, col_vol, data_vol, col_cov, data_cov, {'display': 'flex',
                                                                                              'justify-content': 'space-evenly',
                                                                                              'padding-top': '25px'}, \
               len(asset_list)


@app.callback([Output('update', 'style'),
               Output('view_table_container', 'style'),
               Output('pick_table_container', 'style'),
               Output('omega_table_container', 'style'),
               Output('view_table', 'columns'),
               Output('view_table', 'data'),
               Output('pick_table', 'columns'),
               Output('pick_table', 'data'),
               Output('omega_table', 'columns'),
               Output('omega_table', 'data')],
              [Input('update_views', 'n_clicks')],
              [State('no_views', 'value'),
               State('asset_list', 'value')])
def upd_view_tables(n_clicks, no_views, asset_list):
    if isinstance(asset_list, list):
        asset_list = asset_list[0]
    asset_list = list(asset_list.split(' '))
    col_views = [{'id': 'views_no', 'name': 'Views'}] + [{'id': 'views', 'name': 'views'}]
    data_views = [dict(views_no=k) for k in range(1, no_views + 1)]
    col_pick = [{'id': 'pick', 'name': 'pick_mat'}] + [{'id': asset, 'name': asset} for asset in asset_list]
    data_pick = [dict(pick=k) for k in range(1, no_views + 1)]
    col_omega = [{'id': 'omega', 'name': 'Omega'}] + [{'id': k, 'name': k} for k in range(no_views)]
    data_omega = [dict(omega=k) for k in range(no_views)]
    if n_clicks and n_clicks > 0:
        return {'display': 'block'}, {'display': 'block'}, {
            'display': 'block'}, {'display': 'block'}, col_views, data_views, col_pick, data_pick, col_omega, data_omega


@app.callback([Output('disp_bl_cov', 'children'),
               Output('disp_omega', 'children'),
               Output('bl_results_table', 'columns'),
               Output('bl_results_table', 'data')],
              [Input('update', 'n_clicks')],
              [State('wts_table', 'data'),
               State('cov_table', 'data'),
               State('rhos_table', 'data'),
               State('vol_table', 'data'),
               State('view_table', 'data'),
               State('pick_table', 'data'),
               State('omega_table', 'data'),
               State('asset_list', 'value'),
               State('display_table', 'value'),
               State('dump_vw_df', 'children'),
               State('dump_wt_df', 'children'),
               State('dump_wt_hc_df', 'children'),
               State('dump_sigma_hc_df', 'children'),
               State('tau', 'value'),
               State('delta', 'value')])
def update_values(n_clicks, wts, cov, rhos, vol, views, pick_mat, omega_mat, asset_list, value, ind_vw_2014,
                  ind_mkt_wts_2014, wts_hc, sigma_hc, tau, delta):
    if isinstance(asset_list, list):
        asset_list = asset_list[0]
    asset_list = list(asset_list.split(' '))
    if ind_vw_2014 is not None and ind_mkt_wts_2014 is not None:
        ind_vw_2014 = pd.read_json(ind_vw_2014, orient='table').to_period('M')
        ind_mkt_wts_2014 = pd.read_json(ind_mkt_wts_2014, orient='table').to_period('M').iloc[0].rename('wts_prior')
        wts_prior = ind_mkt_wts_2014
        # corr_results = sps.get_corr_mat(ind_vw_2014, window=36)
        # corr_mat_group = corr_results[0]
        rhos_prior = ind_vw_2014.corr()
        vol_prior = sps.get_ann_vol(ind_vw_2014)
        sigma_prior = vol_prior @ vol_prior.T * rhos_prior
    elif wts_hc is not None and sigma_hc is not None:
        wts_prior = pd.read_json(wts_hc, orient='table').squeeze()
        sigma_prior = pd.read_json(sigma_hc, orient='table')
    else:
        wts_prior = pd.DataFrame(wts).set_index('asset_name_wts').squeeze().rename('wts_prior').fillna(0)
        wts_prior = wts_prior.astype('float')
        rhos_prior = 0
        vol_prior = 0
        if 1 not in value:
            rhos_prior = pd.DataFrame(rhos, columns=[asset for asset in asset_list]).astype('float')
            vol_prior = pd.DataFrame(vol)
            vol_prior = vol_prior.astype('float').fillna(0)
            sigma_prior = vol_prior @ vol_prior.T * rhos_prior
        else:
            sigma_prior = pd.DataFrame(cov, columns=[asset for asset in asset_list]).astype('float').fillna(0)
    views = pd.DataFrame(views)['views']
    views = views.astype('float').fillna(0)
    pick_df = pd.DataFrame(pick_mat, columns=[asset for asset in asset_list], index=views.index).astype('float').fillna(
        0)
    omega_df = pd.DataFrame(omega_mat).astype('float').fillna(0)
    omega_df.drop(axis=1, columns=['omega'], inplace=True)
    pi, bl_mu, bl_sigma, wts, omega_df = get_bl_results(wts_prior, sigma_prior, delta, tau, pick_df, views,
                                                        omega_df, scale=False, wts_he=True)
    bl_results_df = pd.DataFrame({'Asset': asset_list,
                                  'Cur_wts': wts_prior,
                                  'Equil_wts': wts[0],
                                  'Wts_Optimal': wts[1],
                                  'Wts_Diff': wts[2],
                                  'pi': pi,
                                  'bl_mu': bl_mu})
    float_cols = ['Cur_wts', 'Equil_wts', 'Wts_Optimal', 'Wts_Diff', 'pi', 'bl_mu']
    for col in float_cols:
        bl_results_df[col] = percent_format(bl_results_df[col])
    disp_bl_cov = 'Posterior cov: \n{}'.format(bl_sigma)
    disp_omega = 'Omega Matrix: \n{}'.format(omega_df)
    bl_results_columns = [{'name': col, 'id': col} for col in bl_results_df.columns]
    bl_results_data = bl_results_df.to_dict('records')
    if n_clicks and n_clicks > 0:
        return disp_bl_cov, disp_omega, bl_results_columns, bl_results_data


cg.register(app)
app.run_server()
