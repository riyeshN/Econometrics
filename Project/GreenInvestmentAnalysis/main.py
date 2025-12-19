import pandas as pd

from GreenInvestmentAnalysis.components.CalculatePrePostValues import CalculatePrePostValues
from GreenInvestmentAnalysis.components.Util import Util
from GreenInvestmentAnalysis.components.YahooFinanceApi import YahooFinanceApi

tickers_sus = [
    "INTC", "BBY", "PVH", "PFG", "CARR", "CBRE", "AMAT", "RSG", "APD", "ON",
    "ATR", "NSIT", "EPAM", "MKC", "BK", "A", "SON", "ECL", "CLX", "WAT",
    "NVDA", "XYL", "CL", "TROW", "WDAY", "PG", "TXN", "NKE", "AWK", "JLL",
    "KMB", "ADP", "WMT", "MAN", "OC", "RHI", "BLK", "TJX", "LOW", "AMN",
    "MSFT", "NEP", "ADBE", "ES", "BAC", "INTU", "CRUS", "CMI", "VFC", "WST",
    "SPGI", "WCC", "HPQ", "BALL", "ADSK", "MORN", "EW", "C", "RL", "GWW",
    "TFC", "TGT", "FICO", "DAR", "MRK", "FSLR", "GIS", "ALL", "EMN", "LEVI",
    "EL", "KEYS", "AVNT", "VZ", "AMG", "JWN", "HIG", "HPE", "HD", "OSK",
    "LRCX", "PRU", "SWK", "VMW", "SBUX", "TDC", "EXLS", "XEL", "RF", "CPB",
    "COLM", "AMGN", "MCO", "WSM", "MTD"
]


# CONTROL GROUP: Here I utilized LLM's help to help me find matching sector and size tickers based on
# my sustainable group. Went through several iterations (due to unbalanced security
# but got to this which felt more balanced
# still need to evaluate if use of LLM like this is a bad or good practice.
control_tickers = [
    "AMD", "ORCL", "CRM", "QCOM", "IBM", "NOW", "ACN", "CSCO", "AVGO", "MU",
    "APH", "ADI", "KLAC", "SNPS", "CDNS", "PANW", "NXPI", "MSI", "TEL", "IT",
    "GLW",
    "AMZN", "TSLA", "MCD", "BKNG", "MAR", "ORLY", "CMG", "HLT", "AZO", "ROST",
    "EXPE", "LULU",
    "CAT", "GE", "UNP", "HON", "UPS", "ETN", "BA", "LMT", "DE", "MMM",
    "CSX", "NSC", "ITW",
    "JPM", "V", "MA", "WFC", "GS", "MS", "AXP", "CB", "MMC", "PGR",
    "AON", "USB", "ICE", "MET",
    "KO", "PEP", "COST", "PM", "MO", "MDLZ", "K", "SYY", "KHC", "HSY",
    "ADM",
    "UNH", "JNJ", "LLY", "ABBV", "TMO", "ABT", "PFE", "DHR",
    "LIN", "FCX", "SHW", "NEM", "DOW", "PPG", "CTVA",
    "NEE", "SO", "DUK", "AEP",
    "AMT", "PLD",
    "GOOGL"
]

def create_pre_post_object(tickers) -> CalculatePrePostValues:
    alpha_beta_data = YahooFinanceApi(tickers, "2021-06-28", "2022-06-29")
    pre_event = YahooFinanceApi(tickers, "2022-06-29", "2022-07-27")
    post_event = YahooFinanceApi(tickers, "2022-07-26", "2022-08-24")

    benchmark_data = YahooFinanceApi(["^GSPC"], "2021-06-28", "2022-08-24")

    pre_event.initiate()
    post_event.initiate()
    alpha_beta_data.initiate()
    benchmark_data.initiate()

    pre_post_value = CalculatePrePostValues(
        pre_event_return_val_df= pre_event.get_security_value_percent_change_df(),
        post_event_return_val_df=post_event.get_security_value_percent_change_df(),
        alpha_beta_data_frame=alpha_beta_data.get_security_value_percent_change_df(),
        benchmark_data_frame=benchmark_data.get_security_value_percent_change_df()
    )

    return pre_post_value

def create_linear_regression_dummy(sus_obj:CalculatePrePostValues, cntr_obj:CalculatePrePostValues):
    sus_df_car = sus_obj.get_car_with_post_pre_index()
    cntr_obj_car = cntr_obj.get_car_with_post_pre_index()
    sus_melted = sus_df_car.reset_index().melt(
        id_vars=["Date", "POST"],
        var_name = 'Tickers',
        value_name = 'CAR'
    )

    cntr_melted = cntr_obj_car.reset_index().melt(
        id_vars=["Date", "POST"],
        var_name='Tickers',
        value_name='CAR'
    )

    sus_melted["SUS"] = 1
    cntr_melted["SUS"] = 0

    linear_regression_dummy = pd.concat([sus_melted, cntr_melted], axis=0)
    linear_regression_dummy["POST_SUS"] = linear_regression_dummy["POST"] * linear_regression_dummy["SUS"]
    return linear_regression_dummy




sus_obj = create_pre_post_object(tickers_sus)
control_obj = create_pre_post_object(control_tickers)
sus_obj_abnormal_value_df = sus_obj.return_pre_post_combined_abnormal_values()
control_obj_abnormal_val_df = control_obj.return_pre_post_combined_abnormal_values()
Util.create_abnormal_graph(sus_obj_abnormal_value_df, control_obj_abnormal_val_df,
                      "2022-07-27", "data/graph.png")
linear_regression_dummy = create_linear_regression_dummy(sus_obj, control_obj)
Util.linear_regression(
    linear_regression_dummy, feature=["SUS", "POST", "POST_SUS"],
    target="CAR",
    filename= "data/ols_result.png"
)
