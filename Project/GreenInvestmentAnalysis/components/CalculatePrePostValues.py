import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


class CalculatePrePostValues:

    def __init__(self, pre_event_return_val_df: pd.DataFrame, post_event_return_val_df: pd.DataFrame,
                 alpha_beta_data_frame: pd.DataFrame, benchmark_data_frame: pd.DataFrame):
        self.__pre_data_return_val = pre_event_return_val_df.dropna(axis=1, how='all')
        self.__post_event_return_val_df = post_event_return_val_df.dropna(axis=1, how='all')
        self.__alpha_beta_data_frame = alpha_beta_data_frame.dropna(axis=1, how='all')
        self.__benchmark_data_frame = benchmark_data_frame.dropna(axis=1, how='all')
        self.__alpha_beta_map = self.__calculate_beta_alpha()
        self.__abnormal_return_pre = self.calculate_abnormal_return(self.__pre_data_return_val)
        self.__abnormal_return_post = self.calculate_abnormal_return(self.__post_event_return_val_df)

    def __calculate_beta_alpha(self):
        results = {}
        benchmark_data = self.__benchmark_data_frame.iloc[:,0]

        for ticker in self.__alpha_beta_data_frame.columns:
            security_data = self.__alpha_beta_data_frame[ticker]
            combined_security_benchmark = pd.concat(
                [benchmark_data, security_data],
                join="inner",
                axis=1
            ).dropna()

            if combined_security_benchmark.empty:
                print(f"Warning: No overlapping data for {ticker}")
                results[ticker] = (0.0, 0.0)
                continue

            target = combined_security_benchmark.iloc[:,1]
            feature = combined_security_benchmark.iloc[:,[0]]

            model = LinearRegression().fit(X=feature, y=target)
            alpha = model.intercept_
            beta = model.coef_[0]
            results[ticker] = (alpha, beta)
        return results

    def calculate_abnormal_return(self, return_val_df: pd.DataFrame):
        abnormal_return_dict= {}
        benchmark_data = self.__benchmark_data_frame.iloc[:, 0]
        for ticker in return_val_df.columns:
            alpha = self.__alpha_beta_map[ticker][0]
            beta = self.__alpha_beta_map[ticker][1]
            security_data = return_val_df[ticker]
            combined_security_benchmark = pd.concat(
                [benchmark_data, security_data],
                axis=1,
                join="inner"
            ).dropna()

            if combined_security_benchmark.empty:
                continue
            market_col = combined_security_benchmark.iloc[:,0]
            security_col = combined_security_benchmark.iloc[:,1]
            abnormal_return_val = security_col - (alpha + (beta * market_col))
            abnormal_return_dict[ticker] = abnormal_return_val

        return pd.DataFrame(abnormal_return_dict)

    def return_pre_post_combined_abnormal_values(self) -> pd.DataFrame:
        full_abnormal_value = pd.concat([self.__abnormal_return_pre, self.__abnormal_return_post], axis=0)
        full_abnormal_value.sort_index(inplace=True)
        return full_abnormal_value

    def draw_aar_graph_for_pre_post(self, event_date_str, file_name):

        # 1. Prepare Data
        full_ar_df = pd.concat([self.__abnormal_return_pre, self.__abnormal_return_post], axis=0)
        full_ar_df.sort_index(inplace=True)

        car_df = full_ar_df.cumsum()

        # 2. Calculate Statistics
        n = car_df.shape[1]
        mean_car = car_df.mean(axis=1)
        se_car = car_df.sem(axis=1)
        ci_upper = mean_car + (1.96 * se_car)
        ci_lower = mean_car - (1.96 * se_car)

        # 3. Plotting
        plt.figure(figsize=(12, 6))

        plt.plot(mean_car.index, mean_car, color='black', linewidth=2, label='Mean CAR')

        plt.fill_between(
            mean_car.index,
            ci_lower,
            ci_upper,
            color='gray',
            alpha=0.5,
            label='95% Confidence Interval'
        )

        event_ts = pd.Timestamp(event_date_str)
        plt.axvline(x=event_ts, color='blue', linestyle=':', linewidth=2, label='Event Date')

        plt.axhline(0, color='black', linewidth=0.5, linestyle='-')
        plt.title("Mean CAR with 95% Confidence Intervals", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Abnormal Return (CAR)", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(file_name, dpi=300)  # dpi=300 makes it high resolution

    def get_abnormal_return_pre(self) -> pd.DataFrame:
        return self.__abnormal_return_pre.copy(deep=True)

    def get_abnormal_return_post(self) -> pd.DataFrame:
        return self.__abnormal_return_post.copy(deep=True)

    def get_car_with_post_pre_index(self) -> pd.DataFrame:
        full_ar_df = pd.concat([self.__abnormal_return_pre, self.__abnormal_return_post], axis=0)
        full_ar_df.sort_index(inplace=True)
        car_df = full_ar_df.cumsum()
        car_df["POST"] = 0
        car_df.loc[self.__abnormal_return_post.index, "POST"] = 1

        return car_df



