import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

class Util:

    @staticmethod
    def create_abnormal_graph(sus_data_frame: pd.DataFrame, control_data_frame: pd.DataFrame,
                              eventDate: str, file_name: str):

        running_total_abnormal_sus_df: pd.DataFrame = sus_data_frame.cumsum()
        running_total_abnormal_control_df: pd.DataFrame = control_data_frame.cumsum()

        size_column_sus = running_total_abnormal_sus_df.shape[1]
        mean_car_sus = running_total_abnormal_sus_df.mean(axis=1)
        se_car_sus = running_total_abnormal_sus_df.sem(axis=1)
        ci_upper_sus = mean_car_sus + (1.96 * se_car_sus)
        ci_lower_sus = mean_car_sus - (1.96 * se_car_sus)

        size_column_ctrl = running_total_abnormal_control_df.shape[1]
        mean_car_ctrl = running_total_abnormal_control_df.mean(axis=1)
        se_car_ctrl = running_total_abnormal_control_df.sem(axis=1)
        ci_upper_ctrl = mean_car_ctrl + (1.96 * se_car_ctrl)
        ci_lower_ctrl = mean_car_ctrl - (1.96 * se_car_ctrl)

        print(f"Control Size: {size_column_ctrl} ----- SUS Size: {size_column_sus}")
        plt.figure(figsize=(12, 6))

        plt.plot(mean_car_sus.index, mean_car_sus, label='SUS Mean CAR', color='blue')
        plt.fill_between(mean_car_sus.index, ci_lower_sus, ci_upper_sus,
                         color='blue', alpha=0.2, label='SUS 95% CI')

        plt.plot(mean_car_ctrl.index, mean_car_ctrl, label='Control Mean CAR', color='orange')
        plt.fill_between(mean_car_ctrl.index, ci_lower_ctrl, ci_upper_ctrl,
                         color='orange', alpha=0.2, label='Control 95% CI')


        try:
            plt.axvline(pd.to_datetime(eventDate), color='red', linestyle='--', label=f'Event: {eventDate}')
        except ValueError:
            pass

        plt.title(f'Cumulative Abnormal Returns (CAR) -- Control Size: {size_column_ctrl} ----- Sus Size: {size_column_sus}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Abnormal Returns')
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.savefig(file_name)
        plt.close()

    @staticmethod
    def linear_regression(dummy_df:pd.DataFrame, feature: list[str], target: str, filename: str):
        X = dummy_df[feature]
        X = sm.add_constant(X)

        y = dummy_df[target]

        model = sm.OLS(y, X).fit()
        print(model.summary())

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        ax.text(0.01, 0.99, model.summary().as_text(),
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                fontfamily='monospace')

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return model

