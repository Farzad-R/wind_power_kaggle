import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from IPython.display import display


class WpEDS:
    def __init__(self, df_wp, df_ws):
        """
        WpAnalysis class expects to recieve two dataframes
        """
        self.df_wp = df_wp
        self.df_ws = df_ws
        self.data = self.__merge_power_wind()

    def __merge_power_wind(self):
        df_wp_tmp = self.df_wp
        df_wp_tmp["year"] = df_wp_tmp.index.year
        df_wp_tmp["month"] = df_wp_tmp.index.month
        df_wp_tmp["day"] = df_wp_tmp.index.day
        df_wp_tmp["hour"] = df_wp_tmp.index.hour

        df_ws_tmp = self.df_ws
        df_ws_tmp["year"] = df_ws_tmp.index.year
        df_ws_tmp["month"] = df_ws_tmp.index.month
        df_ws_tmp["day"] = df_ws_tmp.index.day
        df_wind_proc = df_ws_tmp[
            df_ws_tmp["hors"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        ]
        df_wind_proc["hors"] = df_wind_proc["hors"] - 1
        df_wind_proc = df_wind_proc.reset_index()
        df_wind_proc["timestamp"] += pd.to_timedelta(df_wind_proc.hors, unit="h")
        df_wind_proc = df_wind_proc.set_index("timestamp")
        df_wind_proc = df_wind_proc[["ws"]]

        self.data = pd.merge(df_wp_tmp, df_wind_proc, left_index=True, right_index=True)
        self.data = self.data[["wp", "ws"]]
        return self.data

    def decopmose_signal(
        self,
        column="wp",
        start_range=0,
        end_range=1000,
        model="additive",
        extrapolate_trend="freq",
    ):
        # Multiplicative Decomposition
        result_add = seasonal_decompose(
            self.data[[column]][start_range:end_range],
            model=model,
            extrapolate_trend=extrapolate_trend,
        )
        # Plot
        result_add.plot().suptitle(f"{model} decompose", fontsize=22)
        # plt.savefig("wind_seasonality.png", dpi=300)
        plt.show()

    def check_correlation(self):
        correlation = self.data.corr()
        display(correlation.style.background_gradient(cmap="coolwarm").set_precision(2))

    def check_missing_values(self):
        print("wp number of missing values:", self.df_wp.isna().sum())
        print("ws number of missing values:", self.df_ws.isna().sum())

    def statistics(self):
        print(self.df_wp.describe())
        print(self.df_ws.describe())

    def check_data_range(self):
        print(
            "WP first valid day:",
            self.df_wp.first_valid_index(),
            "|",
            "WP last valid day:",
            self.df_wp.last_valid_index(),
        )
        print(
            "WS first valid day:",
            self.df_ws.first_valid_index(),
            "|",
            "WS last valid day:",
            self.df_ws.last_valid_index(),
        )
        print(f"WP shape: {self.df_wp.shape}")
        print(f"WS shape: {self.df_ws.shape}")

    def plot_wind_power_curve(self):
        # plt.figure(figsize=(25,10))
        plt.plot(self.data["ws"], self.data["wp"], "o", label="Real Power")
        plt.xlabel("wind speed (km/h)", size=15)
        plt.ylabel("Power Production (kw)", size=15)
        plt.title("Wind Farm Power Production Prediction")
        plt.legend(fontsize=15)
        # plt.savefig("wind_power_curve.png", dpi=300)
        plt.show()

    # https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
    def grangers_causation_matrix(self, maxlag=12, test="ssr_chi2test", verbose=False):
        """Check Granger Causality of all possible combinations of the Time series.
        The rows are the response variable, columns are predictors. The values in the table
        are the P-Values. P-Values lesser than the significance level (0.05), implies
        the Null Hypothesis that the coefficients of the corresponding past values is
        zero, that is, the X does not cause Y can be rejected.

        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        """
        variables = self.data.columns
        df = pd.DataFrame(
            np.zeros((len(variables), len(variables))),
            columns=variables,
            index=variables,
        )
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(
                    self.data[[r, c]], maxlag=maxlag, verbose=False
                )
                p_values = [
                    round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)
                ]
                if verbose:
                    print(f"Y = {r}, X = {c}, P Values = {p_values}")
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + "_x" for var in variables]
        df.index = [var + "_y" for var in variables]
        return df

    # https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
    def cointegration_test(self, alpha=0.05):
        """Perform Johanson's Cointegration Test and Report Summary
        Cointegration test helps to establish the presence of a statistically significant connection between
        two or more time series."""
        out = coint_johansen(self.data, -1, 5)
        d = {"0.90": 0, "0.95": 1, "0.99": 2}
        traces = out.lr1
        cvts = out.cvt[:, d[str(1 - alpha)]]

        def adjust(val, length=6):
            return str(val).ljust(length)

        # Summary
        print("Name   ::  Test Stat > C(95%)    =>   Signif  \n", "--" * 20)
        for col, trace, cvt in zip(self.data.columns, traces, cvts):
            print(
                adjust(col),
                ":: ",
                adjust(round(trace, 2), 9),
                ">",
                adjust(cvt, 8),
                " =>  ",
                trace > cvt,
            )

    def adfuller_test(series, signif=0.05, name="", verbose=False):
        """Perform ADFuller to test for Stationarity of given series and print report"""
        r = adfuller(series, autolag="AIC")
        output = {
            "test_statistic": round(r[0], 4),
            "pvalue": round(r[1], 4),
            "n_lags": round(r[2], 4),
            "n_obs": r[3],
        }
        p_value = output["pvalue"]

        def adjust(val, length=6):
            return str(val).ljust(length)

        # Print Summary
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", "-" * 47)
        print(f" Null Hypothesis: Data has unit root. Non-Stationary.")
        print(f" Significance Level    = {signif}")
        print(f' Test Statistic        = {output["test_statistic"]}')
        print(f' No. Lags Chosen       = {output["n_lags"]}')

        for key, val in r[4].items():
            print(f" Critical value {adjust(key)} = {round(val, 3)}")

        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary.")
        else:
            print(
                f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis."
            )
            print(f" => Series is Non-Stationary.")

    # def run_adfuller_test(self):
    #     for name, column in self.data.iteritems():
    #         self.adfuller_test(series=column, name=column.name)
    #         print("\n")

    def show_lag_plot(self, type="acf", column="wp", lag=72):
        if type == "acf":
            plot_acf(self.data[[column]], lags=lag)
            plt.show()
        elif type == "lag_plot":
            lag_plot(self.data[[column]], lag=lag)
            plt.show()

    def plot_violinplot(self):
        mean = self.data.mean()
        std = self.data.std()
        df_std = (self.data - mean) / std
        df_std = df_std.melt(var_name="Column", value_name="Normalized")
        ax = sns.violinplot(x="Column", y="Normalized", data=df_std)
        _ = ax.set_xticklabels(self.data.keys(), rotation=90)
        plt.show()

    def plot_boxplot(self):
        return self.data.boxplot()
