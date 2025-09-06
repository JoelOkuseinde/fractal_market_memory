'''

Legend:
  ──▶ control or data flow
  Boxes group core functional components
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Fractal-tail and systemic analysis for 1-month return data

This program provides:
  • Price ingestion via EOD API (get_daily_data)
  • Tail and volatility metrics (MetricEvaluator.full_metrics)
  • System feedback and regime complexity (SystemicAnalyzer.analyze_systemics)
  • Tail clustering and reversion pressure (TailReversionAnalyzer.analyze)
  • Optional visualization of price trend and deviations (plot_with_quantification)
  • Structured logging with per-method timing and global execution time

Version:   v3_1
Module:    Adaptive Fractal Analysis
Author:    Joel Okuseinde
Date:      2025-07-27
'''


import pandas as pd
import numpy as np
import requests
from scipy.stats import genpareto, kurtosis, skew
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.linear_model import LinearRegression
import warnings

import os
import time
import logging
import functools
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler
from orch_example.orchestrator import ASSET, API_TOKEN
from orch_example.explanation import adaptive_fractal_explanation
class Config:
    LOG_DIR = "logs"
    SCR_DIR = "screenshots"

class Logger:
    """
    Sets up structured JSON logging (console + rotating file).
    """
    @staticmethod
    def setup() -> logging.Logger:
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        logger = logging.getLogger("adaptive_fractal")
        logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter(json.dumps({
            "time": "%(asctime)s",
            "level": "%(levelname)s",
            "module": "%(module)s",
            "message": "%(message)s"
        }))

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = RotatingFileHandler(f"{Config.LOG_DIR}/adaptive_fractal.log",
                                 maxBytes=5_000_000, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        return logger


def log_method(fn):
    """
    Decorator to log method entry and exit with timing.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        name = f"{self.__class__.__name__}.{fn.__name__}"
        self.logger.info(f"▶ Starting {name}")
        start = time.time()
        result = fn(self, *args, **kwargs)
        elapsed = time.time() - start
        self.logger.info(f"✔ Finished {name} in {elapsed:.2f}s")
        return result
    return wrapper

class MetricEvaluator:
    def __init__(self, data):
        """
        Initialize the MetricEvaluator with historical close price data.

        Parameters:
        - data (pandas.DataFrame): Must contain a 'close' column with price data.
        """
        self.data = data
        self.r = data.close.pct_change().dropna()
        self.rp = self.r * 100
        self.logger = Logger.setup()

    @log_method
    def calculate_evt_metrics(self, last_close, alpha=0.05):
        """
        Calculates extreme value theory (EVT) based metrics for tail risk estimation.

        Parameters:
        - last_close (float): The most recent closing price, used to scale the EVT threshold.
        - alpha (float): The quantile level to define extreme tails (default is 5%).

        Returns:
        - dict: Includes probabilities of extreme returns, tail excesses, EVT threshold,
                asymmetry, and tail Value-at-Risk (VaR).
                Higher tail probabilities and excesses indicate fatter tails and greater tail risk.
                'Asymmetry' shows skewness; positive values imply downside risk dominates.
        """
        if len(self.rp) < 10:
            keys = ['Prob_upper', 'Prob_lower', 'Mean_upper_excess', 'Max_upper_excess',
                    'Mean_lower_excess', 'Max_lower_excess', 'Evt_threshold', 'Asymmetry', 'Var']
            return {k: np.nan for k in keys}
        upper = np.percentile(self.rp, 100 * (1 - alpha))
        lower = np.percentile(self.rp, 100 * alpha)
        eu = self.rp[self.rp > upper] - upper
        el = lower - self.rp[self.rp < lower]
        return {
            'Prob_upper': (self.rp > upper).mean() * 100,  # % of returns in upper extreme
            'Prob_lower': (self.rp < lower).mean() * 100,  # % of returns in lower extreme
            'Mean_upper_excess': eu.mean() if len(eu) > 0 else 0,  # Avg magnitude of extreme positive returns
            'Max_upper_excess': eu.max() if len(eu) > 0 else 0,    # Max observed extreme positive return
            'Mean_lower_excess': el.mean() if len(el) > 0 else 0,  # Avg magnitude of extreme negative returns
            'Max_lower_excess': el.max() if len(el) > 0 else 0,    # Max observed extreme negative return
            'Evt_threshold': last_close * (1 + upper / 100),       # Price level beyond which returns are extreme
            'Asymmetry': el.mean() - eu.mean(),                    # Positive = downside tail risk dominates
            'Var': lower / 100                                     # Historical VaR at alpha confidence level
        }

    @log_method
    def calculate_metrics(self):
        """
        Computes summary statistics and tail metrics for the return distribution.

        Returns:
        - dict: Includes EVT metrics, mean and standard deviation of returns.
                A higher 'ret_std' implies greater overall volatility.
                Positive 'ret_mean' reflects average positive drift in returns.
        """
        res = {}
        if not self.r.empty:
            last = self.data.close.iloc[-1]
            res.update(self.calculate_evt_metrics(last))
            res['ret_mean'] = self.r.mean() * 100       # Mean daily return in %
            res['ret_std'] = self.r.std() * 100         # Std dev of daily returns in %
        else:
            keys = ['Prob_upper', 'Prob_lower', 'Mean_upper_excess', 'Max_upper_excess',
                    'Mean_lower_excess', 'Max_lower_excess', 'Evt_threshold', 'Asymmetry', 'Var',
                    'ret_mean', 'ret_std']
            res.update({k: np.nan for k in keys})
        return res

    @log_method
    def calculate_risk_metrics(self, alpha=0.05):
        """
        Computes historical Value-at-Risk (VaR) and Expected Shortfall (ES) for given quantile.

        Parameters:
        - alpha (float): The left-tail quantile level (default is 5%).

        Returns:
        - tuple: (VaR, ES), where:
            - VaR (float) is the alpha-percent worst return,
            - ES (float) is the average return below the VaR.
            Lower values (more negative) imply greater downside risk.
        """
        if self.r.empty:
            return np.nan, np.nan
        v = np.percentile(self.r, alpha * 100)
        es = self.r[self.r <= v].mean()
        return v, es

    @log_method
    def full_metrics(self, alpha=0.05):
        """
        Computes and aggregates all metrics: return stats, tail risk metrics, VaR and ES.

        Parameters:
        - alpha (float): Quantile threshold for tail risk and VaR/ES computation.

        Returns:
        - dict: Consolidated metrics including return stats, EVT outputs, VaR, and Expected Shortfall.
                'var_hist' and 'es_hist' indicate potential downside losses at the specified confidence level.
        """
        res = self.calculate_metrics()
        v, es = self.calculate_risk_metrics(alpha)
        res['var_hist'], res['es_hist'] = v * 100, es * 100
        return res


class SystemicAnalyzer:
    def __init__(self, data):
        """
        Initializes the SystemicAnalyzer with price data.

        Parameters:
        - data (pandas.DataFrame): A DataFrame containing at least a 'close' column.
        """
        self.data = data
        self.logger = Logger.setup()



    @log_method
    def bateson_feedback(self, window=5):
        """
        Captures feedback asymmetry between positive and negative returns.

        Parameters:
        - window (int): The size of the rolling window for feedback detection (default is 5).

        Returns:
        - float: Mean difference between rolling proportions of up days vs. down days.
                 Positive values suggest reinforcing (positive) feedback dominance.
                 Negative values suggest dampening (negative) feedback effects.
        """
        r = self.data.close.pct_change().dropna()
        pr = (r > 0).rolling(window).mean()
        nr = (r < 0).rolling(window).mean()
        return (pr - nr).mean()

    @log_method
    def requisite_variety(self, n_regimes=2):
        """
        Estimates regime complexity using a Markov-switching model.

        Parameters:
        - n_regimes (int): Number of regimes to model in the Markov regression (default is 2).

        Returns:
        - float: Standard deviation of smoothed regime probabilities.
                 Higher values imply more variability and complexity in regime switching.
                 Returns NaN if the time series is too short or the model fails to converge.
        """
        try:
            ds = self.data.close.diff().dropna()
            if len(ds) < 50:
                return np.nan
            mdl = MarkovRegression(ds, k_regimes=n_regimes)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = mdl.fit(disp=False)
            return res.smoothed_marginal_probabilities.std().mean()
        except:
            return np.nan

    @log_method
    def analyze_systemics(self):
        """
        Runs all systemic analysis methods and aggregates their outputs.

        Returns:
        - dict: Includes 'homeostasis', 'feedback', and 'regime_variety' metrics.
                These values reflect internal system stability, behavioral feedback, and structural complexity.
        """
        return {
            'homeostasis': self.ashby_homeostasis(),
            'feedback': self.bateson_feedback(),
            'regime_variety': self.requisite_variety()
        }


class TailReversionAnalyzer:
    def __init__(self, data, markov_results):
        """
        Initializes the TailReversionAnalyzer with price data and fitted Markov regime results.

        Parameters:
        - data (pandas.DataFrame): Price data with a 'close' column.
        - markov_results (MarkovRegressionResults): Fitted Markov model object.
        """
        self.data = data
        self.markov_results = markov_results
        self.returns = data['close'].pct_change().dropna() * 100
        self.logger = Logger.setup()

    @log_method
    def tail_energy_gradient(self, window=5, alpha=0.05):
        """
        Measures the rate of change in the frequency of extreme returns (tail events).

        Parameters:
        - window (int): Rolling window for counting tail occurrences.
        - alpha (float): Tail cutoff quantile (e.g., 5%).

        Returns:
        - tuple: (upper_gradient, lower_gradient), which indicate how rapidly the extreme up and down moves are changing.
                 Positive values suggest clustering or buildup of tail activity. Values are unbounded.
        """
        upper_tail = self.returns > self.returns.quantile(1 - alpha)
        lower_tail = self.returns < self.returns.quantile(alpha)
        grad_upper = upper_tail.rolling(window).sum().diff()
        grad_lower = lower_tail.rolling(window).sum().diff()
        return grad_upper.mean(), grad_lower.mean()

    @log_method
    def mimetic_tension(self, window=5):
        """
        Captures the difference in crowd-following behavior by comparing rolling up vs. down day ratios.

        Parameters:
        - window (int): Rolling window to compute directional ratios.

        Returns:
        - float: Mean change in the positive-minus-negative return ratio.
                 Higher values suggest rising directional herding or mimetic pressure. Values near zero imply balance.
        """
        pos_ratio = (self.returns > 0).rolling(window).mean()
        neg_ratio = (self.returns < 0).rolling(window).mean()
        feedback = pos_ratio - neg_ratio
        return feedback.diff().mean()

    @log_method
    def dynamic_homeostasis(self, window=10):
        """
        Measures the change in return volatility relative to its rolling average.

        Parameters:
        - window (int): Rolling window used for computing mean and std deviation.

        Returns:
        - float: Mean change in the coefficient of variation (std/mean).
                 High positive values indicate rising instability or stress; near-zero implies stable dynamics.
        """
        rolling_mean = self.returns.rolling(window).mean()
        rolling_std = self.returns.rolling(window).std()
        ratio = rolling_std / rolling_mean
        return ratio.diff().mean()

    @log_method
    def quantum_collapse(self):
        """
        Quantifies the average change in smoothed regime probabilities across time.

        Returns:
        - float: Mean absolute change in Markov regime probabilities.
                 Higher values indicate more frequent regime switching or uncertainty about current regime.
        """
        p = self.markov_results.smoothed_marginal_probabilities
        diff = p.diff().abs()
        return diff.mean().mean()

    @log_method
    def fractal_queue_memory(self, alpha=0.05, lag=10):
        """
        Measures autocorrelation in the presence of extreme events (fractal tail memory).

        Parameters:
        - alpha (float): Quantile threshold to define tails.
        - lag (int): Number of lags to compute autocorrelations over.

        Returns:
        - float: Average autocorrelation of tail event indicators over multiple lags.
                 Values near 0 indicate no memory; values closer to 1 imply persistence/clustering in tail events.
        """
        excess = (self.returns < self.returns.quantile(alpha)) | (self.returns > self.returns.quantile(1 - alpha))
        autocorrs = [excess.autocorr(lag=i) for i in range(1, lag + 1)]
        return np.nanmean(autocorrs)

    @log_method
    def adaptive_tail_probabilities(self, window=20, alpha=0.05):
        """
        Calculates local probabilities of extreme returns over rolling sub-windows.

        Parameters:
        - window (int): Size of the rolling window.
        - alpha (float): Quantile threshold to define extreme values.

        Returns:
        - tuple: (upper_prob, lower_prob), both in [0, 1].
                 Higher values indicate a greater proportion of local tail events, signaling increased risk concentration.
        """
        local_probs = []
        for i in range(window, len(self.returns)):
            sub = self.returns[i - window:i]
            upper_q = sub.quantile(1 - alpha)
            lower_q = sub.quantile(alpha)
            local_probs.append(((sub > upper_q).mean(), (sub < lower_q).mean()))
        upper_prob, lower_prob = np.nanmean(local_probs, axis=0)
        return upper_prob, lower_prob

    @log_method
    def reversion_index(self, metrics):
        """
        Aggregates all tail-based indicators into a single reversion pressure index.

        Parameters:
        - metrics (dict): Dictionary of precomputed tail metrics.

        Returns:
        - float: Sum of absolute tail signal strengths.
                 Higher values reflect more pronounced tail activity, imbalance, or feedback, suggesting reversion risk.
        """
        comps = [
            abs(metrics.get("tail_grad_upper", 0)),
            abs(metrics.get("tail_grad_lower", 0)),
            abs(metrics.get("mimetic_tension", 0)),
            abs(metrics.get("dynamic_homeostasis", 0)),
            abs(metrics.get("quantum_collapse", 0)),
            abs(metrics.get("fractal_queue_memory", 0)),
            metrics.get("adaptive_prob_upper", 0),
            metrics.get("adaptive_prob_lower", 0),
        ]
        return np.nansum(comps)

    @log_method
    def analyze(self):
        """
        Computes and compiles all tail-based and dynamic reversion metrics into a single dictionary.

        Returns:
        - dict: Contains metrics such as 'tail_grad_upper', 'quantum_collapse', 'reversion_index', etc.
                 Useful for diagnosing systemic stress, crowd behavior, and regime sensitivity.
        """
        m = {}
        gu, gl = self.tail_energy_gradient()
        m["tail_grad_upper"], m["tail_grad_lower"] = gu, gl
        m["mimetic_tension"] = self.mimetic_tension()
        m["dynamic_homeostasis"] = self.dynamic_homeostasis()
        m["quantum_collapse"] = self.quantum_collapse()
        m["fractal_queue_memory"] = self.fractal_queue_memory()
        up, lp = self.adaptive_tail_probabilities()
        m["adaptive_prob_upper"], m["adaptive_prob_lower"] = up, lp
        m["reversion_index"] = self.reversion_index(m)
        return m


def get_daily_data(symbol, start_date, end_date, api_token):
    """
    Downloads historical daily close price data for a given financial instrument.

    Parameters:
    - symbol (str): Ticker symbol (e.g., 'AAPL', 'AUDCHF.FOREX') compatible with EOD Historical Data.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - api_token (str): API key for authenticating requests to EODHD.

    Returns:
    - pandas.DataFrame: A DataFrame indexed by datetime, containing a single 'close' column.
      The data is forward- and backward-filled to handle missing days. If the request fails, returns an empty DataFrame.
    """
    url = f'https://eodhd.com/api/eod/{symbol}?api_token={api_token}&from={start_date}&to={end_date}&fmt=json'
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.DataFrame(r.json())
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime').sort_index().asfreq('D', method='ffill')
        return df[['close']].ffill().bfill()
    return pd.DataFrame()


def plot_with_quantification(data, title):
    """
    Visualizes price data with trendlines and standard deviation bands for anomaly detection.

    Parameters:
    - data (pandas.DataFrame): Must contain a 'close' column and datetime index.
    - title (str): Title to display on the plot.

    Returns:
    - None: Displays a matplotlib chart with:
        • Price line and linear trend
        • ±1σ and ±2σ standard deviation bands
        • Highlighted outliers above +1σ (in red) and below –1σ (in purple)

    Interpretation:
    - Points outside ±1σ bands are mild anomalies; outside ±2σ may indicate significant deviation.
    - Helps visually assess price momentum, volatility, and breakout behavior relative to trend.
    """
    df = data.copy().reset_index()
    df['day_num'] = np.arange(len(df))
    X, y = df[['day_num']].values, df['close'].values
    m = LinearRegression().fit(X, y)
    df['trend'] = m.predict(X)
    res = y - df['trend']
    std_res = res.std()
    mean_p = df['close'].mean()
    df['in_upper'] = df['close'] > (df['trend'] + 1 * std_res)
    df['in_lower'] = df['close'] < (df['trend'] - 1 * std_res)
    plt.figure(figsize=(14, 7))
    plt.plot(df['datetime'], df['close'], label='Prix')
    plt.plot(df['datetime'], df['trend'], label='Trend')
    plt.plot(df['datetime'], df['trend'] + std_res, linestyle='--', label='+1σ')
    plt.plot(df['datetime'], df['trend'] - std_res, linestyle='--', label='-1σ')
    plt.plot(df['datetime'], df['trend'] + 2 * std_res, linestyle='-.', label='+2σ')
    plt.plot(df['datetime'], df['trend'] - 2 * std_res, linestyle='-.', label='-2σ')
    plt.scatter(df.loc[df['in_upper'], 'datetime'], df.loc[df['in_upper'], 'close'], c='red')
    plt.scatter(df.loc[df['in_lower'], 'datetime'], df.loc[df['in_lower'], 'close'], c='purple')
    plt.title(f"{title} • Mean: {mean_p:.2f} • σ_res: {std_res:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    logger = Logger.setup()
    start_time = time.time()

    api_token = API_TOKEN
    ticker = ASSET
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    start_date = (pd.to_datetime('today') - pd.DateOffset(months=1)).strftime('%Y-%m-%d')

    data = get_daily_data(ticker, start_date, end_date, api_token)
    if data.empty:
        logger.error("Error retrieving data")
        return

    data_clean = data.dropna(subset=['close'])
    if data_clean.empty:
        logger.error("Insufficient data after cleaning")
        return

    metric_eval = MetricEvaluator(data_clean)
    metrics = metric_eval.full_metrics(alpha=0.05)

    sys = SystemicAnalyzer(data_clean)
    metrics.update(sys.analyze_systemics())

    try:
        ds = data_clean.close.diff().dropna()
        mdl = MarkovRegression(ds, k_regimes=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            markov_results = mdl.fit(disp=False)
    except Exception as e:
        logger.error(f"Markov error: {e}")
        markov_results = None

    if markov_results is not None:
        tra = TailReversionAnalyzer(data_clean, markov_results)
        metrics.update(tra.analyze())

    print("\n=== DAILY ANALYSIS (1 MONTH) ===\n")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")

    # ===CHANGES, write to shared JSON file===
    safe_ticker = ticker.replace('.', '_')
    date_str = datetime.now().strftime("%Y%m%d")
    bundle_path = os.environ.get("ANALYSIS_JSON_PATH", f"analysis_results_{safe_ticker}_{date_str}.json")

    entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "explantation": adaptive_fractal_explanation,
        "metrics": metrics
    }

    try:
        if os.path.exists(bundle_path):
            with open(bundle_path, "r") as fp:
                bundle = json.load(fp)
        else:
            bundle = {}
    except Exception:
        bundle = {}

    bundle["adaptive_fractal"] = entry

    with open(bundle_path, "w") as fp:
        json.dump(bundle, fp, indent=4)

    print(f"\n Wrote adaptive_fractal to {bundle_path}")

    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
