

import pandas as pd
import numpy as np

import requests
import torch
from scipy.stats import genpareto, kurtosis
from sklearn.linear_model import LinearRegression
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os, json, time, logging, functools
import pandas as pd, numpy as np, requests, torch, warnings, matplotlib.pyplot as plt
from orch_example.fractal_tranlator import ASSET, API_TOKEN
from logging.handlers import RotatingFileHandler
from sklearn.linear_model import LinearRegression
from scipy.stats import genpareto, kurtosis
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
# from selenium.webdriver.remote.webdriver import WebDriver

class Config:
    LOG_DIR = "logs"
    SCR_DIR = "screenshots"

class Logger:
    """
    Sets up structured JSON logging (console + rotating file).
    """
    @staticmethod
    def setup() -> logging.Logger:
        """
        Create or retrieve a logger named 'ppi_smoke' that writes INFO+ to console and DEBUG+ to rotating file.
        Returns
        -------
        logging.Logger
        """
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        logger = logging.getLogger("ppi_smoke")
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

        fh = RotatingFileHandler(f"{Config.LOG_DIR}/ppi_smoke.log",
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

class MetricAnalyzer:
    def __init__(self, data, logger: logging.Logger, timeout: int = 20):
        """
        Store a copy of the input DataFrame with a 'close' series.
        Prepares data for metric calculations.

        Parameters
        ----------
        data : pandas.DataFrame
            Must contain a 'close' column.

        Returns
        -------
        None
        """
        self.data = data.copy()
        self.logger = logger

    @log_method
    def quantify_queues_adaptive(self, window_homeo=30, window_fb=5):
        """
        Derive dynamic upper/lower thresholds from rolling variability and short‑term feedback,
        then quantify how often and how far prices breach those thresholds.

        Parameters
        ----------
        window_homeo : int, optional
            Window length for rolling std/mean homeostasis (default 30).
        window_fb : int, optional
            Window length for rolling positive vs negative return feedback (default 5).

        Returns
        -------
        dict
            prob_upper : float
                % of prices above the adaptive upper bound.
                High → frequent upside breaches; low → rare.
            prob_lower : float
                % of prices below the adaptive lower bound.
                High → frequent downside breaches; low → rare.
            mean_upper_excess : float
                Avg overshoot magnitude above upper bound (zero if none).
                Larger → more severe typical upside moves.
            max_upper_excess : float
                Max overshoot above upper bound (zero if none).
                Larger → most extreme single upside move.
            mean_lower_excess : float
                Avg undershoot magnitude below lower bound (zero if none).
                Larger → more severe typical downside moves.
            max_lower_excess : float
                Max undershoot below lower bound (zero if none).
                Larger → most extreme single downside move.
            homeostasis_coef : float
                Average rolling std/mean ratio; measures baseline variability.
                Higher → more relative fluctuation.
            feedback_coef : float
                Mean positive minus negative return rate; measures short‑term bias.
                Positive → more up‑moves, negative → more down‑moves.
        """
        data = self.data
        m, s = data['close'].mean(), data['close'].std()
        rs = data['close'].rolling(window_homeo).std()
        rm = data['close'].rolling(window_homeo).mean()
        homeo = (rs / rm).mean()
        r = data['close'].pct_change().dropna()
        fb_p = r.gt(0).rolling(window_fb).mean().mean()
        fb_m = r.lt(0).rolling(window_fb).mean().mean()
        feedback = fb_p - fb_m
        k_up, k_lo = homeo + feedback, homeo - feedback
        up, lo = m + k_up * s, m - k_lo * s
        in_up = data['close'] > up
        in_lo = data['close'] < lo
        return {
            'prob_upper': in_up.mean() * 100,
            'prob_lower': in_lo.mean() * 100,
            'mean_upper_excess': (data['close'][in_up] - up).mean() if in_up.any() else 0,
            'max_upper_excess': (data['close'][in_up] - up).max() if in_up.any() else 0,
            'mean_lower_excess': (lo - data['close'][in_lo]).mean() if in_lo.any() else 0,
            'max_lower_excess': (lo - data['close'][in_lo]).max() if in_lo.any() else 0,
            'homeostasis_coef': homeo,
            'feedback_coef': feedback
        }

    @log_method
    def calculate_asymmetry_adaptive(self, window_homeo=30, window_fb=5):
        """
        Compute net tilt of large upward vs downward moves using adaptive bounds.
        Positive → average upside overshoot exceeds downside undershoot.

        Parameters
        ----------
        window_homeo : int, optional
            Homeostasis window (default 30).
        window_fb : int, optional
            Feedback window (default 5).

        Returns
        -------
        float
            Difference = (mean(above – upper) − mean(lower − below)).
            Positive → stronger average upside moves; negative → stronger downside.
        """
        data = self.data
        m, s = data['close'].mean(), data['close'].std()
        rs = data['close'].rolling(window_homeo).std()
        rm = data['close'].rolling(window_homeo).mean()
        homeo = (rs / rm).mean()
        r = data['close'].pct_change().dropna()
        fb_p = r.gt(0).rolling(window_fb).mean().mean()
        fb_m = r.lt(0).rolling(window_fb).mean().mean()
        feedback = fb_p - fb_m
        k_up, k_lo = homeo + feedback, homeo - feedback
        up, lo = m + k_up * s, m - k_lo * s
        above = data['close'][data['close'] > up]
        below = data['close'][data['close'] < lo]
        if above.empty or below.empty:
            return 0.0
        return (above - up).mean() - (lo - below).mean()

    @log_method
    def analyze_extreme_values(self, threshold_quantile=0.95):
        """
        Fit a Generalized Pareto distribution to the tail of large positive moves.
        Quantifies shape, location and scale of extreme excursions beyond a high quantile.

        Parameters
        ----------
        threshold_quantile : float, optional
            Quantile for threshold (default 0.95).

        Returns
        -------
        tuple
            params : (shape, loc, scale)
                GPD parameters of excesses above threshold.
                Shape >0 → heavy tail; <0 → finite endpoint.
            threshold : float
                Value at the specified quantile, defines where tail begins.
        """
        data = self.data
        thresh = data['close'].quantile(threshold_quantile)
        exc = data['close'][data['close'] > thresh] - thresh
        if len(exc) < 2:
            return (np.nan, np.nan, np.nan), thresh
        return genpareto.fit(exc), thresh

    @log_method
    def calculate_risk_metrics(self, alpha=0.05):
        """
        Calculate Value at Risk and Expected Shortfall from returns.
        VaR is the alpha‑quantile loss, ES is the average of losses beyond VaR.

        Parameters
        ----------
        alpha : float, optional
            Tail probability (default 0.05).

        Returns
        -------
        tuple
            var : float
                The alpha‑quantile of returns (e.g. -0.05 means 5% worst loss).
            es : float
                Mean of returns ≤ var, measures average tail loss.
        """
        data = self.data
        r = data['close'].pct_change().dropna()
        if r.empty:
            return np.nan, np.nan
        var = np.percentile(r, alpha * 100)
        es = r[r <= var].mean()
        return var, es

    @log_method
    def analyze_metrics(self):
        """
        Run all metrics and collate results.
        Returns adaptive queue stats, extreme‑value threshold, asymmetry, and VaR.

        Returns
        -------
        dict
            All keys from quantify_queues_adaptive (see its return docs), plus:
            evt_threshold : float, threshold used for GPD fit (high quantile of 'close')
            asymmetry : float, adaptive asymmetry result (see its return doc)
            var : float, Value at Risk (see calculate_risk_metrics)
        """
        met = {}
        qqa = self.quantify_queues_adaptive()
        met.update(qqa)
        met.update({
            'evt_threshold': self.analyze_extreme_values()[1],
            'asymmetry': self.calculate_asymmetry_adaptive(),
            'var': self.calculate_risk_metrics()[0]
        })
        return met

class SystemicAnalyzer:
    def __init__(self, data, logger):
        self.data = data.copy()
        self.logger = logger
    @log_method
    def requisite_variety(self, max_regimes=6):
        """
        Compute the Shannon–Wiener entropy (requisite variety) of regime probabilities
        for models selected by AIC and BIC.

        Parameters
        ----------
        max_regimes : int, optional
            Maximum number of regimes to try (default is 6).

        Returns
        -------
        dict
            A dictionary with keys:

            - 'variety_aic' : float
                Shannon entropy of the average smoothed regime probabilities
                under the model with lowest AIC.  Higher values indicate
                a more even spread across regimes (more variety).

            - 'variety_bic' : float
                Shannon entropy of the average smoothed regime probabilities
                under the model with lowest BIC.  Higher values indicate
                a more even spread across regimes (more variety).
        """
        diff = self.data['close'].diff().dropna()
        norm = (diff - diff.mean()) / diff.std()
        best = {'aic': (np.inf, None), 'bic': (np.inf, None)}
        for k in range(2, max_regimes + 1):
            try:
                res = MarkovRegression(norm, k_regimes=k).fit(disp=False)
                aic, bic = res.aic, res.bic
                if aic < best['aic'][0]:
                    best['aic'] = (aic, res)
                if bic < best['bic'][0]:
                    best['bic'] = (bic, res)
            except:
                continue

        out = {}
        for crit in ('aic', 'bic'):
            res = best[crit][1]
            p = res.smoothed_marginal_probabilities.mean(axis=0)
            out[f'variety_{crit}'] = -(p * np.log2(p)).sum()

        return out

    @log_method
    def quantum_collapse(self, max_regimes=6, max_iter=1000):
        """
        Compute the 'collapse' measure (one minus largest regime probability)
        for models selected by AIC and BIC.

        Parameters
        ----------
        max_regimes : int, optional
            Maximum number of regimes to try (default is 6).
        max_iter : int, optional
            Maximum number of EM iterations for fitting (default is 1000).

        Returns
        -------
        dict
            A dictionary with keys:

            - 'quantum_collapse_aic' : float
                One minus the maximum of the average smoothed regime probabilities
                under the model with lowest AIC.  Values near zero indicate
                one regime dominates, values near one indicate no clear regime
                dominance (high uncertainty).

            - 'quantum_collapse_bic' : float
                One minus the maximum of the average smoothed regime probabilities
                under the model with lowest BIC.  Interpretation as above.
        """
        # diff = self.data['close'].diff().dropna()
        # norm = (diff - diff.mean()) / diff.std()
        # best = {'aic': (np.inf, None), 'bic': (np.inf, None)}
        # for k in range(2, max_regimes + 1):
        #     try:
        #         model = MarkovRegression(norm, k_regimes=k, switching_variance=True)
        #         res = model.fit(disp=False, maxiter=max_iter, atol=1e-6)
        #         aic, bic = res.aic, res.bic
        #         if aic < best['aic'][0]:
        #             best['aic'] = (aic, res)
        #         if bic < best['bic'][0]:
        #             best['bic'] = (bic, res)
        #     except:
        #         continue
        #
        # out = {}
        # for crit in ('aic', 'bic'):
        #     res = best[crit][1]
        #     p = res.smoothed_marginal_probabilities.mean(axis=0)
        #     out[f'quantum_collapse_{crit}'] = 1 - p.max()
        #
        # return out
        return {}

    @log_method
    def ashby_homeostasis(self, window=30):
        """
        Compute the Ashby Homeostasis metric for the 'close' price series.

        The Ashby homeostasis metric quantifies the relative stability of a univariate
        time series by averaging its coefficient of variation over a rolling window.
        Each coefficient of variation is sigma_t,n / mu_t,n, where mu_t,n and sigma_t,n
        are the mean and standard deviation over the most recent n observations.

        Parameters
        ----------
        window : int, optional
            Length of the rolling window (number of periods) to compute mean and
            standard deviation, default is 30.

        Returns
        -------
        float
            The average coefficient of variation across all rolling windows.
            Lower values indicate greater stability, higher values reflect
            normalized volatility.
        """
        std = self.data['close'].rolling(window).std()
        mean = self.data['close'].rolling(window).mean()
        return (std / mean).mean()

    @log_method
    def hurst_exponent(self, ts):
        """
        Estimate the Hurst exponent of a univariate time series.

        The Hurst exponent measures the long‑term memory of a time series. Values
        H in (0,0.5) indicate mean‑reverting behavior, H≈0.5 indicates a random walk,
        and H in (0.5,1) indicates persistent, trending behavior. This implementation
        uses a variogram approach: compute tau(lag) = sqrt(std(ts[lag:] - ts[:-lag])),
        fit a line to log(tau) vs log(lag), and set H = 2 * slope, floored at zero.

        Parameters
        ----------
        ts : array‑like of shape (n)
            1D sequence of observations.

        Returns
        -------
        float
            Estimated Hurst exponent, H ≥ 0.
        """
        lags = range(2, min(len(ts) // 2, 20))
        tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return max(poly[0] * 2.0, 0)

    @log_method
    def fractal_memory(self, p=1, q=1, min_data_points=252):
        """
        Estimate the fractal memory parameter of the 'close' price series.

        Fractal memory provides a unified measure of long‑range dependence in volatility.
        Closing prices P_t are converted to percent returns. If the number of observations
        is below min_data_points, the Hurst exponent H is returned, characterizing the
        scaling behavior of volatility (short memory if H≈0, random walk if H≈0.5,
        persistent trends if H>0.5). Otherwise, heavy tails and heteroskedasticity are
        stabilized via the sign‑root transform y_t = sign(r_t) * sqrt(|r_t|), and a
        FIGARCH(p,d,q) model is fitted:
            (1 − L)^d ε_t = ω + [1 − β(L)]^−1 α(L) ε_t,
        where ε_t = y_t. The fractional integration parameter d ∈ (0,1) governs
        the hyperbolic decay of volatility shocks. The estimate of d is clipped
        to [0.01, 0.99] to ensure stationarity. If the FIGARCH fit fails, H is returned.


        Parameters
        ----------
        p : int, optional
            FIGARCH autoregressive order, default is 1.
        q : int, optional
            FIGARCH moving average order, default is 1.
        min_data_points : int, optional
            Minimum number of return observations required to fit FIGARCH, default is 252.

        Returns
        -------
        float
            Estimated long‑memory parameter d from the FIGARCH fit (0.01 ≤ d ≤ 0.99),
            or the Hurst exponent if FIGARCH is not applied or fails.
        """
        r = self.data['close'].pct_change().dropna() * 100
        if len(r) < min_data_points:
            return self.hurst_exponent(r.values)
        try:
            r_t = np.sign(r) * np.sqrt(np.abs(r))
            fit = arch_model(r_t, vol='FIGARCH', p=p, q=q, dist='skewt') \
                .fit(disp='off', update_freq=10)
            d = np.clip(fit.params.get('d', 0), 0.01, 0.99)
            return d
        except:
            return self.hurst_exponent(r.values)

    @log_method
    def shannon_entropy(self, bin_method='sqrt'):
        """
        Compute the Shannon entropy of the return distribution for the 'close' price series.

        Shannon entropy measures the uncertainty in a discrete probability distribution
        derived from time series returns. Returns are binned into a normalized histogram,
        and entropy is calculated as H = -sum(p_k * log2(p_k)), with p_k the probability
        mass in bin k.

        Parameters
        ----------
        bin_method : str, optional
            Method for determining the number of histogram bins. If 'sqrt', use the
            square‑root choice n_bins = int(sqrt(N)). Otherwise use a fixed 10 bins.
            Default is 'sqrt'.

        Returns
        -------
        float
            Shannon entropy in bits. Higher values indicate a more uniform, less
            predictable return distribution; lower values indicate concentration in
            fewer states.
        """
        r = self.data['close'].pct_change().dropna()
        n_bins = int(np.sqrt(len(r))) if bin_method == 'sqrt' else 10
        bins = np.linspace(r.min(), r.max(), n_bins + 1)
        counts = np.histogram(r, bins=bins)[0]
        p = counts[counts > 0] / len(r)
        return -(p * np.log2(p)).sum()


    @log_method
    def mimetic_intensity(self):
        """
        Compute the mimetic intensity (herding) metric for the 'close' price series.

        Mimetic intensity quantifies herding by measuring the autocorrelation between
        returns and the magnitude of lagged returns. A rolling window of length 5 is
        used to compute the correlation at each time step, then averaged over all valid
        windows.

        Returns
        -------
        float
            Average lag-1 autocorrelation between returns and lagged absolute returns.
            Positive values indicate momentum-driven imitation (herding),
            negative values indicate contrarian behavior.
        """
        r = self.data['close'].pct_change().dropna()
        return r.rolling(5).corr(r.shift(1)).mean()

    @log_method
    def persistent_volatility(self, window=30):
        """
        Compute the persistent volatility metric for the 'close' price series.

        Persistent volatility summarizes recent realized volatility by computing the
        rolling standard deviation of returns over a window of specified length and
        then averaging those values.

        Parameters
        ----------
        window : int, optional
            Window length (number of periods) for computing rolling volatility,
            default is 30.

        Returns
        -------
        float
            Mean of rolling standard deviations of returns. Higher values indicate
            more persistent or elevated volatility regimes.
        """
        r = self.data['close'].pct_change().dropna()
        return r.rolling(window).std().mean()

    @log_method
    def liquidity_flux(self, window=30):
        """
        Compute the liquidity flux metric for the 'volume' series.

        Liquidity flux quantifies the typical magnitude of short-term changes in a
        liquidity proxy (trading volume) by averaging absolute period-to-period volume
        differences over a rolling window, then collapsing into a single scalar.

        Parameters
        ----------
        window : int, optional
            Window length (number of periods) for computing rolling mean of volume flux,
            default is 30.

        Returns
        -------
        float
            Average absolute volume change per period. Higher values indicate more
            erratic liquidity conditions. Returns NaN if 'volume' column is missing.
        """
        if 'volume' not in self.data:
            return np.nan
        flux = self.data['volume'].diff().abs().fillna(0)
        return flux.rolling(window).mean().mean()

    @log_method
    def info_asymmetry(self):
        """
        Compute the Information Asymmetry metric for the 'close' price series.

        Information Asymmetry quantifies the degree of extreme return surprises by
        measuring the excess kurtosis of the return distribution. Kurtosis is corrected
        for small-sample bias. Higher values indicate heavier tails and greater
        informational asymmetry relative to a Gaussian benchmark.

        Returns
        -------
        float
            Excess kurtosis of returns.
            Values > 0 signal heavier tails and greater informational asymmetry;
            values ≈ 0 indicate near‑Gaussian tail behavior; values < 0 indicate
            lighter tails than Gaussian.
        """
        r = self.data['close'].pct_change().dropna()
        n = len(r)
        bias = ((n - 1) / (n - 2)) * ((n + 1) / (n - 3)) if n > 3 else 1
        fisher = kurtosis(r, fisher=False, bias=False) * bias - 3
        return fisher

    @log_method
    def analyze_systemics(self):
        """
           Compute a suite of systemic risk and complexity metrics.

           This method aggregates multiple metrics into a single
           dictionary.

           Returns
           -------
           dict
               Dictionary mapping metric names to their computed values:
               - 'requisite_variety': dict, metrics from requisite_variety()
               - 'quantum_collapse': dict, metrics from quantum_collapse()
               - 'homeostasis': float, Ashby homeostasis (stability)
               - 'feedback': float, Bateson feedback (momentum bias)
               - 'fractal_memory': float, long‑memory exponent (d or Hurst)
               - 'entropy': float, Shannon entropy (uncertainty of returns)
               - 'lyapunov': float, Lyapunov exponent proxy (chaos indicator)
               - 'mimetic': float, mimetic intensity (herding indicator)
               - 'vol_memory': float, persistent volatility (avg realized vol)
               - 'liquidity': float, liquidity flux (avg volume change)
               - 'info_asymmetry': float, information asymmetry (excess kurtosis)
           """
        sys = {}
        rv = self.requisite_variety()
        sys.update(rv)
        qc = self.quantum_collapse()
        sys.update(qc)
        sys.update({
            'homeostasis':      self.ashby_homeostasis(),
            'feedback':         self.bateson_feedback(),
            'fractal_memory':   self.fractal_memory(),
            'entropy':          self.shannon_entropy(),
            'lyapunov':         self.lyapunov_proxy(),
            'mimetic':          self.mimetic_intensity(),
            'vol_memory':       self.persistent_volatility(),
            'liquidity':        self.liquidity_flux(),
            'info_asymmetry':   self.info_asymmetry()
        })
        return sys

api_token  = API_TOKEN
ticker_eod = ASSET
end_date   = pd.to_datetime('today').strftime('%Y-%m-%d')

def get_daily_data(symbol, start_date, end_date, api_token, logger):
    """
    Fetch end‑of‑day price and volume for a given symbol from the EODHD API.
    Returns a DataFrame indexed by business‑day datetime with ‘close’ and ‘volume’ columns, or an empty DataFrame on failure.
    """
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    url  = f'https://eodhd.com/api/eod/{symbol}?api_token={api_token}&from={start_date}&to={end_date}&fmt=json'
    resp = requests.get(url)
    if resp.status_code == 200:
        df = pd.DataFrame(resp.json())
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
            df = df.set_index('datetime').asfreq('B').rename_axis(None)
            return df[['close', 'volume']]
    logger.warning(f"No data for {symbol}")
    return pd.DataFrame()

def analyze_temporalities(logger):
    """
    For a set of predefined lookback windows (10 y, 1 y, weekly), fetch data then compute metrics and systemics.
    Returns a dict mapping each period name to {'metrics': …, 'systemics': …} or None if no data.
    """
    temporalities = {
        '10y':    (pd.DateOffset(years=10), 'B'),
        '1y':     (pd.DateOffset(years=1),  'B'),
        'weekly': (pd.DateOffset(years=1),  'W-MON'),
    }
    results = {}
    for name, (offset, freq) in temporalities.items():
        start = (pd.to_datetime(end_date) - offset).strftime('%Y-%m-%d')
        data  = get_daily_data(ticker_eod, start, end_date, api_token, logger)
        if data.empty:
            results[name] = None
            continue
        if freq != 'B':
            data = data.resample(freq).last().ffill()
        data = data[~data.index.duplicated()]
        data.index.freq = data.index.inferred_freq
        data = data.ffill().bfill()

        metrics   = MetricAnalyzer(data, logger).analyze_metrics()
        systemics = SystemicAnalyzer(data, logger).analyze_systemics()
        results[name] = {'metrics': metrics, 'systemics': systemics}

    return results

def analyze_year(logger, day):
    temporalities = {
        '1y': (pd.DateOffset(years=1), 'B'),
    }

    results = {}
    for name, (offset, freq) in temporalities.items():
        start = (pd.to_datetime(day) - offset).strftime('%Y-%m-%d')
        data = get_daily_data(ticker_eod, start, day, api_token, logger)
        if data.empty:
            results[name] = None
            continue
        if freq != 'B':
            data = data.resample(freq).last().ffill()
        data = data[~data.index.duplicated()]
        data.index.freq = data.index.inferred_freq
        data = data.ffill().bfill()

        metrics = MetricAnalyzer(data, logger).analyze_metrics()
        systemics = SystemicAnalyzer(data, logger).analyze_systemics()
        results[name] = {'metrics': metrics, 'systemics': systemics}

    return results

if __name__ == "__main__":

    logger = Logger.setup()
    results = analyze_temporalities(logger)
    # results = analyze_year(logger)
    for period, res in results.items():
        logger.info(f"=== Results for {period} ===")
        if not res:
            logger.warning("No data")
            continue
        logger.info("Metrics:")
        for k, v in res['metrics'].items(): logger.info(f" {k:20s}: {v}")
        logger.info("Systemics:")
        for k, v in res['systemics'].items(): logger.info(f" {k:20s}: {v}")


def plot_with_quantification(data, title):
    """
    Fit and plot a linear trend on the price series, then overlay ±1 σ and ±2 σ bands on the residuals.
    Highlights points outside ±1 σ and shows a Matplotlib figure annotated with mean and residual standard deviation.
    """
    df = data.copy().reset_index().rename(columns={'index':'datetime'})
    df['day_num'] = np.arange(len(df))
    X, y = df[['day_num']].values, df['close'].values
    model = LinearRegression().fit(X, y)
    df['trend'] = model.predict(X)
    resid   = y - df['trend']
    std_res = resid.std()
    mean_p  = df['close'].mean()
    df['in_upper'] = df['close'] > (df['trend'] + std_res)
    df['in_lower'] = df['close'] < (df['trend'] - std_res)
    plt.figure(figsize=(14,7))
    plt.plot(df['datetime'], df['close'],           label='Prix')
    plt.plot(df['datetime'], df['trend'],           label='Trend')
    plt.plot(df['datetime'], df['trend'] + std_res, '--', label='+1σ')
    plt.plot(df['datetime'], df['trend'] - std_res, '--', label='-1σ')
    plt.plot(df['datetime'], df['trend'] + 2*std_res,'-.',label='+2σ')
    plt.plot(df['datetime'], df['trend'] - 2*std_res,'-.',label='-2σ')
    plt.scatter(df.loc[df['in_upper'], 'datetime'], df.loc[df['in_upper'], 'close'],
                marker='o', s=40, label='> +1σ')
    plt.scatter(df.loc[df['in_lower'], 'datetime'], df.loc[df['in_lower'], 'close'],
                marker='o', s=40, label='< -1σ')
    plt.title(f"{title} • Mean: {mean_p:.2f} • σ_res: {std_res:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()



# if __name__ == "__main__":
    # === Your existing demo code goes here ===
    # start_10y = (pd.to_datetime(end_date) - pd.DateOffset(years=10))\
    #                 .strftime('%Y-%m-%d')
    # data_10y  = get_daily_data(ticker_eod, start_10y, end_date, api_token)\
    #                .resample('B').last().ffill()
    # plot_with_quantification(data_10y, 'GSPC – 10 ans')

    # start_1y = (pd.to_datetime(end_date) - pd.DateOffset(years=1))\
    #                .strftime('%Y-%m-%d')
    # data_1y  = get_daily_data(ticker_eod, start_1y, end_date, api_token)\
    #                .resample('B').last().ffill()
    # plot_with_quantification(data_1y, 'GSPC – 1 an')
    #
    # start_weekly = (pd.to_datetime(end_date) - pd.DateOffset(years=1))\
    #                    .strftime('%Y-%m-%d')
    # data_weekly = get_daily_data(ticker_eod, start_weekly, end_date, api_token)\
    #                   .resample('W-MON').last().ffill()
    # plot_with_quantification(data_weekly, 'GSPC – Hebdomadaire')

