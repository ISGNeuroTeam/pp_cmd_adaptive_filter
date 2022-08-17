import pandas as pd
import numpy as np
from otlang.sdk.syntax import Keyword, Positional, OTLType
from pp_exec_env.base_command import BaseCommand, Syntax
from .utils import input_preprocess
from .filters import FilterLMS, Filter, FilterNLMS, FilterRLS


class AdaptiveFilterCommand(BaseCommand):
    # define syntax of your command here
    syntax = Syntax(
        [
            Positional("raw_signal", required=True, otl_type=OTLType.TEXT),
            Positional("desired_signal", required=True, otl_type=OTLType.TEXT),
            Keyword("type", required=False, otl_type=OTLType.TEXT),
            Keyword("mu", required=False, otl_type=OTLType.NUMERIC),
            Keyword("filter_size", required=True, otl_type=OTLType.NUMERIC),
            Keyword("steps", required=False, otl_type=OTLType.NUMERIC),
            Keyword("mu_start", required=False, otl_type=OTLType.NUMERIC),
            Keyword("mu_end", required=False, otl_type=OTLType.NUMERIC)
        ],
    )
    use_timewindow = False  # Does not require time window arguments
    idempotent = True  # Does not invalidate cache

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        np.random.seed(42)
        self.log_progress('Start adaptive_filter command')
        type = self.get_arg("type").value or "LMS"
        raw_signal_name = self.get_arg("raw_signal").value
        desired_signal_name = self.get_arg("desired_signal").value
        mu = self.get_arg("mu").value
        filter_size = self.get_arg("filter_size").value
        steps = self.get_arg("steps").value or 100
        mu_start = self.get_arg("mu_start").value or 0.01
        mu_end = self.get_arg("mu_end").value or 1.

        raw_signal = df[raw_signal_name].values
        raw_signal = input_preprocess(raw_signal, filter_size)[:-1]
        desired_signal = df[desired_signal_name].values
        desired_signal = desired_signal[filter_size:]

        if mu:
            filter_func = self.get_filter(filter_type=type, filter_size=filter_size, mu=mu)
            output_signal, errors, history_weights = filter_func.run(desired_signal, raw_signal)
        else:
            filter_func = self.get_filter(filter_type=type, filter_size=filter_size, mu=mu)
            err, mu_opt = filter_func.explore_learning(desired_signal, raw_signal,
                                                       mu_start=mu_start, mu_end=mu_end, steps=steps)
            mu_opt = mu_opt[np.argmin(err[~np.isnan(err)])]
            filter_func = self.get_filter(filter_type=type, filter_size=filter_size, mu=mu_opt)
            output_signal, errors, history_weights = filter_func.run(desired_signal, raw_signal)

        df = df.iloc[filter_size:, :]
        df[f"output_{desired_signal_name}"] = output_signal
        df["filter_error"] = errors
        self.log_progress('First part is complete.', stage=1, total_stages=1)


        return df

    @staticmethod
    def get_filter(filter_type: str, filter_size: int, mu: float) -> Filter:
        """

        :param filter_type:
        :param filter_size:
        :param mu:
        :return:
        """
        if filter_type == "LMS":
            return  FilterLMS(filter_size=filter_size, mu=mu, weights="zeros")
        elif filter_type == "NLMS":
            return FilterNLMS(filter_size=filter_size, mu=mu, weights="zeros")
        elif filter_type == "RLS":
            return FilterRLS(filter_size=filter_size, mu=mu, weights="zeros")