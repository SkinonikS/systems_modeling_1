from __future__ import annotations
import typing as t
import scipy.stats as st
import numpy as np
import statsmodels.tsa.stattools as ts
import dataclasses as dc
import generators as gen

# Types
T_TesterResult = t.TypeVar('T_TesterResult' , bound=t.NamedTuple)
T_TestResult = t.TypeVar('T_TestResult')

# Helpers
class TestConfig:
    def __init__(
        self,
        sample_size: int,
        bins: int,
        lags: int,
        alpha: float,
    ) -> None:
        self.sample_size = sample_size
        self.bins = bins
        self.lags = lags
        self.alpha = alpha

        self.__assert_params_are_valid()

    def __assert_params_are_valid(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"The value {self.alpha} is out of the allowed range [0, 1].")

class TestInterface(t.Generic[T_TestResult]):
    def run(
        self,
        sample: list[float],
        generator: gen.Generator,
        config: TestConfig,
    ) -> T_TestResult:
        pass

# KolmagorovSmirnovChiSquareTest
class KolmagorovSmirnovResult(t.NamedTuple):
    critical_value: float
    criteria: float
    p_value: float

class ChiSquareResult(t.NamedTuple):
    critical_value: float
    criteria: float
    p_value: float
    df: int

class KolmagorovSmirnovChiSquareTestResult(t.NamedTuple):
    ks: KolmagorovSmirnovResult
    chi_square: ChiSquareResult

class KolmagorovSmirnovChiSquareTest(TestInterface[KolmagorovSmirnovChiSquareTestResult]):
    def run(
        self,
        sample: list[float],
        generator: gen.Generator,
        config: TestConfig,
    ) -> KolmagorovSmirnovChiSquareTestResult:
        num_bins = int(np.sqrt(config.sample_size))
        observed_counts, bin_edges = np.histogram(sample, bins=num_bins)
        expected_counts = config.sample_size * np.diff(generator.cdf(bin_edges.tolist()))

        while np.any(expected_counts < 5):
            num_bins -= 1
            observed_counts, bin_edges = np.histogram(sample, bins=num_bins)
            expected_counts = config.sample_size * np.diff(generator.cdf(bin_edges.tolist()))

        chi2_criteria, chi2_p_value = st.chi2_contingency([observed_counts, expected_counts])[0:2]
        chi2_df = num_bins - 1
        chi_critical_value = st.chi2.ppf(1 - config.alpha, chi2_df)

        ks_stat, ks_p_value = st.kstest(sample, lambda x: generator.cdf(x.tolist()))
        ks_criteria = ks_stat * np.sqrt(config.sample_size)
        ks_critical_value = np.sqrt(-0.5 * np.log(config.alpha / 2))

        return KolmagorovSmirnovChiSquareTestResult(
            chi_square=ChiSquareResult(
                criteria=chi2_criteria,
                critical_value=chi_critical_value,
                p_value=chi2_p_value,
                df=chi2_df,
            ),
            ks=KolmagorovSmirnovResult(
                criteria=ks_criteria,
                critical_value=ks_critical_value,
                p_value=ks_p_value,
            )
        )

# AutocorrelationFunctionTest
class AutocorrelationFunctionResult(t.NamedTuple):
    upper_confidence: list[float]
    lower_confidence: list[float]
    acf_values: t.Any
    lags: list[int]

class AutocorrelationFunctionTest(TestInterface[AutocorrelationFunctionResult]):
    def run(
        self,
        sample: list[float],
        generator: gen.Generator,
        config: TestConfig,
    ) -> AutocorrelationFunctionResult:
        acf = ts.acf(sample, nlags=config.lags, fft=True)
        lags = list(range(1, len(acf)))
        conf_interval = 1.96 / np.sqrt(config.sample_size)

        upper_confidence = [conf_interval] * (lags[-1])
        lower_confidence = [-conf_interval] * (lags[-1])
        acf_values = acf[1:]

        return AutocorrelationFunctionResult(
            upper_confidence=upper_confidence,
            lower_confidence=lower_confidence,
            acf_values=acf_values,
            lags=lags,
        )

# CompareParametersTest
class CompareParametersResult(t.NamedTuple):
    actual_mu: float
    actual_sigma: float
    expected_mu: float
    expected_sigma: float

class CompareParametersTest(TestInterface[CompareParametersResult]):
    def run(
        self,
        sample: list[float],
        generator: gen.Generator,
        config: TestConfig,
    ) -> CompareParametersResult:
        actual_mu, actual_sigma = np.mean(sample), np.std(sample)
        expected_mu, expected_sigma = generator.get_mu(), generator.get_sigma()

        return CompareParametersResult(
            actual_mu=actual_mu,
            actual_sigma=actual_sigma,
            expected_mu=expected_mu,
            expected_sigma=expected_sigma,
        )

# DistributionTest
class DistributionTestResult(t.NamedTuple):
    hist: list
    bin_edges: list
    x: list
    y: list

class DistributionTest(TestInterface[DistributionTestResult]):
    def run(
        self,
        sample: list[float],
        generator: gen.Generator,
        config: TestConfig,
    ) -> DistributionTestResult:
        hist, bin_edges = np.histogram(sample, bins=config.bins, density=True)

        x = np.linspace(start=min(sample), stop=max(sample), num=config.sample_size)
        y = generator.pdf(x.tolist())

        return DistributionTestResult(hist=hist.tolist(), bin_edges=bin_edges.tolist(), x=x.tolist(), y=y)

# Tester
@dc.dataclass
class TesterResult(t.Generic[T_TesterResult]):
    sample: list[float]
    results: T_TesterResult

class Tester(t.Generic[T_TesterResult]):
    def __init__(
        self,
        result_factory: t.Callable[[t.NamedTuple], T_TesterResult],
        tests: dict[str, TestInterface[t.Any]],
        generators: dict[str, gen.Generator],
    ) -> None:
        self._result_factory = result_factory
        self._tests = tests
        self._generators = generators

    def run_tests(
        self,
        config: TestConfig,
    ) -> dict[str, TesterResult[T_TesterResult]]:
        final_results: dict[str, TesterResult[T_TesterResult]] = {}

        for g_key, generator in self._generators.items():
            sample = generator.rvs(config.sample_size)
            results: dict[str, t.NamedTuple] = {}

            for t_key, test in self._tests.items():
                results[t_key] = test.run(
                    sample=sample,
                    generator=generator,
                    config=config,
                )
                
            final_results[g_key] = TesterResult(
                sample=sample,
                results=self._result_factory(results),
            )
        
        return final_results

__all__ = [
    'KolmagorovSmirnovChiSquareTest'
    'AutocorrelationFunctionTest'
    'CompareParametersTest'
    'DistributionTest',
    'KolmagorovSmirnovResult',
    'ChiSquareResult',
    'KolmagorovSmirnovChiSquareTestResult',
    'AutocorrelationFunctionResult',
    'CompareParametersResult',
    'DistributionTestResult',
    'Tester',
    'TestConfig',
    'TestInterface',
]