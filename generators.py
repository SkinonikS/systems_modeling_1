from __future__ import annotations
import math
import random
import typing
import numpy as np

class Generator:
    def pdf(self, x: float | list[float]) -> float | list[float]:
        pass

    def cdf(self, x: float | list[float]) -> float | list[float]:
        pass

    def rvs(self, size: int = 1) -> float | list[float]:
        pass

    def get_mu(self) -> float:
        pass

    def get_sigma(self) -> float:
        pass

    def _call(self, x: int | float | list[float] | tuple[float, ...], resolver: typing.Callable[[int | float], float]) -> float | list[float]:
        values: list[float] = []

        if isinstance(x, int):
            for i in range(x):
                r = random.random()

                values.append(resolver(r))
        else:
            if isinstance(x, (list, tuple)) == False:
                x = [x]
            
            for i in x:
                i = np.max([0, i])
                
                values.append(resolver(i))

        return values[0] if len(values) == 1 else values

class ExponentialGenerator(Generator):
    def __init__(self, lamb: float) -> None:
        self._lamb = lamb

    def rvs(self, size: int = 1) -> float | list[float]:
        return self._call(size, lambda x: -math.log(1 - x) / self._lamb)

    def pdf(self, x: float | list[float]) -> float | list[float]:
        return self._call(x, lambda x: self._lamb * math.exp(-self._lamb * x))

    def cdf(self, x: float | list[float]) -> float | list[float]:
        return self._call(x, lambda x: 1 - math.exp(-self._lamb * x))
    
    def get_mu(self) -> float:
        return 1 / self._lamb
    
    def get_sigma(self) -> float:
        return np.sqrt(1 / np.power(self._lamb, 2))

class ErlangGenerator(Generator):
    def __init__(self, l: int, lamb: float) -> None:
        self._l = l
        self._lamb = lamb
    
    def rvs(self, size: int = 1) -> float | list[float]:
        def f(_):
            u = [random.random() for _ in range(self._l)]
            return -sum(math.log(1 - u_i) / self._lamb for u_i in u)
        
        return self._call(size, f)

    def pdf(self, x: float | list[float]) -> float | list[float]:
        return self._call(x, lambda x: (self._lamb ** self._l * x ** (self._l - 1) * math.exp(-self._lamb * x)) / math.factorial(self._l - 1))

    def cdf(self, x: float | list[float]) -> float | list[float]:
        return self._call(x, lambda x: 1 - sum((self._lamb ** i * x ** i * math.exp(-self._lamb * x)) / math.factorial(i) for i in range(self._l)))
    
    def get_mu(self) -> float:
        return self._l / self._lamb
    
    def get_sigma(self) -> float:
        return np.sqrt(self._l / np.power(self._lamb, 2))

class PoissonGenerator(ExponentialGenerator):
    def __init__(self, mean: float) -> None:
        super().__init__(1/mean)

class NormalGenerator(Generator):
    def __init__(self, mu: float, sigma: float) -> None:
        self._mu = mu
        self._sigma = sigma

    def rvs(self, size: int = 1) -> float | list[float]:
        def f(_):
            u1 = random.random()
            u2 = random.random()
            z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

            return self._mu + self._sigma * z0

        return self._call(size, f)

    def pdf(self, x: float | list[float]) -> float | list[float]:
        return self._call(x, lambda x: (1 / (self._sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - self._mu) / self._sigma) ** 2))

    def cdf(self, x: float | list[float]) -> float | list[float]:
        return self._call(x, lambda x: 0.5 * (1 + math.erf((x - self._mu) / (self._sigma * math.sqrt(2)))))

    def get_mu(self) -> float:
        return self._mu
    
    def get_sigma(self) -> float:
        return self._sigma

__all__ = [
    'Generator',
    'ExponentialGenerator',
    'ErlangGenerator',
    'PoissonGenerator',
    'NormalGenerator',
]