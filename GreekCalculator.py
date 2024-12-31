import numpy as np
from scipy.stats import norm


class GreekCalculator:
    """
    A class to calculate the Greeks for European options using the Black-Scholes model.
    """

    def __init__(self, Option_type, S, K, sigma, r, T, q):
        """
        Initialize the GreekCalculator with the given parameters.

        Parameters:
        - Option_type: 'C' for call, 'P' for put
        - S: Current stock price
        - K: Strike price
        - sigma: Volatility
        - r: Risk-free interest rate
        - T: Time to maturity (in years)
        - q: Dividend yield
        """
        self.Option_type = Option_type
        self.S = S
        self.K = K
        self.sigma = sigma
        self.r = r
        self.T = T
        self.q = q

    def delta(self):
        """
        Calculate the delta of the option.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * (self.sigma ** 2)) * self.T) / (
                    self.sigma * np.sqrt(self.T))
        N_d1 = norm.cdf(d1)

        if self.Option_type == 'C':
            return np.exp(-self.q * self.T) * N_d1
        else:
            return np.exp(-self.q * self.T) * (1 - N_d1)

    def gamma(self):
        """
        Calculate the Gamma of the option.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * (self.sigma ** 2)) * self.T) / (
                    self.sigma * np.sqrt(self.T))
        pdf_d1 = norm.pdf(d1)
        return np.exp(-self.q * self.T) * pdf_d1 / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """
        Calculate the Vega of the option.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * (self.sigma ** 2)) * self.T) / (
                    self.sigma * np.sqrt(self.T))
        pdf_d1 = norm.pdf(d1)
        return self.S * np.exp(-self.q * self.T) * pdf_d1 * np.sqrt(self.T)

    def theta(self):
        """
        Calculate the Theta of the option.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * (self.sigma ** 2)) * self.T) / (
                    self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        pdf_d1 = norm.pdf(d1)
        cdf_d2 = norm.cdf(d2)

        if self.Option_type == 'C':
            theta = -self.S * pdf_d1 * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(
                -self.r * self.T) * cdf_d2
        else:
            cdf_neg_d2 = norm.cdf(-d2)
            theta = -self.S * pdf_d1 * self.sigma / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(
                -self.r * self.T) * cdf_neg_d2
        return theta

    def rho(self):
        """
        Calculate the Rho of the option.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * (self.sigma ** 2)) * self.T) / (
                    self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d2 = norm.cdf(-d2)

        if self.Option_type == 'C':
            rho = self.T * self.K * np.exp(-self.r * self.T) * cdf_d2
        else:
            rho = -self.T * self.K * np.exp(-self.r * self.T) * cdf_neg_d2
        return rho

    def psi(self):
        """
        Calculate the Psi of the option.
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * (self.sigma ** 2)) * self.T) / (
                    self.sigma * np.sqrt(self.T))
        cdf_d1 = norm.cdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)

        if self.Option_type == 'C':
            psi = -self.T * self.S * np.exp(-self.q * self.T) * cdf_d1
        else:
            psi = self.T * self.S * np.exp(-self.q * self.T) * cdf_neg_d1
        return psi

    def get_all_greeks(self):
        """
        Calculate and return all Greeks in a dictionary.
        """
        return {
            'Delta': self.delta(),
            'Gamma': self.gamma(),
            'Vega': self.vega(),
            'Theta': self.theta(),
            'Rho': self.rho(),
            'Psi': self.psi()
        }

if __name__ == '__main__':
    # Example usage:
    option = GreekCalculator('C', 100, 100, 0.2, 0.05, 1, 0.02)
    print(option.get_all_greeks())