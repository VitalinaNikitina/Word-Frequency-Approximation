import scipy.stats as st
import scipy.special as sc
import scipy.optimize as sp

import numpy as np
import re
import chardet

from collections import Counter

class Text:
    DEFAULT_M0 = 400             # Initial value to approximate the tail of the distribution by numerical integration
    DEFAULT_N_KNOTS = 100        # Number of knotes for numerical integration

    DEFAULT_A_START = 1.0        # Initial value of parameter a
    DEFAULT_Q_START = 15.0       # Initial value of parameter q
    DEFAULT_RANK_START = 21      # Starting rank for error calculation
    A_BOUNDS = [1.0, 2.0]            # Bounds of parameter a
    Q_BOUNDS = [-1.0, 50.0]          # Bounds of parameter q
    ERROR_VALUE = 1e10           # Error value

    def __init__(
            self,
            file_path: str,
    ) -> None:
        # Preprocess text and extract words
        self.words = Text.preprocessing(file_path)

        # Calculate word frequencies (sorted in descending order)
        words_counter = Counter(self.words)
        self.frequencies_empirical = np.array(sorted(words_counter.values(), reverse=True))

        # Generate theoretical frequency axis (1, 2, ..., max_freq)
        self.frequencies_range =  np.arange(1, np.max(self.frequencies_empirical) + 1, 1).astype(int)

        # Empirical ranks (1, 2, ..., number_of_unique_words)
        self.ranks_empirical = np.arange(1, len(self.frequencies_empirical) + 1, 1)

        # Total word count in the text
        self.size = sum(words_counter.values())

        # Number of unique words
        self.different_words_count = len(words_counter)

        # Calculate r(n, k): number of words with frequency >= k in text with size n
        self.r_nk_empirical = Text.count_greater_or_equal(self.frequencies_empirical)

    @staticmethod
    def preprocessing(file_path: str) -> list:
        """
        Load and preprocess text file:
        1. Detect encoding automatically.
        2. Remove block comments (e.g., ***** ... *****).
        3. Convert to lowercase.
        4. Extract words (Unicode-aware, excluding numbers).

        :param file_path: Path to txt file.
        :return: Preprocessed text file.
        """
        # Detect file encoding (e.g., UTF-8, Windows-1252)
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()

        # Remove block comments enclosed in *****
        text = re.sub(r'\*{5}.*?\*{5}', '', text, flags=re.DOTALL)

        # Extract words in lowercase (ignoring numbers and punctuation)
        text = text.lower()
        words = re.findall(r'\b[^\W\d_]+\b', text, flags=re.UNICODE)

        return words
    
    @staticmethod
    def count_greater_or_equal(frequencies: np.ndarray) -> np.ndarray[np.int64]:
        """
        For each frequency k, count how many words have frequency >= k.

        Example:\\
        Input: [4, 2, 2, 1] \\
        Output: [4, 3, 1, 1] (for k=1,2,3,4)
        
        :param frequencies: A list of word frequencies in the text, ordered in descending order.
        :return: Array, where result[k] = number of words with frequency ≥ k (for k from 0 to max(frequencies) - 1).
        """
        result = np.zeros(np.max(frequencies))

        for i in range(np.max(frequencies)):
            result[i] = np.sum(frequencies >= i)

        return result.astype(int)

    def r_nk_approx(self, a: float, q: float, m: float, 
                    m_0: int = DEFAULT_M0,
                    n_knots: int = DEFAULT_N_KNOTS) -> np.ndarray:
        """
        Approximate r(n, k) using a Zipf-like distribution with parameters (a, q, m),
        where r(n, k) is the mathematical expectation of number of words that
        appear not less than k times in the text contains n words
        and m is the number of all possible words.
            
        Combines:
        1. Exact calculation for low ranks (rank <= m_0).
        2. Approximation by numerical integration for the rank > m_0.

        :param a: Zipf exponent.
        :param q: Mandelbrot shift.
        :param m: Number of all possible words.
        :param m_0: Initial value to approximate the tail of the distribution by numerical integration.
        :param n_knots: Number of knotes for numerical integration.
        :return: Numpy.ndarray array, where each element result[k] represents
                 the mathematical expectation of the number of words that appear in the text ≥ k times calculated 
                 based on the parameters (a, q, m).
        """
        def compute_approx_tail():
            # Numerical integration setup:
            # Split interval [1/m, 1/m_0] into n_knots points
            j = np.arange(n_knots)
            y_j = 1 / m + j * (1 / m_0 - 1 / m) / n_knots

            probability = c * (1 / y_j + q) ** -a

            # Compute survival function (SF) for each frequency k
            frequencies_upgrade_axis = self.frequencies_range[:, np.newaxis]
            summands= st.binom.sf(frequencies_upgrade_axis - 1, self.size, probability) * y_j ** -2

            # Integrate using trapezoidal rule
            integrals = np.sum(summands[:, 1:], axis=1) * (1 / m_0 - 1 / m) / n_knots
            return integrals

        # Normalization constant for the Zipf-like distribution
        with np.errstate(invalid='ignore'):
            c = 1 / (sc.zeta(a, q + 1) - sc.zeta(a, q + m))
        
        # Exact part: tail sum over words with rank <= m_0
        partial_values = np.arange(1, m_0 + 1)
        probabilities = c * (partial_values + q) ** -a
        tail_exact = st.binom.sf(self.frequencies_range[:, np.newaxis] - 1, self.size, probabilities).sum(axis=1)

        # Approximate tail (rank > m_0)
        tail_approx = compute_approx_tail()

        return tail_exact + tail_approx

    
    def powell_optimization(self, a_start: float = DEFAULT_A_START,
                            q_start: float = DEFAULT_Q_START,
                            rank_start: int = DEFAULT_RANK_START,
                            a_bounds: list = A_BOUNDS,
                            q_bounds: list = Q_BOUNDS) -> None:
        """
        Fit parameters (a, q, m) to empirical data using Powell's method.
        Minimizes MSE between empirical r_nk and estimated r_nk.

        Initializes fields:
         
        - `estimated_parameters` with list of estimated parameters a, q, m.
        - `approx_std` with approximation std.

        :param a_start: Initial value of parameter a
        :param q_start: Initial value of parameter q
        :param rank_start: Starting rank for error calculation
        :param a_bounds: Bounds of parameter a
        :param q_bounds: Bounds of parameter q
        """
        def ranks_target_function(params: list) -> float:
            a, q, m = params
            m = float(round(m))

            # Compare model predictions vs empirical ranks (only for ranks >= rank_start)
            r_nk_estimated = self.r_nk_approx(a, q, m)
            mse_error = np.mean(np.square(r_nk_estimated[rank_start:] - self.r_nk_empirical[rank_start:]))

            # Print optimization status
            message = f"Current parameters: a = {a:.6f}, q = {q:.6f}, m = {m}. Current mse: {mse_error}"
            print(f"\r{message}", end='', flush=True)
            
            if np.isnan(mse_error):
                return self.ERROR_VALUE
            
            return mse_error

        # Initial guess for optimization
        initial_guess = [a_start, q_start, self.different_words_count]

        # Optimize using Powell's method with bounds
        result = sp.minimize(ranks_target_function, 
                             initial_guess, 
                             bounds=(a_bounds, q_bounds, [self.different_words_count, None]), 
                             method='Powell')
        
        # Save fitted parameters
        self.estimated_parameters = [result.x[0], result.x[1], int(round(result.x[2]))]
        self.approx_std = np.sqrt(result.fun)
