import scipy.stats as st
import scipy.special as sc
import scipy.optimize as sp

import numpy as np
import re
import chardet

from collections import Counter
from functools import lru_cache

class Text:
    DEFAULT_M0 = 2000            # Initial value to approximate the tail of the distribution by numerical integration
    DEFAULT_N_KNOTS = 500       # Number of knotes for numerical integration
    DEFAULT_A_START = 1.2        # Initial value of parameter a
    DEFAULT_Q_START = 10.0       # Initial value of parameter q
    DEFAULT_F_START = 1      # Starting frequency for error calculation
    DEFAULT_F_FINAL = None      # Final frequency for error calculation, defined in the constructor
    A_BOUNDS = [0.0, 3.0]            # Bounds of parameter a
    Q_BOUNDS = [-1.0, 500.0]          # Bounds of parameter q
    ERROR_VALUE = 1e10           # Error value
    DEFAULT_EDGE_FOR_CORE = 500      # Frequencies edge for two-step optimization
    DEFAULT_INITIAL_M_MAX = 600000      # Initial max M
    DEFAULT_INITIAL_Q_MAX = 20      # Initial max Q
    DEFAULT_INITIAL_M_COUNT = 20      # Number of variants of initial parameter M estimates
    DEFAULT_INITIAL_Q_COUNT = 10      # Number of variants of initial parameter Q estimates
    DEFAULT_MAX_ERROR_BOUND = 10 + 1e-6      # Maximum permissible approximation error

    def __init__(
            self,
            file_path: str,
    ) -> None:
        
        self.path = file_path

        # Preprocess text and extract words
        self.words = Text.preprocessing(file_path)

        # Calculate word frequencies (sorted in descending order)
        words_counter = Counter(self.words)
        self.frequencies_empirical = np.array(sorted(words_counter.values(), reverse=True))

        # Final frequency for error calculation
        Text.DEFAULT_F_FINAL = np.max(self.frequencies_empirical)      

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

        # Calculate frequency such that the ranks for it become greater than edge
        self.f_0 = np.where(self.r_nk_empirical < self.DEFAULT_EDGE_FOR_CORE)[0][0]

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

        for i in range(1, np.max(frequencies) + 1):
            result[i - 1] = np.sum(frequencies >= i)

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

        Note:
        - For `a ≤ 1`, switches to an approximate Riemann zeta summation to avoid divergence.
        - The `invalid='ignore'` context suppresses NaN warnings during `sc.zeta` calls.

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
            y_j = 1 / m + (j + 1 / 2) * (1 / m_0 - 1 / m) / n_knots

            probability = c * (1 / y_j + q) ** -a

            # Compute survival function (SF) for each frequency k
            frequencies_upgrade_axis = self.frequencies_range[:, np.newaxis]
            summands= st.binom.sf(frequencies_upgrade_axis, self.size, probability) * y_j ** -2

            # Integrate using trapezoidal rule
            integrals = np.sum(summands[:, 1:], axis=1) * (1 / m_0 - 1 / m) / n_knots
            return integrals

        # Normalization constant for the Zipf-like distribution
        with np.errstate(invalid='ignore'):
            if a > 1:
                c = 1 / (sc.zeta(a, q + 1) - sc.zeta(a, q + m))
            else:
                c = 0
                for i in range(1, m_0 + 1):
                    c += (i + q) ** -a
                c += ((m + q) ** (1 - a) - (m_0 + 1 + q) ** (1 - a)) / (1 - a)    
                c = 1 / c
        
        # Exact part: tail sum over words with rank <= m_0
        partial_values = np.arange(1, m_0 + 1)
        probabilities = c * (partial_values + q) ** -a
        tail_exact = st.binom.sf(self.frequencies_range[:, np.newaxis], self.size, probabilities).sum(axis=1)

        # Approximate tail (rank > m_0)
        tail_approx = compute_approx_tail()

        return tail_exact + tail_approx

    def _ranks_target(self, params: list, first_index: int, last_index: int, stage: int, verbose: bool) -> float:
        @lru_cache(maxsize=None)
        def cached_rnk_approx(a, q, m):
            return tuple(self.r_nk_approx(float(a), float(q), float(m)))
        
        a, q, m = params

        # Compare model predictions vs empirical ranks (only for index in [first_index : last_index])
        r_nk_estimated = np.array(cached_rnk_approx(a, q, m))
        mse = np.mean(np.square(r_nk_estimated[first_index : last_index]
                                    - self.r_nk_empirical[first_index : last_index]))

        # Print optimization status
        if verbose:
            message = f"Optimization {stage}. Current parameters: a = {a:.6f}, q = {q:.6f}, m = {m}. Current mse: {mse}"
            print(f"\r{message}", end='', flush=True)
        
        if np.isnan(mse):
            return self.ERROR_VALUE
        
        return mse
    
    def _find_guess(self, params: np.ndarray, idx: list, name: str, a_start: float, q_start: float):
        '''
        Finds the optimal initial guess for parameter `m` or `q` by grid search.
        
        Evaluates a range of parameter values and selects the one that minimizes 
        the mean squared error (MSE) between the empirical and estimated r(n,k) 
        over a specified frequency range.

        Prints the best candidate for debugging purposes.
        '''
        v, best_param = np.inf, None      # Initialize minimum MSE as infinity, stores the optimal parameter value

        for p in params:
            # Generate r(n,k) estimates for current parameter candidate
            if name == 'm':
                est = np.array(self.r_nk_approx(a_start, q_start, p))      # Vary `m`
            else:
                est = np.array(self.r_nk_approx(a_start, p, self.different_words_count))      # Vary `q`

            # Calculate MSE over the target frequency range
            mse = np.mean(np.square(est[idx[0] : idx[1]]
                                        - self.r_nk_empirical[idx[0] : idx[1]]))
            
            # Update best parameter if current MSE is lower
            if mse < v:
                v = mse
                best_param = p

        print(f'Best start {name} = {best_param}')
        return best_param
    
    def _optimize_and_store(self, guess, bounds, args, tol=1e-6, m_inf=False):
        '''
        Optimizes parameters to minimize the target function and stores the results.

        Return:
            1. Result of r_nk_approx with optimized parameters
            2. Optimized parameters (with m converted to integer if not infinity)
            3. Square root of final objective value (like RMSE)
        '''
        result = sp.minimize(self._ranks_target, guess, args=args, bounds=bounds, 
                            options={'xtol': tol, 'ftol': tol}, method='Powell')
        params = result.x
        if m_inf: params[2] = np.inf

        return self.r_nk_approx(*params), [params[0], params[1], int(round(params[2]))], np.sqrt(result.fun)

    def powell_optimization(self, a_start: float = DEFAULT_A_START,
                            q_start: float = DEFAULT_Q_START,
                            f_start: int = DEFAULT_F_START,
                            f_final: int = DEFAULT_F_FINAL,
                            a_bounds: list = A_BOUNDS,
                            q_bounds: list = Q_BOUNDS,
                            init_m_bound = DEFAULT_INITIAL_M_MAX,
                            init_q_bound = DEFAULT_INITIAL_Q_MAX,
                            init_m_count = DEFAULT_INITIAL_M_COUNT,
                            init_q_count = DEFAULT_INITIAL_Q_COUNT,
                            max_error_bound = DEFAULT_MAX_ERROR_BOUND,
                            verbose: bool = True) -> None:
        """
        Fit parameters (a, q, m) to empirical data using Powell's method.
        Minimizes MSE between empirical r_nk and estimated r_nk.
        Performs a two-step optimization procedure:

            1. For the 500 most frequent words.
            2. For other words.

        Update object's attributes:

            - `r_nk_prediction_1` with list of approximated r_nk after the first optimization.
            - `estimated_parameters_1` with list of estimated parameters a, q, m after the first optimization.
            - `approx_std_1` with approximation std after the first optimization.
            
            - `r_nk_prediction_2` with list of approximated r_nk after the second optimization.
            - `estimated_parameters_2` with list of estimated parameters a, q, m after the second optimization.
            - `approx_std_2` with approximation std after the second optimization.

            - `r_nk_prediction_full` with list of approximated r_nk after the both optimizations.
            - `full_approx_std` with approximation std after the both optimizations.

        Note:
            - Using caching to speed up calculations.
            - The best initial approximation is selected for the parameter M of the second optimization
              and the parameter Q of the first one.
            - If the approximation is completed with mse >= max_error_bound, 
              set the parameter M = inf for the second optimization and run both optimizations again.
            - In case f_0 < 3 do only one optimization for all frequencies at once. 
              The estimated parameters are saved in `estimated_parameters_1`

        :param a_start: Initial value of parameter a
        :param q_start: Initial value of parameter q
        :param f_start: Starting frequency for error calculation
        :param f_final: Final frequency for error calculation
        :param a_bounds: Bounds of parameter a
        :param q_bounds: Bounds of parameter q
        :param init_m_bound: Initial max M
        :param init_q_bound: Initial max Q
        :param init_m_count: Number of variants of initial parameter M estimates
        :param init_q_count: Number of variants of initial parameter Q estimates
        :param max_error_bound: Maximum permissible approximation error
        :param verbose: True if optimization progress is shown
        """
        
        # Calculate frequency such that the ranks for it become greater than edge
        f_0 = self.f_0

        bounds = (a_bounds, q_bounds, [self.different_words_count, None])

        if f_0 < 3:
            (self.r_nk_prediction_full, self.estimated_parameters_1, self.approx_std_full) = self._optimize_and_store(
            [a_start, q_start, self.different_words_count], bounds, (f_start - 1, f_final, 1, verbose))
        
        else:
            # Find the initial guess for M that minimizes the objective function
            m_start2 = self._find_guess(np.linspace(self.different_words_count, init_m_bound, init_m_count),
                                [f_start - 1, f_0], 'm', a_start, q_start)

            # Find the initial guess for Q that minimizes the objective function
            q_start1 = self._find_guess(np.linspace(self.Q_BOUNDS[0], init_q_bound, init_q_count), 
                                [f_0, f_final], 'q', a_start, q_start)
            
            # Optimize using Powell's method with bounds for higher frequencies
            bounds = (a_bounds, q_bounds, [self.different_words_count, None])

            (self.r_nk_prediction_1, self.estimated_parameters_1, self.approx_std_1) = self._optimize_and_store(
                [a_start, q_start1, self.different_words_count], bounds, (f_0, f_final, 1, verbose))
            
            print('\n')

            # Optimize using Powell's method with bounds for other frequencies
            (self.r_nk_prediction_2, self.estimated_parameters_2, self.approx_std_2) = self._optimize_and_store(
                [a_start, q_start, m_start2], bounds, (f_start - 1, f_0, 2, verbose))
            
            # Combine results
            self.r_nk_prediction_full = np.concatenate((
                self.r_nk_prediction_2[:f_0], 
                self.r_nk_prediction_1[f_0 : f_final]))
            
            self.full_approx_std = np.sqrt(np.mean(np.square(
                self.r_nk_prediction_full - self.r_nk_empirical)[f_start - 1 : f_final]))

            # Fallback if error is too large
            if self.full_approx_std >= max_error_bound:

                (self.r_nk_prediction_1, self.estimated_parameters_1, self.approx_std_1) = self._optimize_and_store(
                    [a_start, q_start1, self.different_words_count], bounds, (f_0, f_final, 1, verbose), 1e-2)
                
                (self.r_nk_prediction_2, self.estimated_parameters_2, self.approx_std_2) = self._optimize_and_store(
                    [a_start, q_start, np.inf], bounds, (f_start-1, f_0, 2, verbose), 1e-2, True)
                
                self.r_nk_prediction_full = np.concatenate((
                    self.r_nk_prediction_2[:f_0], 
                    self.r_nk_prediction_1[f_0 : f_final]))
                self.full_approx_std = np.sqrt(np.mean(np.square(
                    self.r_nk_prediction_full - self.r_nk_empirical)[f_start - 1 : f_final]))
            
        #Convert ranks to frequencies
        self._rank_to_frequency(self.r_nk_prediction_full)

    def _rank_to_frequency(self, prediction):
        '''
        Converts predicted r(n,k) values (number of words with frequency ≥ k) 
        into a rank-frequency distribution (lists of ranks and corresponding frequencies).

        This transformation enables direct comparison with empirical rank-frequency data.
        The method updates the object's attributes `ranks_estimated` and `frequencies_estimated`.

        :param prediction: Array of predicted r(n,k) values, where
                   prediction[k] = estimated number of words occurring ≥ (k+1) times.
        '''
        r_nk_est = prediction

        # Round estimated values to integers for frequency counts
        r_nk_est_rounded = np.round(r_nk_est, 0)

        # Calculate exact word counts per frequency
        # Difference between consecutive r(n,k) values gives count of words at each frequency
        exact_counts = (r_nk_est_rounded[:-1] - r_nk_est_rounded[1:]).astype(int)
        exact_counts = np.append(exact_counts, r_nk_est_rounded[-1])  # Add last frequency count

        # Reconstruct rank-frequency distribution
        ranks_new = []
        freq_new = []
        current_rank = 1
        
        # For each frequency level, create corresponding ranks
        for freq, count in zip(range(1, len(exact_counts) + 1), exact_counts.astype(int)):
            ranks_new.extend(range(current_rank, current_rank + count))  # Assign ranks
            freq_new.extend([freq] * count)  # Repeat frequency for each word
            current_rank += count  # Increment rank counter

        self.ranks_estimated = ranks_new
        self.frequencies_estimated = freq_new[::-1]
