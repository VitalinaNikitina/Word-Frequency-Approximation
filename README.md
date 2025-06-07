# Documentation
A class for word frequency analysis and frequency distribution approximation 
using the Zipf-Mandelbrot law.

Initialization:
--------------
To create an instance of the Text class, provide the path to a text file:

```
text_analysis = Text("path/to/textfile.txt")
```

Upon initialization, the text is loaded, preprocessed, and analyzed to compute word frequencies 
and ranks.

Class Attributes:
-----------------
### Constants
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `DEFAULT_M0` | `int` | `2000` | Initial value for numerical integration of the distribution tail |
| `DEFAULT_N_KNOTS` | `int` | `500` | Number of knots for numerical integration |
| `DEFAULT_A_START` | `float` | `1.2` | Initial value for Zipf exponent parameter (a) |
| `DEFAULT_Q_START` | `float` | `10.0` | Initial value for Mandelbrot shift parameter (q) |
| `DEFAULT_F_START` | `int` | `1` | Starting frequency index for error calculation |
| `A_BOUNDS` | `list[float]` | `[0.0, 3.0]` | Valid range for parameter a during optimization |
| `Q_BOUNDS` | `list[float]` | `[-1.0, 500.0]` | Valid range for parameter q during optimization |
| `ERROR_VALUE` | `float` | `1e10` | Error value |
| `DEFAULT_EDGE_FOR_CORE` | `int` | `500` | Frequency threshold for two-stage optimization |
| `DEFAULT_INITIAL_M_MAX` | `int` | `600000` | Initial max M |
| `DEFAULT_INITIAL_Q_MAX` | `int` | `20` | Initial max Q |
| `DEFAULT_INITIAL_M_COUNT` | `int` | `20` | Number of variants of initial parameter M estimates |
| `DEFAULT_INITIAL_Q_COUNT` | `int` | `10` | Number of variants of initial parameter Q estimates |
| `DEFAULT_MAX_ERROR_BOUND` | `float` | `10 + 1e-6` | Maximum permissible approximation error |

### Instance Attributes (set during initialization)

* `path` - path to the input text file

* `words` - list of preprocessed words from the text

* `frequencies_empirical` - array of observed word frequencies (descending order)

* `frequencies_range` - theoretical frequency axis (1 to max frequency)

* `ranks_empirical` - empirical ranks (1 to number of unique words)

* `size` - total word count in the text

* `different_words_count` - number of unique words

* `r_nk_empirical` - number of words with frequency ≥ k for each k

* `f_0` - frequency where ranks become greater than DEFAULT_EDGE_FOR_CORE

Methods:
--------
### Core Methods

1. `__init__`(file_path: str)
  
    Loads a text file, preprocesses it, computes word frequencies, and ranks them.

2. `preprocessing`(file_path: str) -> list
  
    Static method to load and clean text: detects encoding, removes block comments, 
    converts to lowercase, and extracts words.

3. `count_greater_or_equal`(frequencies: list) -> np.ndarray
  
    Static method to compute the number of words occurring at least a given number of times.

4. `r_nk_approx`(a: float, q: float, m: float, m_0: int = DEFAULT_M0, n_knots: int = DEFAULT_N_KNOTS) -> np.ndarray
  
    Approximates r(n, k) (the number of words appearing at least k times) 
    using the Zipf-Mandelbrot law.

### Optimization Methods

5. `_ranks_target`(params: list, first_index: int, last_index: int, stage: int, verbose: bool) -> float

    - Objective function for optimization (calculates MSE between empirical and estimated r(n,k))
    - Uses LRU caching for performance

6. `_find_guess`(params: np.ndarray, idx: list, name: str, a_start: float, q_start: float)

    - Finds optimal initial guesses for parameters m or q via grid search

7. `_optimize_and_store`(guess, bounds, args, tol, m_inf)

    - Helper method that runs optimization and stores results

8. `powell_optimization`(a_start: float = DEFAULT_A_START, q_start: float = DEFAULT_Q_START, 
                    rank_start: int = DEFAULT_RANK_START, a_bounds: list = A_BOUNDS, 
                    q_bounds: list = Q_BOUNDS) -> None
  
    Main optimization routine that:
    - Performs two-step optimization (core frequencies first, then others)
    - Handles edge cases (f_0 < 3)
    - Implements fallback for large errors
    Updates multiple result attributes (r_nk_prediction_, estimated_parameters_, approx_std_*)

### Utility Methods

9. `_rank_to_frequency`(prediction)

    Converts predicted r(n,k) values to rank-frequency distribution
    Updates ranks_estimated and frequencies_estimated attributes

Result Attributes (set after optimization):
--------------
* `r_nk_prediction_1/2/full` - various prediction results from optimization

* `estimated_parameters_1/2` - optimized parameters (a, q, m)

* `approx_std_1/2/full` - approximation standard errors

* `ranks_estimated` - estimated ranks from the model

* `frequencies_estimated` - estimated frequencies from the model

Usage Example:
--------------
```
text_analysis = Text("//text_path")
print(text_analysis.different_words_count)  # Number of unique words
text_analysis.powell_optimization()  # Estimate distribution parameters
print(text_analysis.estimated_parameters_1)  # Estimated parameters (a1, q1, M1)
```

