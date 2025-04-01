# Documentation
A class for word frequency analysis and frequency distribution approximation 
using the Zipf-Mandelbrot law.

Class Attributes:
-----------------
*`DEFAULT_M0 : int` - initial value for numerical integration of the distribution tail.
    
*`DEFAULT_N_KNOTS : int` - number of knots for numerical integration.
    
*`DEFAULT_A_START : float` - initial value of parameter a.
    
*`DEFAULT_Q_START : float` - initial value of parameter q.
    
*`DEFAULT_RANK_START : int` - starting rank for error calculation.
    
*`A_BOUNDS : list' - bounds for parameter a.
    
*`Q_BOUNDS : list` - bounds for parameter q.
    
*`ERROR_VALUE : float` - error value (e.g., for NaN cases).

Initialization:
--------------
To create an instance of the Text class, provide the path to a text file:

```
text_analysis = Text("path/to/textfile.txt")
```

Upon initialization, the text is loaded, preprocessed, and analyzed to compute word frequencies 
and ranks.

Fields:
--------------
*`frequencies_empirical` — word frequencies (sorted in descending order)

*`frequencies_range` — possible word frequencies in text

*`ranks_empirical` — empirical ranks (1, 2, ..., number_of_unique_words)

*`size` — total word count in the text

*`different_words_count` — number of unique words

*`r_nk_empirical` — number of words with frequency >= k in text with size n

Methods:
--------
*`__init__`(file_path: str)
    Loads a text file, preprocesses it, computes word frequencies, and ranks them.

*`preprocessing`(file_path: str) -> list
    Static method to load and clean text: detects encoding, removes block comments, 
    converts to lowercase, and extracts words.

*`count_greater_or_equal`(frequencies: list) -> np.ndarray
    Static method to compute the number of words occurring at least a given number of times.

*`r_nk_approx`(a: float, q: float, m: float, m_0: int = DEFAULT_M0, n_knots: int = DEFAULT_N_KNOTS) -> np.ndarray
    Approximates r(n, k) (the number of words appearing at least k times) 
    using the Zipf-Mandelbrot law.

*`powell_optimization`(a_start: float = DEFAULT_A_START, q_start: float = DEFAULT_Q_START, 
                    rank_start: int = DEFAULT_RANK_START, a_bounds: list = A_BOUNDS, 
                    q_bounds: list = Q_BOUNDS) -> None
    Estimates the parameters (a, q, m) by minimizing the mean squared error 
    between empirical and approximated r(n, k) values.
    
    Initializes fields:
    
    * `estimated_parameters` with list of estimated parameters a, q, m.
    * `approx_std` with approximation std.

Usage Example:
--------------
```
text_analysis = Text("//text_path")
print(text_analysis.different_words_count)  # Number of unique words
text_analysis.powell_optimization()  # Estimate distribution parameters
print(text_analysis.estimated_parameters)  # Estimated parameters (a, q, m)
```

