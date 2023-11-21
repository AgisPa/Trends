# Trends

Trends is a Python application designed for in-depth analysis of trend lines present in data. Leveraging the pandas DataFrame, the tool partitions the dataset into subsets with varying sizes, weighted around an increment of interest, enabling a comprehensive and rigorous analysis.

## Features

- **User Interface:** Execute the Trends.py file to launch a popup user interface. Input the path to the CSV file, total size of the dataset, increment of interest, predictive steps, training percentage, and choose plot options (ls, pls, full, none).

- **Efficiency Measures:** The training percentage serves as a metric for predicting data efficiency. Set it to 100 for a purely predictive analysis. If the training range differs from the dataset, the algorithm dynamically adjusts steps to cover the entire set.

- **Statistical Analysis:** The program begins by calculating the normal distribution for each data point based on the variance of each subset at that region.

- **Predictive Algorithms:**
    - **Least Squares (LS):** Calculates the least square for each subset inside the training and extends the range according to the inputted steps.
  
    - **Partitioned Least Squares (PLS):** Applies the formula described in [this article](https://arxiv.org/abs/2006.16202) to each subset. The minimization of the partitioned least square difference is done through an alternating machine learning process using the scipy library. This process minimizes the least square difference using two coefficients (alpha, beta) to describe the linear approximation's inclination. Alpha represents its magnitude, and beta represents its group contribution, set in the range (-1, 1), equal to every other beta in the same segment of the subset. This machine learning process extends beyond the training range using previous predictions to determine trend behavior during predictive steps.

- **Live Progress Bar:** The program prints increments around the specified interest while displaying a live progress bar.

- **Rich Plots:** The tool utilizes matplotlib.pyplot to generate a (2x2) plot containing:
    - Data set
    - Training range
    - Normal distribution
    - Variance
    - Least squares and partitioned least squares algorithm solutions
    - Critical points (defined by the crossing of the upper boundary prediction value of the next step with the current lower one from the normal distribution and the inverse). 

- **Outputs:** The program outputs the predictions of the two algorithms, their weights, and a .csv file in the local folder. The final result is printed at the training + steps * interest point of the dataset predictions.

## Dependencies

- matplotlib
- numpy
- pandas
- alive-progress
- scipy
- tkinter

## Future Implementations

In future versions, we plan to:
- Output style supporting more information.
- Optimize runn time of the code (now ~30 seconds for a data set of size 50).
- Implement a deep learning model for continuous improvement of the machine learning algorithm.

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Trends.git
    cd Trends
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Trends.py file and explore the power of trend analysis!

```python
python Trends.py
```

Feel free to contribute, report issues, or suggest enhancements. Happy trend analysis!
