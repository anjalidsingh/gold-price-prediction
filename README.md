## Colab Notebook

For an interactive view of the code, graphs, and results, check out the Colab notebook: [Gold Price Prediction Colab Notebook]([[https://colab.research.google.com/drive/1bNHktSwJ2VqMq9kOuDfENokZgApmLPK9?usp=sharing](https://colab.research.google.com/drive/1bNHktSwJ2VqMq9kOuDfENokZgApmLPK9?usp=sharing)])

# Gold Price Prediction

This project aims to predict the price of gold using various economic indicators. The prediction is done using a Random Forest Regressor model.

## Dataset

The dataset used for this project is `gld_price_data.csv`, which contains the following columns:
- `Date`: The date of the record (not used in prediction)
- `SPX`: S&P 500 index
- `USO`: United States Oil Fund price
- `SLV`: Silver ETF price
- `EUR/USD`: Exchange rate between the Euro and the US Dollar
- `GLD`: Gold ETF price (target variable)

## Libraries Used

- pandas
- plotly.express
- scikit-learn
- matplotlib

## Project Structure

- `gld_price_data.csv`: The dataset used for training and testing.
- `gold_price_prediction.ipynb`: The Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- `README.md`: This file.

## Steps

1. **Import Libraries**: Import necessary libraries for data manipulation, visualization, and machine learning.
2. **Load Data**: Load the dataset using pandas.
3. **Data Preprocessing**: 
   - Drop the `Date` column as it is not needed for prediction.
   - Check for missing values.
4. **Exploratory Data Analysis**: 
   - Visualize the correlation matrix using Plotly Express.
   - Display the correlation of each feature with the target variable `GLD`.
5. **Feature Selection**: Separate the features (`X`) and the target variable (`y`).
6. **Data Splitting**: Split the data into training and testing sets.
7. **Model Training**: 
   - Initialize the Random Forest Regressor model.
   - Train the model on the training data.
8. **Model Evaluation**: 
   - Predict gold prices for the testing data.
   - Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
9. **Visualization**: Plot the actual vs predicted gold prices.

## Results

- **Mean Absolute Error (MAE)**: 1.34
- **Mean Squared Error (MSE)**: 7.14
- **R² Score**: 0.986

## How to Run

1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas plotly scikit-learn matplotlib
2. Download the dataset gld_price_data.csv and place it in the project directory.
3. Run the Jupyter notebook gold_price_prediction.ipynb to train the model and visualize the results.

