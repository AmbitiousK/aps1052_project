# No Express Warranty
# MIT License
import numpy as np
# this fixes certain numpy version issues, has to be changed before importing pandas
np.float = float
np.int = int
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, precision_recall_curve


def main():
    df = load_data()  # call the function to load the data and it returns a Pandas DataFrame
    print('Downloaded data ...')
    run_regression(df)  # pass the dataframe to our function


def run_regression(data):
    """This function takes the raw dataframe and performs the splitting and standardization of the quantitative variables
    before fitting to the model and making predictions
    """
    features = data[data.columns[1:-1]]  #  Excludes first (date) and last (target) columns.
    target = data[data.columns[-1]]  # Targets from the last column.

    SPLIT_PCT = 0.20  #  Proportion to hold out for testing. typically its .20 or .30
    # Splitting the data into training and test sets, ensuring stratified sampling.
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=SPLIT_PCT, stratify=target)

    # Creating a scaler and applying it Min()
    sc = StandardScaler()

    # Using pipeline correctly in grid search and cross-validation.
    steps = [('scaler', sc),  #  Pipeline includes scaling.
             ('logistic_regression', LogisticRegression())]
    pipeline = Pipeline(steps)  # Pipeline creation.

    # Parameter grid for SearchCV
    param_grid = {
        'logistic_regression__C': [0.01, 0.1, 1, 10, 100],
        'logistic_regression__class_weight': [None, 'balanced'],
        'logistic_regression__solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag']
    }  # Parameters for logistic regression in the pipeline.

    #  Using features and target directly for grid search is problematic unless pipeline handles scaling.
    grid_search_model = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search_model.fit(X_train, y_train)  #  Should fit using train sets to avoid data leakage.

    # Outputting results of grid search
    print("Best parameters:", grid_search_model.best_params_)  # No change: Outputs best parameters.
    print("Best cross-validated score (best score):", grid_search_model.best_score_)  # No change: Outputs best score.

    best_model = grid_search_model.best_estimator_  #  Retrieves best model from grid search.

    # Predict probabilities
    probabilities = best_model.predict_proba(X_test)[:, 1]

    # Determine the new threshold
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

    # Find an optimal threshold: You can choose based on a trade-off between precision and recall
    optimal_idx = np.argmax(np.sqrt(recall * precision))
    optimal_threshold = thresholds[optimal_idx]

    # Apply threshold
    print(f'optimal threshold for probability: {optimal_threshold}')
    y_pred_adj = (probabilities >= optimal_threshold).astype(int)

    conf_matrix = metrics.confusion_matrix(y_test, y_pred_adj)
    print(conf_matrix)
    class_matrix = metrics.classification_report(y_test, y_pred_adj)
    print(class_matrix)


def load_data():
    """Function will load the Binance BTCUSDT spot timeseries and the Binance blockchain blocks information and
       combine into one pandas dataframe, we will join values using the same date
        """
    url_stats = 'https://www.cryptodatadownload.com/cdd/premium/plus/Blockchain_BTC_historical_blocks.csv'  # endpoint for stats data
    url_ohlc = 'https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv'  # endpoint for Binance spot OHLC file
    stats_df = pd.read_csv(url_stats, skiprows=1)
    ohlc_df = pd.read_csv(url_ohlc, skiprows=1)  # skip header row and read data

    # change the index of each dataframe to be the date column to join on these values
    stats_df.set_index('date', inplace=True)
    ohlc_df.set_index("Date", inplace=True)

    # drop some columns we dont need/want
    stats_df.drop(columns=['symbol', 'first_block', 'last_block'], inplace=True)
    ohlc_df.drop(columns=['Unix', 'Open', 'High', 'Low', 'Symbol', 'Volume USDT'], inplace=True)

    ohlc_df.sort_values('Date', ascending=True, inplace=True)  # sort values ascending first
    ohlc_df['log_price'] = np.log(ohlc_df['Close'])  # get log values of closes
    ohlc_df['pct_change'] = ohlc_df['log_price'].diff()  # get % change
    ohlc_df.sort_values('Date', ascending=False, inplace=True)  # resort back to most recent date first
    ohlc_df.drop(columns=['log_price'], inplace=True)  # drop column we dont need
    ohlc_df.drop(columns=['Close'], inplace=True)

    # join where there is a unix timestamp for spot, reset the index back to int ID afterwards, then rename back to date
    df = stats_df.join(ohlc_df, how='inner').reset_index()
    df.rename(columns={'index': 'date'}, inplace=True)

    # now lets add our target column for the result of whether the following day is POSITIVE or NEGATIVE
    df['target'] = np.where(df['pct_change'] > 0, 1, 0)  # this will add a column called target and set value = 1 if the daily change > 0, otherwise 0
    df['target'] = df['target'].shift(1)  # now we want to shift these values lower by 1 row

    # drop first and last row from our dataset
    # first row has a NAN target classification cause we shifted rows down by one
    # last row is using a half of day of activity and so we will exclude it
    df.drop(index=df.index[0], axis=0, inplace=True)
    df.drop(index=df.index[-1], axis=0, inplace=True)
    return df


if __name__ == "__main__":
    main()