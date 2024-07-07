import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(transaction_file_path, label_file_path):
    """
    Load data from CSV files.
    """
    df_transaction = pd.read_csv(transaction_file_path)
    df_label = pd.read_csv(label_file_path)
    df_transaction['is_fraud'] = df_transaction['eventId'].isin(df_label['eventId']).astype(int)  #add the class label
    return df_transaction

def preprocess_transaction_data(df_transaction):
    """
    Preprocess the transaction data.
    """
    # Convert transactionTime to datetime
    df_transaction['transactionTime'] = pd.to_datetime(df_transaction['transactionTime'])
    
    # Extract additional features from transactionTime
    df_transaction['transactionHour'] = df_transaction['transactionTime'].dt.hour
    df_transaction['transactionDay'] = df_transaction['transactionTime'].dt.day
    df_transaction['transactionMonth'] = df_transaction['transactionTime'].dt.month
    df_transaction['transactionDayOfWeek'] = df_transaction['transactionTime'].dt.dayofweek
    
    # Drop the original transactionTime and eventId columns
    df_transaction = df_transaction.drop(columns=['transactionTime', 'eventId'])
    
    return df_transaction

def split_data(df_transaction, test_size=0.2, validation_size=0.25):
    """
    Splitting data into train, test and validation sets
    """
    X = df_transaction.drop(columns=['is_fraud'])
    y = df_transaction['is_fraud']
    
    # Split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Split training+validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size, random_state=42, stratify=y_train_val)
    
    return X_train, X_val, X_test, y_train, y_val, y_test