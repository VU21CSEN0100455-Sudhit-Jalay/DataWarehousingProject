import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    """Load the dataset from the given file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None

def preprocess_data(data):
    """Preprocess the dataset by cleaning and transforming it."""
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    # Remove transactions with amounts exceeding a certain threshold
    data = data[data['Transaction_Amount'] <= 1000]
    
    # Encoding categorical variables
    encoder = OneHotEncoder()
    encoded_data = pd.DataFrame(encoder.fit_transform(data[['Product_Category', 'Customer_Segment']]).toarray(),
                                columns=encoder.get_feature_names_out(['Product_Category', 'Customer_Segment']))
    preprocessed_data = pd.concat([data.drop(['Product_Category', 'Customer_Segment'], axis=1), encoded_data], axis=1)
    
    # Assign buying frequency based on transaction amount
    preprocessed_data['Buying_Frequency'] = pd.cut(preprocessed_data['Transaction_Amount'],
                                                   bins=[0, 5, 10, float('inf')],
                                                   labels=['Low', 'Medium', 'High'])
    
    return preprocessed_data

def main():
    # Load the dataset
    file_path = "supermarket_dataset.csv"
    data = load_data(file_path)
    if data is None:
        return
    
    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    
    # Save the preprocessed data to a new CSV file
    preprocessed_data.to_csv('preprocessed_supermarket_dataset.csv', index=False)
    print("Preprocessed data saved to 'preprocessed_supermarket_dataset.csv'")

if __name__ == "__main__":
    main()
