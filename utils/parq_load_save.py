# Copy to top of each Module.py or Utility.py
project_name = 'NLP_Disaster_Tweets'
dependencies_path = f'C:/Users/delst/OneDrive/Desktop/Code/Workspace/{project_name}/Libraries/dependencies.py'
with open(dependencies_path, 'r') as file:
    code = file.read()
    exec(code)


def save_to_parquet(df, file_name, save_dir):
    """
    Save a DataFrame to a .parquet file.
    """
    file_path = os.path.join(save_dir, file_name)
    df.to_parquet(file_path)
    print(f"DataFrame saved to {file_path}")


def load_from_parquet(file_name, load_dir):
    """
    Load a .parquet file into a DataFrame.
    """
    file_path = os.path.join(load_dir, file_name)
    df = pd.read_parquet(file_path)
    print(f"{file_path} loaded into DataFrame")
    return df
