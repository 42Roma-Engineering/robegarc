import os
import config
import pathlib
import threading
import pandas as pd
import numpy as np


class DataCleaner:

    def __init__(self, raw_data_dir, clean_data_dir):
        self.raw_data_dir = raw_data_dir
        self.clean_data_dir = clean_data_dir
        self.current_folder = ""
        self.dataframes = []

    def _find_data_files(self):
        """
        Iterates through folders and files in the raw data folder,
        returning list of file paths.
        """

        file_paths = []
        for root, directories, files in os.walk(self.raw_data_dir):
            if "test" in root.split(os.sep):
                continue
            for folder in directories:
                self.current_folder = folder
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    def _clean_data(self, df) -> pd.DataFrame:
        """
        Performs basic cleaning operations on a pandas DataFrame.
        Can definitely be extended to incorporate several cleaning strategies
        and feature engineering based on your needs.

        Args:
            df: DataFrame to be cleaned.

        Returns:
            The cleaned DataFrame.
        """

        df = df.dropna()
        df = df.iloc[:, :-3]

        # df = df.drop_duplicates()
        # other cleaning operations as needed

        return df

    def _save_cleaned_data(self, df, i):
        """
        Saves the cleaned DataFrame to a CSV file in clean data folder.
        """

        pathlib.Path(self.clean_data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(os.path.join(self.clean_data_dir, f"{i}.csv"), index=False)

    def prepare_data(self):
        """
        Prepares data by finding, cleaning, and saving files.
        This method can be customized to include additional cleaning steps
        and feature engineering as needed.

        Returns:
            List of DataFrames.
        """

        file_paths = self._find_data_files()

        for i, file_path in enumerate(file_paths):
            print(f"Processing file {i + 1}: {file_path}")
            try:
                df = pd.read_csv(file_path, sep=';')
                cleaned_df = self._clean_data(df.copy())
                self.dataframes.append(cleaned_df)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        for i, df in enumerate(self.dataframes):
            self._save_cleaned_data(df, i)

        print("Data preparation complete!")
        return self.dataframes


class DataMerger:

    def __init__(self, data_dir, train_data_dir, dataframes):
        self.data_dir = data_dir
        self.train_data_dir = train_data_dir
        self.dataframes = dataframes

    def _read_csv_file(self, file_path: str):
        """
        Reads a CSV file within a thread, handling exceptions.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame, or None if an error occurs.
        """

        try:
            df = pd.read_csv(file_path)
            df = df.dropna()
            df = df.sample(frac=0.05, random_state=42)
            return df
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def _merge_dataframes(self):
        """
        Merges all dataframes into a single DataFrame.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """

        if not self.dataframes:
            raise ValueError("No CSV files found in the data directory!")

        dfs = self.dataframes[:5]
        data = pd.concat(dfs, ignore_index=True, sort=False)
        new_data = pd.DataFrame(data)

        for column in new_data.columns:
            try:
                new_data[column] = new_data[column].astype(float)
            except ValueError as e:
                print(f"Error converting column {column}: {e}")
        return new_data

    def merge_files_with_threads(self) -> pd.DataFrame:
        """
        Merges CSV files in the data train folder using threads for parallelism.

        Raises:
            ValueError: If no CSV files are found.
        """

        threads = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(self.data_dir, file)
                thread = threading.Thread(target=self._read_csv_file, args=(file_path,))
                thread.start()
                threads.append(thread)

        for thread in threads:
            thread.join()

        all_data = self._merge_dataframes()

        pathlib.Path(config.TRAIN_DATA_DIR).mkdir(parents=True, exist_ok=True)
        all_data.to_csv(os.path.join(config.TRAIN_DATA_DIR, "train_data.csv"), index=False)

        print("Returning the merged files!")
        return all_data


def run():
    data_cleaner = DataCleaner(config.RAW_DATA_DIR, config.CLEAN_DATA_DIR)
    data = data_cleaner.prepare_data()
    data_merger = DataMerger(config.CLEAN_DATA_DIR, config.TRAIN_DATA_DIR, data)
    merged_data = data_merger.merge_files_with_threads()


if __name__ == "__main__":
    run()
