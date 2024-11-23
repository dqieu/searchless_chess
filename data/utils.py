import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src import bagz, constants


class BagToCsvConverter:
    """Converts chess bag files to CSV format."""

    def __init__(self, bag_file_path: str):
        """Initialize the converter with a bag file path.

        Args:
            bag_file_path: Path to the .bag file
        """
        self.bag_reader = bagz.BagReader(bag_file_path)

    def convert_to_dataframe(self) -> pd.DataFrame:
        """Converts the bag file content to a pandas DataFrame.

        Returns:
            DataFrame with the chess data
        """
        records: List[Dict[str, Any]] = []

        for bytes_data in self.bag_reader:
            # Detect the data type based on the file name
            if 'action_value' in self.bag_reader._filename:
                fen, move, win_prob = constants.CODERS['action_value'].decode(
                    bytes_data)
                records.append({
                    'fen': fen,
                    'move': move,
                    'win_probability': win_prob
                })
            elif 'state_value' in self.bag_reader._filename:
                fen, win_prob = constants.CODERS['state_value'].decode(
                    bytes_data)
                records.append({
                    'fen': fen,
                    'win_probability': win_prob
                })
            elif 'behavioral_cloning' in self.bag_reader._filename:
                fen, move = constants.CODERS['behavioral_cloning'].decode(
                    bytes_data)
                records.append({
                    'fen': fen,
                    'move': move
                })

        return pd.DataFrame(records)

    def save_to_csv(self, output_path: str) -> None:
        """Saves the bag file content to a CSV file.

        Args:
            output_path: Path where to save the CSV file
        """
        df = self.convert_to_dataframe()
        df.to_csv(output_path, index=False)


# Example usage:
converter = BagToCsvConverter(
    "/Users/datkieu/PycharmProjects/searchless_chess/data/test/state_value_data.bag")

# Convert to DataFrame
df = converter.convert_to_dataframe()
print(df.head())

# Save to CSV
converter.save_to_csv("output.csv")
