
import os
from bagpy import bagreader
import pandas as pd

def convert_bag_to_csv(bag_file_path, output_dir):
    try:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read the bag file
        b = bagreader(bag_file_path)

        # Get all topics in the bag file
        print("Topics available in the bag file:")
        print(b.topic_table)

        # Loop through each topic and save it as CSV
        for topic in b.topics:
            print(f"Processing topic: {topic}")
            csv_file = b.message_by_topic(topic)
            if csv_file:
                print(f"CSV for {topic} saved at: {csv_file}")
                # Optionally: Move CSV files to the specified output directory
                topic_file_name = os.path.basename(csv_file)
                os.rename(csv_file, os.path.join(output_dir, topic_file_name))

        print(f"All topics have been converted to CSV in the directory: {output_dir}")
    except Exception as e:
        print(f"Error occurred: {e}")

# Input bag file and output directory
bag_file = r"C:\Users\tiend\PycharmProjects\searchless_chess\data\action_value_data.bag"  # Replace with your .bag file path
output_directory = r"C:\Users\tiend\PycharmProjects\searchless_chess\data"  # Replace with your output folder

# Call the function
convert_bag_to_csv(bag_file, output_directory)
