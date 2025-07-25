import os

import dask.dataframe as dd
import pandas as pd


def remove_duplicate_positions(input_file, output_file):
    print(f"Processing {input_file} with Dask")

    # Read the jsonl file with optimizations
    print("Reading input file...")
    ddf = dd.read_json(
        input_file,
        lines=True,
        blocksize="128MB",  # Larger blocks for better parallelization
        compression=None,  # Disable auto compression detection
    )
    print("Done reading input file")

    # Convert the array format to a proper dataframe structure
    # Each line is [position_string, value], so we'll name the columns
    print("Structuring data...")
    ddf.columns = ["position", "value"]

    # Drop duplicates based on the 'position' column (which contains the chess position)
    # This is a lazy operation.
    print("Dropping duplicates...")
    unique_ddf = ddf.drop_duplicates(subset=["position"])
    print("Done dropping duplicates")

    # Save to JSONL file in chunks to avoid memory issues
    print(f"Saving unique positions to {output_file}")
    import json

    with open(output_file, "w") as f:
        # Process in smaller chunks to avoid memory overflow
        for partition in unique_ddf.to_delayed():
            print(f"Processing partition: {partition}")
            chunk_df = partition.compute()
            for _, row in chunk_df.iterrows():
                print(f"Writing row: {row}")
                f.write(json.dumps([row["position"], row["value"]]) + "\n")

    print(f"Unique positions saved to {output_file}")


if __name__ == "__main__":
    input_path = "data/chess_evals.jsonl"
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
    else:
        output_path = "data/chess_evals_unique.jsonl"
        remove_duplicate_positions(input_path, output_path)
