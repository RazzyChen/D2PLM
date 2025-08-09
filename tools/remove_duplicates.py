import pandas as pd
import sys

def remove_duplicates(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Remove duplicates from the 'sequence' column, keeping the first occurrence
    df_unique = df.drop_duplicates(subset=['sequence'], keep='first')
    
    # Save the deduplicated data to a new CSV file
    df_unique.to_csv(output_file, index=False)
    
    print(f"Deduplication complete, the new file has been saved as: {output_file}")

if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) != 3:
        print("Usage: python remove_duplicates.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        remove_duplicates(input_file, output_file)
