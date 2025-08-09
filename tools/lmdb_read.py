import lmdb
import pickle
import json
import argparse

def count_lmdb_entries(lmdb_path):
    """
    Count the number of entries in an LMDB file.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    with env.begin() as txn:
        len_bytes = txn.get(b'__len__')
        if len_bytes is not None:
            return pickle.loads(len_bytes)
        else:
            return txn.stat()['entries']

def print_lmdb_keys_and_values(lmdb_path, num_samples=5):
    """
    Print the keys and values of the first num_samples entries in an LMDB file.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= num_samples:
                break
            print(f"Key: {key.decode('ascii')}")
            try:
                # First try to parse as JSON
                json_value = json.loads(value.decode('utf-8'))
                print(f"Value (JSON): {json_value}")
            except json.JSONDecodeError:
                try:
                    # If JSON parsing fails, try to parse as pickle
                    pickle_value = pickle.loads(value)
                    print(f"Value (Pickle): {pickle_value}")
                except Exception as e:
                    print(f"Value could not be deserialized (neither JSON nor Pickle): {e}")
                    print(f"Raw value: {value[:100]}...")  # Print the first 100 bytes of the raw value
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read an LMDB file, count the number of entries, and print the first few key-value pairs.")
    parser.add_argument('-f', dest='lmdb_path', type=str, required=True, help='Path to the LMDB file')
    
    args = parser.parse_args()
    
    num_entries = count_lmdb_entries(args.lmdb_path)
    print(f"The LMDB file contains {num_entries} entries")

    print("\nPrinting the keys and values of the first 5 entries:")
    print_lmdb_keys_and_values(args.lmdb_path, num_samples=10)
