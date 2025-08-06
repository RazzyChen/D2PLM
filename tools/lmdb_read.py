import lmdb
import pickle
import json
import argparse

def count_lmdb_entries(lmdb_path):
    """
    统计 LMDB 文件中的数据条数。
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
    打印 LMDB 文件中的前 num_samples 条数据的键和值。
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= num_samples:
                break
            print(f"Key: {key.decode('ascii')}")
            try:
                # 首先尝试 JSON 解析
                json_value = json.loads(value.decode('utf-8'))
                print(f"Value (JSON): {json_value}")
            except json.JSONDecodeError:
                try:
                    # 如果 JSON 解析失败，尝试 pickle 解析
                    pickle_value = pickle.loads(value)
                    print(f"Value (Pickle): {pickle_value}")
                except Exception as e:
                    print(f"Value could not be deserialized (neither JSON nor Pickle): {e}")
                    print(f"Raw value: {value[:100]}...")  # 打印前100个字节的原始值
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取 LMDB 文件并统计数据条数，打印前几条数据的键和值。")
    parser.add_argument('-f', dest='lmdb_path', type=str, required=True, help='LMDB 文件的路径')
    
    args = parser.parse_args()
    
    num_entries = count_lmdb_entries(args.lmdb_path)
    print(f"LMDB 文件中包含 {num_entries} 条数据")

    print("\n打印前 5 条数据的键和值：")
    print_lmdb_keys_and_values(args.lmdb_path, num_samples=10)