import pandas as pd
import sys

def remove_duplicates(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)
    
    # 去重 'sequence' 列，保留第一次出现的值
    df_unique = df.drop_duplicates(subset=['sequence'], keep='first')
    
    # 保存去重后的数据到新的 CSV 文件
    df_unique.to_csv(output_file, index=False)
    
    print(f"去重完成，新的文件已保存为: {output_file}")

if __name__ == "__main__":
    # 获取命令行参数
    if len(sys.argv) != 3:
        print("用法: python remove_duplicates.py <输入文件> <输出文件>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        remove_duplicates(input_file, output_file)
