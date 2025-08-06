import pandas as pd

def csv_to_fasta(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)
    
    # 打开输出文件进行写入
    with open(output_file, 'w') as fasta_file:
        for index, row in df.iterrows():
            # 获取序列编号和tm值
            sequence = row['sequence']
            tm = row['tm']
            # 设置 FASTA 标题格式
            title = f">{index+1}_{tm:.2f}"
            # 写入标题和序列
            fasta_file.write(f"{title}\n{sequence}\n")
    
    print(f"转换完成，FASTA 文件已保存为: {output_file}")

if __name__ == "__main__":
    input_file = 'WBtrain_no_dup.csv'  # 替换为去重后的 CSV 文件路径
    output_file = 'WBtrain_no_dup.fasta'         # 替换为输出的 FASTA 文件路径
    csv_to_fasta(input_file, output_file)
