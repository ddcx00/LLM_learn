import pandas as pd


file_path = r"C:\Users\Admin\Desktop\Energy_Consumption_Predict_BEE\给出数据\ALL_in_one.csv"

# 先查看前几行了解结构
print("=== 文件结构预览 ===")
preview = pd.read_csv(file_path, nrows=5)
print(preview)
print(f"\n列名: {list(preview.columns)}")
print(f"数据类型:\n{preview.dtypes}")

preview.to_csv("sample5.csv", index=False, encoding='utf-8')
print("\n样本已保存")