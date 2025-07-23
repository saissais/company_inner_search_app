import sys  
sys.path.append('c:/Users/81805/Desktop/生成AIエンジニア_Lesson23_サンプルアプリ1/ダウンロード用/company_inner_search_app')  
  
with open('initialize.py', 'r', encoding='utf-8') as f:  
    content = f.read()  
  
# Replace the CSV section  
content = content.replace(  
    '        # CSVファイルの場合はカスタムローダーを使用\n        if file_extension == \".csv\":\n            from utils import custom_csv_loader\n            docs = custom_csv_loader(path)\n        else:\n            # その他のファイルは従来通り\n            loader = loader_config(path)\n            docs = loader.load()',  
    '        # 全てのファイルタイプで標準ローダーを使用\n        loader = loader_config(path)\n        docs = loader.load()'  
)  
  
with open('initialize.py', 'w', encoding='utf-8') as f:  
    f.write(content)  
  
print('File updated successfully') 
