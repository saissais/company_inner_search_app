import sys  
sys.path.append('c:/Users/81805/Desktop/����AI�G���W�j�A_Lesson23_�T���v���A�v��1/�_�E�����[�h�p/company_inner_search_app')  
  
with open('initialize.py', 'r', encoding='utf-8') as f:  
    content = f.read()  
  
# Replace the CSV section  
content = content.replace(  
    '        # CSV�t�@�C���̏ꍇ�̓J�X�^�����[�_�[���g�p\n        if file_extension == \".csv\":\n            from utils import custom_csv_loader\n            docs = custom_csv_loader(path)\n        else:\n            # ���̑��̃t�@�C���͏]���ʂ�\n            loader = loader_config(path)\n            docs = loader.load()',  
    '        # �S�Ẵt�@�C���^�C�v�ŕW�����[�_�[���g�p\n        loader = loader_config(path)\n        docs = loader.load()'  
)  
  
with open('initialize.py', 'w', encoding='utf-8') as f:  
    f.write(content)  
  
print('File updated successfully') 
