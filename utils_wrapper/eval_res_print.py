import os


def read_and_print_txt_files(folder_path):
    files = os.listdir(folder_path)
    txt_files = [file for file in files if file.endswith('.txt')]
    _cnt = 0
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if '<NO ALARM>' not in line:
                    print(line.strip())
                    _cnt += 1
    print(f'_cnt --> {_cnt}')


folder_path = '/home/manu/tmp/fire_test_results'
read_and_print_txt_files(folder_path)
