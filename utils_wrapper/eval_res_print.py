import os


def read_and_print_txt_files(folder_path, output_path):
    files = os.listdir(folder_path)
    txt_files = [file for file in files if file.endswith('.txt')]
    _cnt = 0

    # 打开输出文件以写入
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for txt_file in txt_files:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if '<NO ALARM>' not in line:
                        # 打印行内容
                        print(line.strip())
                        # 写入输出文件
                        output_file.write(line.strip() + '\n')
                        _cnt += 1

    # 打印并写入计数值
    print(f'_cnt --> {_cnt}')
    with open(output_path, 'a', encoding='utf-8') as output_file:
        output_file.write(f'_cnt --> {_cnt}\n')


folder_path = '/home/manu/tmp/fire_test_results'
output_path = '/home/manu/tmp/fire_test_results.txt'
read_and_print_txt_files(folder_path, output_path)
