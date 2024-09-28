import os


def read_and_print_txt_files(folder_path, output_path):
    files = os.listdir(folder_path)
    txt_files = [file for file in files if file.endswith('.txt')]

    txt_files.sort(key=lambda file: os.path.getmtime(os.path.join(folder_path, file)))

    _cnt = 0

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for txt_file in txt_files:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if '<NO ALARM>' not in line:
                        print(line.strip())
                        output_file.write(line.strip() + '\n')
                        _cnt += 1

    print(f'_cnt --> {_cnt} in {len(txt_files)}')
    with open(output_path, 'a', encoding='utf-8') as output_file:
        output_file.write(f'_cnt --> {_cnt} in {len(txt_files)}\n')


folder_path = '/home/manu/tmp/fire_test_results'
output_path = '/home/manu/tmp/fire_test_results.txt'
read_and_print_txt_files(folder_path, output_path)
