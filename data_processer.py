import os
import gzip
import shutil
import glob

def decompress_and_remove_gz_files(gz_file):

    nii_file = gz_file.replace(".gz", "")  # 去掉 .gz 扩展名，得到目标文件名
    with gzip.open(gz_file, 'rb') as f_in:
        with open(nii_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        
    os.remove(gz_file)  # 删除原始 .nii.gz 文件
    print(f"File '{gz_file}' has been decompressed to '{nii_file}' and original file removed.")

    
def filename_processer(path):
    list = os.listdir(path)
    for file in list:
        filepath = os.path.join(path, file)
        if os.path.isdir(filepath):
            filename_processer(filepath)
        elif filepath.split('.')[-1] == "gz":
            decompress_and_remove_gz_files(filepath)

if __name__ == '__main__':

    filename_processer("dataset/")

        
            
            