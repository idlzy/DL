import os
import sys
import shutil

cwd_path = os.getcwd()
cwd_dir = os.path.split(cwd_path)[-1]
if cwd_dir != "DL":
    print("please work at directory 'DL' before start training")
    sys.exit()

temp_dir = "wastebin"

shutil.rmtree(temp_dir)
os.makedirs(temp_dir)
print("finished clearing wastebin")