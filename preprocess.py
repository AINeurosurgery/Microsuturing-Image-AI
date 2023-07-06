
import os
import pandas as pd
import shutil

################# Lines to be changed ###############################
path = 'data/Validation_cohort/Validation/' # This is the path where the images are present
path_excel = 'data/Validation_cohort/Validation_annotations.xlsx' # This is the path where the scores for the images are present
#####################################################################

excel = pd.read_excel(path_excel, index_col=0)
excel_list = excel.values.tolist()
names = [i[0] for i in excel_list]

for i in range(len(names)):
    nm = names[i]
    new_nm = nm[:-4]
    for j in range(1,7):
        new_nm+='_'
        new_nm+=str(excel_list[i][j])
    new_nm+='.png'
    os.rename(path+nm,path+new_nm)

path_split = path.split('/')
new_path = ""
for i in range(len(path_split)-2):
    new_path+=path_split[i]
    new_path+='/'

os.rename(path,new_path+"images/")
