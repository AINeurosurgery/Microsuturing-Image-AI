
import os
import pandas as pd
import shutil

################# Lines to be changed ###############################
path = 'Train_data/images/' # This is the path where the images are present
path_excel = 'Train_data/train_scores.xlsx' # This is the path where the scores for the images are present
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
