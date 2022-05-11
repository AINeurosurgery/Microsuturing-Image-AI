import os
import numpy as np
import cv2
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms



class customdata(Dataset):
    def __init__(self,root,train=True,transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms

        image1 = "images/"
        self.image1=image1
        self.image1_folder = sorted([os.path.join(self.root+image1,x) for x in os.listdir(self.root+image1)])
        #self.image2_folder = sorted([os.path.join(self.root+image2,x) for x in os.listdir(self.root+image2)])
    

    def __len__(self):
        return len(self.image1_folder)
    def read_files(self,filename):
        P = sorted(os.listdir(filename))
        p1 = [i.split('_') for i in P]
        s6 = []
        s5 = []
        s4 = []
        s3 = []
        s2 = []
        s1 = []
        
        lines3 = []
        for i in range(len(P)):
            num = p1[i][-1]
            num1 = num[0:-4]
            num2 = p1[i][-2]
            num3 = p1[i][-3]
            num4 = p1[i][-4]
            num5 = p1[i][-5]
            num6 = p1[i][-6]
            s6.append(float(num1)/10)
            s5.append(float(num2)/10)
            s4.append(float(num3)/10)
            s3.append(float(num4)/10)
            s2.append(float(num5)/10)
            s1.append(float(num6)/10)
            lines3.append(P[i])
        return lines3,np.array(s6).astype(np.float32),np.array(s5).astype(np.float32),np.array(s4).astype(np.float32),np.array(s3).astype(np.float32),np.array(s2).astype(np.float32),np.array(s1).astype(np.float32)


    def __getitem__(self, index):

        images_name,s6,s5,s4,s3,s2,s1 = self.read_files(self.root+self.image1)
        img1 =  cv2.imread(self.root+self.image1+images_name[index])
        img1 = cv2.resize(img1,(img1.shape[0]*2,img1.shape[1]*2))
        img1 = np.array(img1)
        s_6 = s6[index]
        s_5 = s5[index]
        s_4 = s4[index]
        s_3 = s3[index]
        s_2 = s2[index]
        s_1 = s1[index]
        #print(index, images_name[index], label)

        #froc = frocs[index]

        if self.transforms is not None:
            img1 = self.transforms(img1)


        # # print(self.image1_folder[index],label)
        


        return (images_name,img1,s_6,s_5,s_4,s_3,s_2,s_1)


# mt = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Resize((388,175))
#     ]
# )

# dataset = customdata(root="DATA/test/",train=True,transforms=mt)

# loader = DataLoader(dataset,batch_size=8,shuffle=False,drop_last=True)

# for i,j in loader:
#     print(i.shape,j.shape)
    #exit()
