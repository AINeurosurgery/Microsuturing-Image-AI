import torch
from torch._C import BenchmarkExecutionStats
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from loader_test1 import customdata
from final_model import arch1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as RAS
from sklearn.metrics import mean_squared_error
import numpy as np
import os


def test(model, device, test_loader,criterion):
    model.eval()
    test_loss = 0

    p =[]
    t =[]

    with torch.no_grad():
        for (name, data1,s6,s5,s4,s3,s2,s1) in test_loader:

            data1,s6,s5,s4,s3,s2,s1 = data1.to(device),s6.to(device),s5.to(device),s4.to(device),s3.to(device),s2.to(device),s1.to(device)

            op1, op2,op3,op4,op5,op6 = model(data1)
            test_loss += criterion(op1[:,0], s1).item()  # sum up batch loss

            op = op1[:,0]

            t.extend(s1.detach().cpu().numpy())

            p.extend(op.detach().cpu().numpy())

        print(len(p))
        print(len(name))
        # for i in t:
        #     print(i,type(i))
##################Image Name and overall score ###########
        with open('test_img_names1.txt','w') as f:
            for i in name:
                f.write(i[0])
                f.write('\n')
        with open('test_gt1.txt','w') as f:
            for i in t:
                f.write(str(i*10))
                f.write('\n')
        with open('test_pred1.txt','w') as f:
            for i in p:
                f.write(str(i*10))
                f.write('\n')
######################################################### 
        
            

        MSE = mean_squared_error(np.array(t),np.array(p),squared=False)
        A = MSE



    if not os.path.isdir('models'):
        os.mkdir('models')
    print('\nTest set: Average loss: {:.4f}, MSE score: ({:.4f})\n'.format(
        test_loss/len(test_loader.dataset),10.*A))
    global maxv
    if A<maxv:
        maxv = A



if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.manual_seed(42)

    mt = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((388,175)),
        # transforms.Scale()/
    ])

    trainset = customdata(root="all_param_data/train/",train=True,transforms=mt)
    testset = customdata(root="all_param_data/test/",train=False,transforms=mt)

    train_loader = DataLoader(trainset,batch_size=32,shuffle=True,drop_last=True,num_workers=6)
    test_loader = DataLoader(testset,batch_size=1,shuffle=False,drop_last=False,num_workers=6)

    model = arch1().to(device)
    model.load_state_dict(torch.load('models/test_model83.pth.tar'))
    print(model)


    criterion_train = nn.MSELoss().to(device)


    criterion_test = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)


    num_epochs = 100
    global maxv
    maxv = 100


    test(model,device,test_loader,criterion_test)



