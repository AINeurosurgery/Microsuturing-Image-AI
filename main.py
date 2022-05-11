import torch
from torch._C import BenchmarkExecutionStats
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from loader import customdata
from final_model import arch1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as RAS
from sklearn.metrics import mean_squared_error
import numpy as np
import os


def train(model, device, train_loader, optimizer,criterion, epoch):
    model.train()
    p =[]
    t =[]

    for batch_idx, (data1,s6,s5,s4,s3,s2,s1) in enumerate(train_loader):
        data1,s6,s5,s4,s3,s2,s1 = data1.to(device),s6.to(device),s5.to(device),s4.to(device),s3.to(device),s2.to(device),s1.to(device)

        optimizer.zero_grad()
        op1, op2,op3,op4,op5,op6 = model(data1)

        loss0 = bce_loss(op1[:,0], s1)
        loss1 = bce_loss(op2[:,0],s2)
        loss2 = bce_loss(op3[:,0], s3)
        loss3 = bce_loss(op4[:,0], s4)
        loss4 = bce_loss(op5[:,0], s5)
        loss5 = bce_loss(op6[:,0], s6)


        loss1.backward(retain_graph=True)

        loss2.backward(retain_graph=True)

        loss3.backward(retain_graph=True)

        loss4.backward(retain_graph=True)

        loss5.backward(retain_graph=True)

        loss0.backward(retain_graph=True)
        optimizer.step()

        op = op1[:,0]

        t.extend(s1.detach().cpu().numpy())
        p.extend(op.detach().cpu().numpy())

        if batch_idx % 1 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss0.item()))

    MSE = mean_squared_error(np.array(t),np.array(p),squared=False)
    print(f"MSE:{MSE:.4f}")








def test(model, device, test_loader,criterion):
    model.eval()
    test_loss = 0

    p =[]
    t =[]
    with torch.no_grad():
        for (data1,s6,s5,s4,s3,s2,s1) in test_loader:

            data1,s6,s5,s4,s3,s2,s1 = data1.to(device),s6.to(device),s5.to(device),s4.to(device),s3.to(device),s2.to(device),s1.to(device)

            op1, op2,op3,op4,op5,op6 = model(data1)
            test_loss += criterion(op1[:,0], s1).item()  # sum up batch loss
            op = op1[:,0]
            t.extend(s1.detach().cpu().numpy())
            p.extend(op.detach().cpu().numpy())
        #rocauc = RAS(np.array(t),np.array(p))
        print(len(p))
        MSE = mean_squared_error(np.array(t),np.array(p),squared=False)
        # A = accuracy/len(test_loader.dataset)
        A = MSE
        # A = rocauc/len(test_loader.dataset)
        # A = rocauc


    if not os.path.isdir('models'):
        os.mkdir('models')
    print('\nTest set: Average loss: {:.4f}, MSE score: ({:.4f})\n'.format(
        test_loss/len(test_loader.dataset),10.*A))
    global maxv
    if A<maxv:
        temp1  = A
        temp2 = temp1*1000
        temp3 = int(temp2)
        torch.save(model.state_dict(),f"models/test_model{temp3}.pth.tar")
        print("model saved")
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
    print(model)


    criterion_train = nn.MSELoss().to(device)
    bce_loss = nn.MSELoss().to(device)

    criterion_test = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)


    num_epochs = 100
    global maxv
    maxv = 100

    for epochs in range(num_epochs):
        train(model,device,train_loader,optimizer,criterion_train,epochs)
        test(model,device,test_loader,criterion_test)



