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

global adjusted
adjusted = np.zeros(10)
global adjusted_freq
adjusted_freq = np.zeros(10)

def compute_adjustment(train_loader, device, tro = 1.0):
    """compute the base probabilities"""

    label_freq = {}
    for i, (inputs,target,s5,s4,s3,s2,s1) in enumerate(train_loader):
        target = target.to(device)
        for j in target:
            key = int(j*10)
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adj_freq = np.zeros(len(adjustments))
    for i in range(len(adjustments)):
        adj_freq[i] = adjustments[i]*(i+1)*0.1
    mean_val = sum(adj_freq)/sum(adjustments)
    var = (0.1-mean_val)**2*adjustments[0]
    for i in range(1,len(adjustments)):
        var+= ((i+1)*0.1-mean_val)**2*adjustments[i]
    var = var/sum(adjustments)    
    return mean_val,var


def train(model, device, train_loader, optimizer,criterion, epoch):
    global adjusted
    global adjusted_freq
    model.train()
    p =[]
    t =[]
    mean_adj,var_adj = compute_adjustment(train_loader, device)

    for batch_idx, (data1,s6,s5,s4,s3,s2,s1) in enumerate(train_loader):
        data1,s6,s5,s4,s3,s2,s1 = data1.to(device),s6.to(device),s5.to(device),s4.to(device),s3.to(device),s2.to(device),s1.to(device)

        optimizer.zero_grad()
        op11,op21,op31,op41,op51,op61 = model(data1)
        op12,op22,op32,op42,op52,op62 = model(data1)
        op13,op23,op33,op43,op53,op63 = model(data1)
        op14,op24,op34,op44,op54,op64 = model(data1)
        op15,op25,op35,op45,op55,op65 = model(data1)

        op1 = (op11+op12+op13+op14+op15)/5
        op2 = (op21+op22+op23+op24+op25)/5
        op3 = (op31+op32+op33+op34+op35)/5
        op4 = (op41+op42+op43+op44+op45)/5
        op5 = (op51+op52+op53+op54+op55)/5
        op6 = (op61+op62+op63+op64+op65)/5

        var6 = (op61-op6)**2
        var6+= (op62-op6)**2
        var6+= (op63-op6)**2
        var6+= (op64-op6)**2
        var6+= (op65-op6)**2
        var6/=5

        op6_final = var6*mean_adj+var_adj*op6
        op6_final = op6_final/(var6+var_adj)


        loss0 = bce_loss(op1[:,0], s1)
        loss1 = bce_loss(op2[:,0],s2)
        loss2 = bce_loss(op3[:,0], s3)
        loss3 = bce_loss(op4[:,0], s4)
        loss4 = bce_loss(op5[:,0], s5)
        loss5 = bce_loss(op6_final[:,0], s6)


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
    if not os.path.isdir("models"):
        os.mkdir("models")
    torch.manual_seed(42)

    mt = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((388,175)),
        # transforms.Scale()/
    ])

    trainset = customdata(root="data/Train_cohort/",train=True,transforms=mt)
    testset = customdata(root="data/Validation_cohort/",train=False,transforms=mt)

    train_loader = DataLoader(trainset,batch_size=32,shuffle=True,drop_last=True,num_workers=6)
    test_loader = DataLoader(testset,batch_size=1,shuffle=False,drop_last=False,num_workers=6)

    model = arch1().to(device)
    print(model)


    criterion_train = nn.MSELoss().to(device)
    bce_loss = nn.MSELoss().to(device)

    criterion_test = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)


    num_epochs = 50
    global maxv
    maxv = 100

    for epochs in range(num_epochs):
        train(model,device,train_loader,optimizer,criterion_train,epochs)
        test(model,device,test_loader,criterion_test)



