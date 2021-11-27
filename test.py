import torch
import os
import torch.optim as optim
from  torch import nn
from Mydataset import Mydataset
from torch.utils.data import DataLoader
# from resnet_cbam import ResNet, resnet50_cbam,resnet18_cbam,resnet101_cbam,resnet34_cbam,resnet152_cbam,
from model import FC_model
import visdom ,time


root = os.getcwd()+'\\all_chong\\'
# root = os.getcwd()+'\\all_sample\\'
viz = visdom.Visdom()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)

test_data = Mydataset(txt_path=root + '2.txt', resize=224, mode='test')


test_loader = DataLoader(test_data, batch_size=20, shuffle=True)


def evalute(model, loader):

    model.eval()

    correct0 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    correct6 = 0
    correct7 = 0
    correct8 = 0
    total = len(loader.dataset)
    # print(total)
    # for data in loader:
    #     print(type(data))
    #     break

    for x, y1, y2, y3, y4, y5, y6, y7, y8 in loader:
        x = x.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        y3 = y3.to(device)
        y4 = y4.to(device)
        y5 = y5.to(device)
        y6 = y6.to(device)
        y7 = y7.to(device)
        y8 = y8.to(device)

        with torch.no_grad():
            logits = model(x)
            # print(logits.shape)
            pred0 = logits[0].argmax(dim=1)
            pred1 = logits[1].argmax(dim=1)
            pred2 = logits[2].argmax(dim=1)
            pred3 = logits[3].argmax(dim=1)
            pred4 = logits[4].argmax(dim=1)
            pred5 = logits[5].argmax(dim=1)
            pred6 = logits[6].argmax(dim=1)
            pred7 = logits[7].argmax(dim=1)
            # pred8 = logits[8].argmax(dim=1)

        correct0 += torch.eq(pred0, y1).sum().float().item()
        correct1 += torch.eq(pred1, y2).sum().float().item()
        correct2 += torch.eq(pred2, y3).sum().float().item()
        correct3 += torch.eq(pred3, y4).sum().float().item()
        correct4 += torch.eq(pred4, y5).sum().float().item()
        correct5 += torch.eq(pred5, y6).sum().float().item()
        correct6 += torch.eq(pred6, y7).sum().float().item()
        correct7 += torch.eq(pred7, y8).sum().float().item()
        # correct8 += torch.eq(pred8, y).sum().float().item()

    return correct0 / total, correct1 / total, correct2 / total, correct3 / total, correct4 / total, correct5 / total, correct6 / total, correct7 / total




def main():
    model = FC_model().to(device)
    # opt1 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # # criteon = nn.CrossEntropyLoss()
    # criteon = nn.CrossEntropyLoss(reduction='mean')
    # opt1.zero_grad()
    # # Loss.backward()
    # # optimizer.step()
    # opt1.step()



    # model.load_state_dict(torch.load('model.mdl'))
    # print('loaded from GKD!')
    # test_acc = evalute(model, test_loader)
    # print('test acc:', test_acc)
    # model.load_state_dict(torch.load('last.mdl'))
    # print('loaded from last!')
    # test_acc1 = evalute(model, test_loader)
    # print('test acc last:', test_acc1)
    #
    # time.sleep(2)

    # model.load_state_dict(torch.load('best1.mdl'))
    # print('loaded from 1!')
    # test_acc1 = evalute(model, test_loader)
    # print('test acc1:', test_acc1)
    #
    # time.sleep(2)

    model.load_state_dict(torch.load('best2.mdl'))
    print('loaded from 2!')
    test_acc2 = evalute(model, test_loader)
    print('test acc2:', test_acc2)

    time.sleep(2)

    model.load_state_dict(torch.load('best3.mdl'))
    print('loaded from 3!')
    test_acc3 = evalute(model, test_loader)
    print('test acc3:', test_acc3)


    model.load_state_dict(torch.load('best4.mdl'))
    print('loaded from 4!')
    test_acc1 = evalute(model, test_loader)
    print('test acc4:', test_acc1)

    time.sleep(2)

    model.load_state_dict(torch.load('best5.mdl'))
    print('loaded from 5!')
    test_acc2 = evalute(model, test_loader)
    print('test acc5:', test_acc2)

    time.sleep(2)

    model.load_state_dict(torch.load('best6.mdl'))
    print('loaded from 6!')
    test_acc3 = evalute(model, test_loader)
    print('test acc6:', test_acc3)



if __name__ == '__main__':
    main()





