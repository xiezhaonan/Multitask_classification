import torch
import os
import torch.optim as optim
from  torch import nn
from Mydataset import Mydataset
from torch.utils.data import DataLoader
# from resnet_cbam import ResNet, resnet50_cbam,resnet18_cbam,resnet101_cbam,resnet34_cbam,resnet152_cbam
from model import FC_model
import visdom ,time

# from torchvision.models import resnet50
root = os.getcwd()+'\\all_chong\\'
# root = os.getcwd()+'\\all_sample\\'
viz = visdom.Visdom()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.manual_seed(1234)

train_data = Mydataset(txt_path=root + '2.txt', resize=224, mode='train')
val_data = Mydataset(txt_path=root + '2.txt', resize=224, mode='val')
test_data = Mydataset(txt_path=root + '2.txt', resize=224, mode='test')

x, y1, y2, y3, y4, y5, y6, y7, y8 = next(iter(train_data))
# x, y1, y2, y3, y4, y5, y6, y7, y8 = next(iter(val_data))
# x, y1, y2, y3, y4, y5, y6, y7, y8 = next(iter(test_data))


train_loader = DataLoader(train_data, batch_size=6, shuffle=True)
val_loader = DataLoader(val_data, batch_size=6, shuffle=True,drop_last=True)
test_loader = DataLoader(test_data, batch_size=6, shuffle=True)

# for x, y1, y2, y3, y4, y5, y6, y7, y8 in train_loader:
#     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
#     viz.text(str(y1.numpy()), win='label1', opts=dict(title='batch-y1'))
#     viz.text(str(y2.numpy()), win='label2', opts=dict(title='batch-y2'))
#     viz.text(str(y3.numpy()), win='label3', opts=dict(title='batch-y3'))
#     viz.text(str(y4.numpy()), win='label4', opts=dict(title='batch-y4'))
#     viz.text(str(y5.numpy()), win='label5', opts=dict(title='batch-y5'))
#     viz.text(str(y6.numpy()), win='label6', opts=dict(title='batch-y6'))
#     viz.text(str(y7.numpy()), win='label7', opts=dict(title='batch-y7'))
#     viz.text(str(y8.numpy()), win='label8', opts=dict(title='batch-y8'))
#
#     # time.sleep(4)
# print('Finish!!!')



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
    # pass
    model = FC_model().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    # criteon = nn.CrossEntropyLoss()
    criteon = nn.CrossEntropyLoss(reduction='mean')

    best_acc1, best_epoch1 = 0, 0
    best_acc2, best_epoch2 = 0, 0
    best_acc3, best_epoch3 = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))



    for epoch in range(60):
        for step, (x,y1,y2,y3,y4,y5,y6,y7,y8) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            x, y1 = x.to(device), y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)
            y4 = y4.to(device)
            y5 = y5.to(device)
            y6 = y6.to(device)
            y7 = y7.to(device)
            y8 = y8.to(device)

            model.train()
            logits = model(x)
            # print('1')
            loss1 = criteon(logits[0], y1)
            loss2 = criteon(logits[1], y2)
            loss3 = criteon(logits[2], y3)
            loss4 = criteon(logits[3], y4)
            loss5 = criteon(logits[4], y5)
            loss6 = criteon(logits[5], y6)
            loss7 = criteon(logits[6], y7)
            loss8 = criteon(logits[7], y8)


            # Loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8
            Loss = 0.7 * loss1 + 0.7 * loss2 + 0.7 * loss3 + 1.1 * loss4 + 0.7 * loss5 + 1.2 * loss6 + 1.3 * loss7 + 1.6 * loss8
            # print('2')
            # opt1.zero_grad()
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            # opt1.step()

            print('epoch:', epoch, 'loss', Loss)
            viz.line([Loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:

            val_acc1, val_acc2, val_acc3, val_acc4,val_acc5, val_acc6, val_acc7, val_acc8 = evalute(model,val_loader)
            if  val_acc8 > best_acc1 and val_acc8 > best_acc2 and val_acc8 > best_acc3 and epoch<=40:
                best_epoch1 = epoch
                best_acc1 = val_acc8
                torch.save(model.state_dict(), 'best1.pkl')
                print('1')
            elif val_acc8 > best_acc2 and val_acc8 >best_acc3 and epoch<=40:
                best_epoch2 = epoch
                best_acc2 = val_acc8
                print('2')
                torch.save(model.state_dict(), 'best2.pkl')

            elif val_acc8 >best_acc3 and epoch<=40:
                best_epoch3 = epoch
                best_acc3 = val_acc8
                torch.save(model.state_dict(), 'best3.pkl')
                print('3')
            # else:
            #     torch.save(model.state_dict(), 'last.mdl')
            if val_acc8 > best_acc1 and val_acc8 > best_acc2 and val_acc8 > best_acc3 and epoch>40:
                best_epoch1 = epoch
                best_acc1 = val_acc8
                torch.save(model.state_dict(), 'best4.pkl')
                print('4')
            elif val_acc8 > best_acc2 and val_acc8 >best_acc3 and epoch>40:
                best_epoch2 = epoch
                best_acc2 = val_acc8
                print('5')
                torch.save(model.state_dict(), 'best5.pkl')

            elif val_acc8 >best_acc3 and epoch>40:
                best_epoch3 = epoch
                best_acc3 = val_acc8
                torch.save(model.state_dict(), 'best6.pkl')
                print('6')
            # else:
            torch.save(model.state_dict(), 'last.pkl')




            viz.line([val_acc8], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc1, 'best epoch:', best_epoch1)
    print('best acc:', best_acc2, 'best epoch:', best_epoch2)
    print('best acc:', best_acc3, 'best epoch:', best_epoch3)
    time.sleep(1)
    model.load_state_dict(torch.load('best1.pkl'))
    print('loaded from 1!')
    test_acc1 = evalute(model, test_loader)
    print('test acc1:', test_acc1)

    time.sleep(2)

    model.load_state_dict(torch.load('best2.pkl'))
    print('loaded from 2!')
    test_acc2 = evalute(model, test_loader)
    print('test acc2:', test_acc2)

    time.sleep(2)

    model.load_state_dict(torch.load('best3.pkl'))
    print('loaded from 3!')
    test_acc3 = evalute(model, test_loader)
    print('test acc3:', test_acc3)

    time.sleep(2)

    model.load_state_dict(torch.load('best4.pkl'))
    print('loaded from 4!')
    test_acc3 = evalute(model, test_loader)
    print('test acc4:', test_acc3)

    time.sleep(2)

    model.load_state_dict(torch.load('best5.pkl'))
    print('loaded from 5!')
    test_acc3 = evalute(model, test_loader)
    print('test acc5:', test_acc3)

    time.sleep(2)

    model.load_state_dict(torch.load('best6.pkl'))
    print('loaded from 6!')
    test_acc3 = evalute(model, test_loader)
    print('test acc6:', test_acc3)

    time.sleep(2)

    model.load_state_dict(torch.load('last.pkl'))
    print('loaded from last!')
    test_acc3 = evalute(model, test_loader)
    print('test acc last:', test_acc3)




if __name__ == '__main__':
    main()











