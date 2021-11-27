import numpy as np
from torchvision import transforms
# from torchvision.models import
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import visdom
import os
import torch
import time
# import serial


# lr = 0.0001

root = os.getcwd()+'\\all_chong\\'




class Mydataset(Dataset):
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt_path, resize, mode,transform = None):
    # def __init__(self, root, resize, mode):
        super(Mydataset, self).__init__()

        self.resize = resize
        self.transform = transform
        fh= open(txt_path, 'r')      #按照传入的路径和txt文本参数，以只读的方式打开这个文本
        imgs = []
        # images = []
        label_list = []

        label_list1 = []
        label_list2 = []
        label_list3 = []
        label_list4 = []
        label_list5 = []
        label_list6 = []
        label_list7 = []
        label_list8 = []

        for line in fh:        ##########  迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')   # 删除 本行string 字符串末尾的指定字符

            words = line.split(',')   #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            label_list1.append((int(words[4])))
            label_list2.append((int(words[5])))
            label_list3.append((int(words[6])))
            label_list4.append((int(words[7])))
            label_list5.append((int(words[8])))
            label_list6.append((int(words[9])))
            label_list7.append((int(words[10])))
            label_list8.append((int(words[11])))
            # imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # images.append((words[0]))
            imgs.append((words[0]))

        # print(imgs)
        # print(len(imgs))
        # print(imgs)
        # self.images = images

        # print(len(label_list4))
        # print(label_list)
        self.images = imgs

        self.labels1 = label_list1
        self.labels2 = label_list2
        self.labels3 = label_list3
        self.labels4 = label_list4
        self.labels5 = label_list5
        self.labels6 = label_list6
        self.labels7 = label_list7
        self.labels8 = label_list8


        if mode == 'train':  ####  60%  traning
            self.images = self.images[:int(0.65 * len(self.images))]
            self.labels1 = self.labels1[:int(0.65 * len(self.labels1))]
            self.labels2 = self.labels2[:int(0.65 * len(self.labels2))]
            self.labels3 = self.labels3[:int(0.65 * len(self.labels3))]
            self.labels4 = self.labels4[:int(0.65 * len(self.labels4))]
            self.labels5 = self.labels5[:int(0.65 * len(self.labels5))]
            self.labels6 = self.labels6[:int(0.65 * len(self.labels6))]
            self.labels7 = self.labels7[:int(0.65 * len(self.labels7))]
            self.labels8 = self.labels8[:int(0.65 * len(self.labels8))]
        elif mode == 'val':  ####  60%-80%  test
            self.images = self.images[int(0.65 * len(self.images)):int(0.85 * len(self.images))]
            self.labels1 = self.labels1[int(0.65 * len(self.labels1)):int(0.85 * len(self.labels1))]
            self.labels2 = self.labels2[int(0.65 * len(self.labels2)):int(0.85 * len(self.labels2))]
            self.labels3 = self.labels3[int(0.65 * len(self.labels3)):int(0.85 * len(self.labels3))]
            self.labels4 = self.labels4[int(0.65 * len(self.labels4)):int(0.85 * len(self.labels4))]
            self.labels5 = self.labels5[int(0.65 * len(self.labels5)):int(0.85 * len(self.labels5))]
            self.labels6 = self.labels6[int(0.65 * len(self.labels6)):int(0.85* len(self.labels6))]
            self.labels7 = self.labels7[int(0.65 * len(self.labels7)):int(0.85 * len(self.labels7))]
            self.labels8 = self.labels8[int(0.65 * len(self.labels8)):int(0.85 * len(self.labels8))]

        else: #  ####  80%-100%  valuation
            self.images = self.images[int(0.85 * len(self.images)):]
            self.labels1 = self.labels1[int(0.85 * len(self.labels1)):]
            self.labels2 = self.labels2[int(0.85 * len(self.labels2)):]
            self.labels3 = self.labels3[int(0.85 * len(self.labels3)):]
            self.labels4 = self.labels4[int(0.85 * len(self.labels4)):]
            self.labels5 = self.labels5[int(0.85 * len(self.labels5)):]
            self.labels6 = self.labels6[int(0.85 * len(self.labels6)):]
            self.labels7 = self.labels7[int(0.85 * len(self.labels7)):]
            self.labels8 = self.labels8[int(0.85 * len(self.labels8)):]


        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************
    def __getitem__(self, index):         #用于按照索引读取每个元素的具体内容
        # print(index)
        x = self.images[index]  #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        # print(self.labels1)
        label1 = self.labels1[index]
        label2 = self.labels2[index]
        label3 = self.labels3[index]
        label4 = self.labels4[index]
        label5 = self.labels5[index]
        label6 = self.labels6[index]
        label7 = self.labels7[index]
        label8 = self.labels8[index]
        # img = Image.open(fn)
        # img = Image.open(root+x).convert('RGB')
        # ###按路径读取图片
        # print('1')
        # print(img)
        # print(type(img))
        # tf2 = transforms.Compose([
        #     lambda x: Image.open(root + x).convert('RGB'),  # string path => image data
        #     # transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
        #     # transforms.RandomRotation(15),  #####   旋转15度
        #     # transforms.CenterCrop(self.resize),
        #     # transforms.ToTensor(),
        #     transforms.ToTensor()
        #     # transforms.Normalize(mean=[0.9, 0.9, 0.9],
        #     #                      std=[0.15, 0.15, 15])
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #     #                      std=[0.229, 0.224, 0.225])
        #     # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #     #                      std=[0.5, 0.5, 0.5])
        #     # transforms.Normalize(mean=[0.3, 0.3, 0.3],
        #     #                      std=[0.3, 0.3, 0.3])
        # ])


        tf = transforms.Compose([
            lambda x: Image.open(root+x).convert('RGB'),  # string path => image data
            transforms.Resize((int(self.resize *1.7), int(self.resize * 1.7))),
            transforms.RandomRotation(5),  #####   旋转15度
            transforms.CenterCrop(self.resize),
            # transforms.ToTensor(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.9, 0.9, 0.9],
                                 # std=[0.15, 0.15, 15])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.3, 0.3, 0.3],
            #                      std=[0.3, 0.3, 0.3])
            # transforms.Normalize(mean=[0.37929365, 0.4124957, 0.38056934],
            #                      std=[0.09200633, 0.10309385, 0.08817028])
            ##########################################################train_mean: [0.39453954 0.43660778 0.3923058 ]train_std: [0.07784387 0.09333186 0.07919627]
            transforms.Normalize(mean=[0.37929365, 0.4124957, 0.38056934],
                                 std=[0.09200633, 0.10309385, 0.08817028])
          # 1.7  train_mean: [0.38549006 0.410828   0.38950878]，
          #   train_std: [0.07810324 0.08398561 0.08411217]
        ])


        label1 = torch.tensor(label1)
        label2 = torch.tensor(label2)
        label3 = torch.tensor(label3)
        label4 = torch.tensor(label4)
        label5 = torch.tensor(label5)
        label6 = torch.tensor(label6)
        label7 = torch.tensor(label7)
        label8 = torch.tensor(label8)
        # print(label1.shape)
        # if transforms is True:
        img = tf(x)    #数据标签转换为Tensor
        # print(img.shape)
        # else: img = self.transform(img)
        return img, label1, label2, label3, label4, label5, label6, label7, label8   #return回哪些内容，训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):
        return len(self.images)  #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分


def main():
    # viz = visdom.Visdom()
    # #
    train_data=Mydataset(txt_path=root+'2.txt',resize=224, mode='tarin')
    # val_data = Mydataset(txt_path=root + '2.txt', resize=224, mode='val')
    # test_data = Mydataset(txt_path=root + '2.txt', resize=224, mode='test')
    # # # print(len(train_data))
    # # # for data in train_data:
    # # #     print(data)
    # x, y1, y2, y3, y4, y5, y6, y7, y8 = next(iter(train_data))
    # # x, y1, y2, y3, y4, y5, y6, y7, y8 = next(iter(val_data))
    # # x, y1, y2, y3, y4, y5, y6, y7, y8 = next(iter(test_data))
    # print('sample', x.shape, y1)
    # # viz.image(x, win='sample_x', opts=dict(title='sample_x'))
    #
    loader1 = DataLoader(train_data, batch_size=4, shuffle=True)
    # loader2 = DataLoader(val_data, batch_size=4, shuffle=True)
    # loader3 = DataLoader(test_data, batch_size=4, shuffle=True)

    # for x, y1,y2,y3,y4,y5,y6,y7,y8 in loader1:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y1.numpy()), win='label1', opts=dict(title='batch-y1'))
    #     viz.text(str(y2.numpy()), win='label2', opts=dict(title='batch-y2'))
    #     viz.text(str(y3.numpy()), win='label3', opts=dict(title='batch-y3'))
    #     viz.text(str(y4.numpy()), win='label4', opts=dict(title='batch-y4'))
    #     viz.text(str(y5.numpy()), win='label5', opts=dict(title='batch-y5'))
    #     viz.text(str(y6.numpy()), win='label6', opts=dict(title='batch-y6'))
    #     viz.text(str(y7.numpy()), win='label7', opts=dict(title='batch-y7'))
    #     viz.text(str(y8.numpy()), win='label8', opts=dict(title='batch-y8'))
    # #
    #     time.sleep(2)
    # print('Finish!!!')
    #



    ###############################################################           Normalize   计算！！！！
    train = iter(loader1).next()[0]  # 3000张图片的mean、std
    train_mean = np.mean(train.numpy(), axis=(0, 2, 3))
    train_std = np.std(train.numpy(), axis=(0, 2, 3))

    print("train_mean:",train_mean)
    print("train_std:",train_std)








if __name__ == '__main__':
    main()







