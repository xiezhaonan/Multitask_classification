import cv2
import os
import json



# 读取json文件内容,返回字典格式
with open('E:\\shujuji\\Data_ICRA18\\cloth_metadata.json', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)

label_img = open('E:\\ccc\\all_chong\\label_img.txt', 'w')


def extract_img(video_file_path, save_path):
    times = 0

    camera = cv2.VideoCapture(video_file_path)
    frameFrequency = int(camera.get(7))
    while True:
        times += 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            image = cv2.resize(image, (224, 224))  # 分辨率
            cv2.imwrite(save_path + '.jpg', image)
            print(save_path + '.jpg')

    camera.release()

def extract_all_img(dir_path):
    for root, dirs, files in os.walk(dir_path):
        if len(files) > 0:
            video_file_path = root + '\\' + files[1]  # 视频路径
            root_split = root.split('\\')

            # save_path = 'E:\\ccc\\chong\\' + str(root_split[len(root_split) - 2]) + '_' + str(root_split[len(root_split) - 1])+"_"+str(json_data[root_split[len(root_split) - 2]])
            save_path = 'E:\\ccc\\all_chong\\' + str(root_split[len(root_split) - 2]) + '_' + str(root_split[len(root_split) - 1]+"_"+"".join(str(i) for i in json_data[root_split[len(root_split) - 2]]))
            # save_path = 'E:\\ccc\\chong\\' + str(root_split[len(root_split) - 2]) + '_' + str(root_split[len(root_split) - 1])
            extract_img(video_file_path, save_path)
            label_img.write(save_path + '.jpg,' + str(json_data[root_split[len(root_split) - 2]]) + '\n')
            label_img.flush()
    label_img.close()


extract_all_img('E:\\shujuji\\Data_ICRA18\\Data')













