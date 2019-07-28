import os
import random
import cv2

path = "E:\\Umich\\Dataset\\labeled"


def rename_all_files(path):
    filelist = os.listdir(path)
    count = 0
    for file in filelist:
        print(file)
    for file in filelist:
        Olddir = os.path.join(path, file)
        if os.path.isdir(Olddir):
            rename_all_files(Olddir)
            continue
        filetype = os.path.splitext(file)[1]
        Newdir = os.path.join(path, str(count).zfill(4) + filetype)
        os.rename(Olddir, Newdir)
        count += 1


def any2jpg(input_path, output_path):

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    filelist = os.listdir(input_path)
    for file in filelist:
        Olddir = os.path.join(input_path, file)
        if os.path.isdir(Olddir):
            continue
        file_name = os.path.splitext(file)[0]
        file_type = '.jpg'
        Newdir = os.path.join(output_path, file_name + file_type)
        try:
            img = cv2.imread(Olddir)
        except:
            continue
        cv2.imwrite(Newdir, img)


def divide_dataset(path):

    trainval_percent = 0.9
    train_percent = 0.9
    xmlfilepath = os.path.join(path, 'Annotations')
    txtsavepath = os.path.join(path, 'ImageSets\\Main')

    if not os.path.exists(xmlfilepath):
        raise ValueError("Error: Path doesn't exist")

    if not os.path.exists(txtsavepath):
        os.mkdir(txtsavepath)

    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    tv = int(num*trainval_percent)
    tr = int(tv*train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
    ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
    fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

    for i in list:
        name = total_xml[i][:-4]+'\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest .close()


any2jpg("E:\\Umich\\waiting_for_label\\images", "E:\\Umich\\waiting_for_label\\throat_dataset\\JPEGImages")