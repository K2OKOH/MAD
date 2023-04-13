import cv2
import numpy as np
import glob
import os
import sys

def FreCom(img):
    h,w = img.shape[:2]
    img_dct = np.zeros((h,w,3))
    #img_dct = np.fft.fft2(img, axes=(0, 1))
    for i in range(3):
        img_ = img[:, :, i] # 获取rgb通道中的一个
        img_ = np.float32(img_) # 将数值精度调整为32位浮点型
        img_dct[:,:,i] = cv2.dct(img_)  # 使用dct获得img的频域图像

    return img_dct

def Matching(img,reference,alpha=0.2,beta=1):
    #lam = np.random.uniform(alpha, beta)
    theta = np.random.uniform(alpha, beta)
    h, w = img.shape[:2]
    img_dct=FreCom(img)
    r = np.random.randint(1,5)
    img_dct[r,r,:]=0
    ref_dct=FreCom(reference)
    img_fc = img_dct + ref_dct * theta
    img_out = np.zeros((h, w, 3))
    for i in range(3):
        img_ = img_fc[:, :, i]  # 获取rgb通道中的一个
        img_out[:, :, i] = cv2.idct(img_).clip(0,255)  # 使用dct获得img的频域图像

    return img_out


if __name__ == '__main__':

    print('start')
    img_path = './cityscape_s1/VOC2007/JPEGImages'     # 原始数据集
    save_path = './cityscape_s2/VOC2007/JPEGImages'    # 增强数据集
    
    file_name_list = os.listdir(img_path)
    
    print(save_path)

    os.makedirs(save_path)

    img_lists = glob.glob(img_path + '/*.jpg')

    img_basenames = []
    
    # 遍历所有的图片，取图片名
    for item in img_lists:
        img_basenames.append(os.path.basename(item))

    # print(img_basenames)
    i=0
    for img_n, img_p in zip(img_basenames,img_lists):
        img = cv2.imread(img_p)
        h1, w1 = img.shape[:2]
        # 如果 长宽 不是偶数 -> 缩放成偶数
        if h1%2!=0 or w1%2!=0:
            img=cv2.resize(img,(w1-w1%2,h1-h1%2),interpolation=cv2.INTER_AREA)

        refrence=np.ones_like(img)
        # 参考图片随机色彩 (三个通道分别随机化)
        refrence[:,:,0] = refrence[:,:,0]*np.random.randint(0,255)
        refrence[:,:,1] = refrence[:,:,1]*np.random.randint(0,255)
        refrence[:,:,2] = refrence[:,:,2]*np.random.randint(0,255)

        img_matched = Matching(img,refrence)
        cv2.imwrite(save_path + '/' + img_n, img_matched)
        print(i)
        i+=1


