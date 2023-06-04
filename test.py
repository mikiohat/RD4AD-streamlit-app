#@title test.py
# test.py
import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
from skimage.color import label2rgb, rgb2gray
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import seaborn as sns
import os
import streamlit as st

# amap_mode 'a'=addition(足し算),'mul'=multiplication(掛け算)
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):

    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size]) # 要素0の256×256行列
        #print(anomaly_map.shape)
    a_map_list = []
    #print(len(ft_list))
    for i in range(len(ft_list)): # list size = 3
        fs = fs_list[i] #i=0[1,256,64,64], i=1[1,512,32,32], i=2[1,1024,16,16]
        ft = ft_list[i] #i=0[1,256,64,64], i=1[1,512,32,32], i=2[1,1024,16,16]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        
        #plt.imshow(torch.squeeze(fs,dim=0).permute(1,2,0).to('cpu').detach().numpy())#.copy()) #.copy()不可でtensorとメモリを共有しないようにする
        #plt.axis('off')
        #plt.show()

        #torch.save(fs_list,f'./sample{i}.pt')

        #print(fs.size())
        #print(ft.size())
        a_map = 1 - F.cosine_similarity(fs, ft) # カラーdimensionでのコサイン類似度計算（defaultでdim=1、結果はsqueeze(dim=1)）。pixelレベルでのコサイン類似度になる
        #print("1: {}".format(a_map.size()))
        a_map = torch.unsqueeze(a_map, dim=1) #↑で削除したdimensionを追加（次元を合わせるため）
        #print("2: {}".format(a_map.size()))
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True) # サイズ256 にup/down sammpling
        #print("3: {}".format(a_map.size()))
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy() # 256*256=65536
        #print("4: {}".format(a_map.size))
        #print("4: {}".format(a_map[0][0]))
        #print(f'{i}')
        #pd.DataFrame(a_map).to_csv("test.csv")
        #print(a_map)
        #plt.imshow(a_map)
        #plt.axis('off')
        #plt.show()

        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
        #print("[{}]: {}".format(i,anomaly_map[0][0]))
        
    # anomaly_map＝3次元のi=0[1,1,64,64], i=1[1,1,32,32], i=2[1,1,16,16]⇒すべて[1,1,256,256]の各要素を足したもの
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    if anomaly_map.shape != img.shape:
      anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET) # 疑似カラーマップ
    return heatmap

def evaluation(encoder, bn, decoder, dataloader,device,_class_=None):
    #_, t_bn = resnet50(pretrained=True)
    #bn.load_state_dict(bn.state_dict())
    bn.eval()
    #bn.training = False
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, lb_type in dataloader:
            #print('img:{}, gt:{}, type:{}'.format(img.size(), gt.size(), lb_type))
            # plt.figure(figsize=(5,5))
            # plt.subplot(1,2,1)
            # plt.imshow(np.uint8(
            #     min_max_norm(
            #         torch.squeeze(img,dim=0).permute(1,2,0).to('cpu').detach().numpy())*255
            # ))
            # plt.axis('off')
            # plt.subplot(1,2,2)
            # plt.imshow(np.uint8(
            #     min_max_norm(
            #         torch.squeeze(gt,dim=0).permute(1,2,0).to('cpu').detach().numpy())*255
            # ))
            # plt.axis('off')
            # plt.show()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            #print('inp:{}'.format( [torch.Tensor.size(input) for input in inputs]))
            #print('out:{}'.format( [torch.Tensor.size(output) for output in outputs]))
            #print(img.shape[-1]) # 256
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            #print("len:{}, shape:{}".format(len(anomaly_map),anomaly_map.shape))

            #print(f'before max:{anomaly_map.max()}, min:{anomaly_map.min()}')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # plt.imshow(anomaly_map)
            # plt.axis('off')
            # plt.show()
            #print(f'after max:{anomaly_map.max()}, min:{anomaly_map.min()}')

            # print(f"gt:{gt.shape},gt[gt > 0.5].shape:{gt[gt > 0.5].shape},gt[gt <= 0.5].shape:{gt[gt <= 0.5].shape}")
            # ground truthを白(1)黒(0)に。goodはすべて0（size65536）になる
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            #print(f'{anomaly_map.shape}')
            #print(f'{anomaly_map[np.newaxis,:,:].shape}')

            if label.item()!=0:  # self.labels => good : 0, anomaly : 1
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            # print(gt.cpu().numpy().astype(int).ravel())
            # print(anomaly_map.ravel())
            # print(np.max(gt.cpu().numpy()))
            # print(np.max(anomaly_map))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

            # print(len(gt_list_px))
            # print(len(pr_list_px))
            # print(len(gt_list_sp))
            # print(len(pr_list_sp))

        #ano_score = (pr_list_sp - np.min(pr_list_sp)) / (np.max(pr_list_sp) - np.min(pr_list_sp))
        #vis_data = {}
        #vis_data['Anomaly Score'] = ano_score
        #vis_data['Ground Truth'] = np.array(gt_list_sp)
        # print(type(vis_data))
        # np.save('vis.npy',vis_data)
        #with open('{}_vis.pkl'.format(_class_), 'wb') as f:
        #    pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)

        #pd.DataFrame(gt_list_px).to_csv('carpte_gt.csv')
        #pd.DataFrame(pr_list_px).to_csv('carpte_pr.csv')
        
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

        fpr_all,tpr_all,th_all = roc_curve(gt_list_px, pr_list_px,drop_intermediate=False)
        print(f'{len(fpr_all)}, {len(tpr_all)}')
        #print(f'len(auc_threshold):{len(th_all)}')
        #print(f'auroc_px:{auroc_px}')
        plt.plot(fpr_all,tpr_all)
        plt.title('pixel-level AUC (Segmentation AUROC)')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()        
        plt.show()

        # 最適な閾値で分類
        thres = Find_Optimal_Cutoff(gt_list_sp,pr_list_sp)
        #thres = 0.57
        print(f'threshold:{thres}')
        thremap = map(lambda x: 1 if x > thres else 0, pr_list_sp)
        pr_list_sp_thres = list(thremap)
        sp_class_names = ["good", "others"]
        sp_labels, sp_classes = pd.factorize(sp_class_names)
        cf_matrix = pd.crosstab(sp_classes[np.int64(pr_list_sp_thres)],
                                sp_classes[np.int64(gt_list_sp)], 
                                rownames=['Predicted'],
                                colnames=['Actual'])
        plt.figure(figsize = (6,4))
        sns.heatmap(cf_matrix, cmap = "OrRd", annot = True, fmt = "g")
        
        # 分類ミスを可視化
        false_indexes  = [i for i,val in enumerate(np.logical_xor(gt_list_sp,pr_list_sp_thres)) if val]
        print(false_indexes)
        fcount = 0
        for img, gt, label, lb_type in dataloader:
          if fcount in false_indexes:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            print(f'type:{lb_type}, max:{anomaly_map.max()}, min:{anomaly_map.min()}')
            plt.figure(figsize=(5,5))
            plt.subplot(1,2,1)
            plt.imshow(min_max_norm((torch.squeeze(img,dim=0).permute(1,2,0).to('cpu').detach().numpy())))
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(min_max_norm((torch.squeeze(gt,dim=0).permute(1,2,0).to('cpu').detach().numpy())))
            plt.axis('off')
            plt.show()
          fcount+=1

        # precision_all, recall_all, prth_all = precision_recall_curve(gt_list_px, pr_list_px)
        # print(f'len(pr_threshold):{len(prth_all)}')
        # fig, ax = plt.subplots(facecolor="w")
        # ax.set_xlabel("Threshold")
        # ax.grid()
        # ax.plot(prth_all, precision_all[:-1], label="Precision")
        # ax.plot(prth_all, recall_all[:-1], label="Recall")
        # ax.legend()

        #pd.DataFrame({'th_all': th_all, 'tpr_all': tpr_all, 'fpr_all': fpr_all}).to_csv('roc_val.csv')
 
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)



def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------     
    list type, with optimal cutoff value        
    """
    fpr, tpr, threshold = roc_curve(target, predicted,drop_intermediate=False)
    i = np.arange(len(tpr)) 

    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    # youden index
    #roc = pd.DataFrame({'tf' : pd.Series(tpr+(1-fpr)-1, index=i), 'threshold' : pd.Series(threshold, index=i)})
    #roc_t = roc.iloc[roc.tf.argsort()[:1]]

    return list(roc_t['threshold']), fpr, tpr, threshold

def makelist(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)
    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './mvtec/' + _class_
    ckp_path = './checkpoints/' + 'wres50_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    if torch.cuda.is_available(): # GPUが使える場合
        ckp = torch.load(ckp_path)
    else: # CPUの場合(GPUで訓練したモデルをCPUで読み込む場合)
        ckp = torch.load(ckp_path,map_location=device)

    for k, v in list(ckp['bn'].items()):
        #print('k:{},  v:{}'.format(k,v.size()))
        if 'memory' in k:
            ckp['bn'].pop(k)
            #print(' up, memory!!!!') #ない
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    bn.eval()
    decoder.eval()
    img_list,anomaly_map_list,gt_list,label_list,lb_type_list,pro_th_list = [],[],[],[],[],[]
    
    with torch.no_grad():
        for img, gt, label, lb_type in test_dataloader:
            img = img.to(device)
            img_list.append(img)

            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map_list.append(anomaly_map)
            
            #############   gaussian  ##############
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt_list.append(gt)

            if label.item()!=0:  # self.labels => good : 0, anomaly : 1
                pro_th_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            else:
                pro_th_list.append(0.)

            label_list.append(label.item())
            lb_type_list.append(lb_type)
    
    if not os.path.exists('./data/'+_class_):
        os.makedirs('./data/'+_class_)
    torch.save(img_list, f'./data/{_class_}/img_list.pt')
    torch.save(anomaly_map_list, f'./data/{_class_}/anomaly_map_list.pt')
    torch.save(gt_list, f'./data/{_class_}/gt_list.pt')
    pd.DataFrame(label_list).to_csv(f'./data/{_class_}/label_list.csv',header=False,index=False)
    pd.DataFrame(lb_type_list).to_csv(f'./data/{_class_}/lb_type_list.csv',header=False,index=False)
    pd.DataFrame(pro_th_list).to_csv(f'./data/{_class_}/pro_th_list.csv',index=False)

    return img_list, anomaly_map_list, gt_list, label_list, list(pro_th_list)

def test(_class_, mklstflag=False):
    if mklstflag:
        img_list, anomaly_map_list, gt_list, label_list, pro_th_list = makelist(_class_)
    else:
        img_list = torch.load(f'./data/{_class_}/img_list.pt')
        anomaly_map_list = torch.load(f'./data/{_class_}/anomaly_map_list.pt')
        gt_list = torch.load(f'./data/{_class_}/gt_list.pt')
        label_list = pd.read_csv(f'./data/{_class_}/label_list.csv',header=None)
        pro_th_list = pd.read_csv(f'./data/{_class_}/pro_th_list.csv')

    #auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device,_class_)
    #print(_class_,':',auroc_px,',',auroc_sp,',',aupro_px)
    #st.title(auroc_px)

    return img_list, anomaly_map_list, gt_list, label_list, pro_th_list

def visualization(_class_):
    imgsave_path = './mvtec_test_img/' + _class_ + '/'
  
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './mvtec/' + _class_
    ckp_path = './checkpoints/' + 'wres50_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    count = 0
    with torch.no_grad():
        for img, gt, label, lb_type in test_dataloader:
            if (label.item() == 0):
                continue
            #if count <= 10:
            #    count += 1
            #    continue

            decoder.eval()
            bn.eval()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            #inputs.append(feature)
            #inputs.append(outputs)
            #t_sne(inputs)

            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB) # OpenCVはBGR、PillowはRGB、これを変換。BGR⇒RGB
            img = np.uint8(min_max_norm(img)*255)
            if not os.path.exists('./results_all/'+_class_):
               os.makedirs('./results_all/'+_class_)
            cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'org.png',img)
            plt.imshow(img)
            plt.axis('off')
            plt.savefig('org.png')
            plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'ad.png', ano_map)
            plt.imshow(ano_map)
            plt.axis('off')
            plt.savefig(imgsave_path+'{}{}.png'.format(lb_type[0],count))
            plt.show()
            gt = gt.cpu().numpy().astype(int)[0][0]*255
            cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'gt.png', gt)

            # b, c, h, w = inputs[2].shape
            # print(f'b,c,h,w:{b},{c},{h},{w}')
            # print(f'max before normalized:{inputs[2].max()}, after:{F.normalize(inputs[2], p=2).max()}')
            # print(f'shape:{F.normalize(inputs[2], p=2).shape} -> {F.normalize(inputs[2], p=2).view(c, -1).shape} -> {F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).shape}')
            # t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            # s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            # c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            # print(c.shape)
            # t_sne([t_feat, s_feat], c)
            # assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            count += 1
            #if count>20:
            #    return 0
                #assert 1==2



def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) :

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    #print(f'{type(masks)},{masks.shape}')
    #print(f'{type(masks.flatten())},{masks.flatten().shape}')
    #print(f'{type(set(masks.flatten()))},{set(masks.flatten())},{len(set(masks.flatten()))}')

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray" #amapsがndarrya型でなかったらAssertionエラー出力
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}" # masksの中身は0or1でなければならない。一旦flatten()に投げるのはsetが１次元でないとだめだから
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool) #全部Falseの配列をつくる

    #print(set(binary_amaps.flatten()))
    #print(amaps)
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    
    for th in np.arange(min_th, max_th, delta): # 200分割、threadは200or201個に
        #print(f'{th},{min_th}, {max_th}, {delta}')
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        #print(f'th:{th}, 0:{binary_amaps[amaps <= th].shape}, 1:{binary_amaps[amaps > th].shape}')

        pros = []

        # plt.figure(figsize=(5,5))
        # plt.subplot(1,2,1)
        # plt.imshow(binary_amaps.transpose(1,2,0)*255)
        # plt.axis('off')
        # plt.subplot(1,2,2)
        # plt.imshow(masks.transpose(1,2,0)*255)
        # plt.axis('off')
        # plt.show()

        for binary_amap, mask in zip(binary_amaps, masks):
            #print(f'{binary_amaps.shape},{binary_amap.shape} ')
            #print(f'{binary_amap}, {mask}')
            # plt.figure(figsize=(5,5))
            # plt.subplot(1,2,1)
            # plt.imshow(binary_amap*255)
            # plt.axis('off')
            # plt.subplot(1,2,2)
            # plt.imshow(mask*255)
            # plt.axis('off')
            # plt.show()
            #pd.DataFrame(measure.label(mask)).to_csv("_lm.csv")
            # measure.labelでground truthが白（1）の部分(連結・隣接領域にラベリングされた領域)を切り抜く。
            # 連結・隣接領域が２つ以上ある場合でもfor文で一つ一つregionとして渡される。
            for region in measure.regionprops(measure.label(mask)): 
                axes0_ids = region.coords[:, 0] # 連携領域の座標「row」
                axes1_ids = region.coords[:, 1] # 連携領域の座標「column」

                # plt.figure(figsize=(5,5))
                # plt.subplot(1,4,1)
                # plt.imshow(binary_amap*255)
                # plt.axis('off')
                # plt.subplot(1,4,2)
                # plt.imshow(mask*255)
                # plt.axis('off')
                # plt.subplot(1,4,3)
                # plt.imshow((1-mask)*255)
                # plt.axis('off')
                # plt.subplot(1,4,4)
                # plt.imshow(region.image)
                # plt.axis('off')
                # plt.show()
                
                # binary_amapにmaskが白い(1)の部分の座標を与えると、そこが1(True)か0(False)かが分かる。
                # True(1)のピクセル数をカウントし、pros.appendでmaskが白い領域全体のピクセル数のどれだけを占めるかを連結領域の数だけ追加
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area) # True Positive Rate
                #print(f'{np.unique(axes0_ids)}, {np.unique(axes1_ids)}')
                #print(f'bmaxarea:{binary_amap[axes0_ids, axes1_ids].shape}, tp_pixels:{tp_pixels}, region.area:{region.area}') 
                #print(binary_amap[axes0_ids, axes1_ids])
                #print(f'tp_pixels / region.area:{tp_pixels / region.area}')

            #print(pros)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum() # 黒(0)の部分なのに、binary_amapsが1の部分
        fpr = fp_pixels / inverse_masks.sum() # False Positive Rate

        #print(f'pros: {pros}')

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        #print(f'dfshape:{df.shape}')
        #df.to_csv('./prosdf.csv',index=False)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    # print(df["fpr"])
    df_less03 = df[df["fpr"] < 0.3]
    df_less03["fpr"] = (df_less03["fpr"] / df_less03["fpr"].max())

    #print(df)
    #print(df_less03)

    # 最適な閾値を計算 （statics）
    #i_roc_t = np.arange(len(df_less03['pro'])) # pro = true positive rate 
    #roc = pd.DataFrame({'tf' : pd.Series(df_less03['pro']-(1-df_less03['fpr']), index=i_roc_t),
    #                    'threshold' : pd.Series(df_less03['threshold'], index=i_roc_t)})
    # print(roc) 
    # print(roc.tf.argsort()[:1])
    # roc_thres = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    # print(f'roc_thres:{roc_thres.threshold}')
    # print(f'roc_thres:{roc_thres}')

    # 最適な閾値を計算 （youden index）
    i_roc_t = np.arange(len(df_less03['pro'])) # pro = true positive rate 
    roc = pd.DataFrame({'tf' : pd.Series(df_less03['pro']+(1-df_less03['fpr'])-1, index=i_roc_t),
                        'threshold' : pd.Series(df_less03['threshold'], index=i_roc_t)})
    roc_thres = roc.iloc[roc.tf.argsort()[:1]]
    # 最適な閾値を計算 （単純）
    # roc = pd.DataFrame({'tf' : pd.Series(df_less03['pro']-df_less03['fpr'], index=i_roc_t),
    #                     'threshold' : pd.Series(df_less03['threshold'], index=i_roc_t)})
    # roc_thres = roc.iloc[roc.tf.argsort()[:1]]


    #plt.figure(figsize=(5,5))
    plt.subplot(1,2,1)
    plt.plot(df['fpr'],df['pro'])
    plt.title("Per-Resion Overlap AUC before normalization",fontsize=7)
    plt.xlabel('FPR: False positive rate',fontsize=7)
    plt.ylabel('TPR: True positive rate',fontsize=7)
    plt.subplot(1,2,2)
    plt.plot(df_less03['fpr'],df_less03['pro'])
    # print(df_less03['fpr'].iloc[roc_thres.index])
    # print(df_less03['pro'].iloc[roc_thres.index])
    # print(df_less03[df_less03['threshold'] == roc_thres.iloc[0][1]])
    # print(roc_thres.iloc[0][0])
    # print(roc_thres.iloc[0][1])
    # print(roc_thres.iloc[0][2])        
    # Series.iloc[-1]で値だけを取り出せる。そうしないとNameとかDtypeとか一杯ついてる
    # screwで失敗
    # plt.plot(df_less03[df_less03['threshold'] == roc_thres.iloc[0][1]].fpr.iloc[-1],
    #          df_less03[df_less03['threshold'] == roc_thres.iloc[0][1]].pro.iloc[-1],
    #          '-Dr')
    plt.title("Per-Resion Overlap AUC",fontsize=7)
    plt.xlabel('FPR: False positive rate',fontsize=7)
    plt.ylabel('TPR: True positive rate',fontsize=7)
    plt.show()
    # print('PRO(TPR): {:.4f}, FPR: {:.4f}, thers: {:.4f}'.
    #       format(df_less03[df_less03['threshold'] == roc_thres.iloc[0][1]].pro.iloc[-1],
    #              df_less03[df_less03['threshold'] == roc_thres.iloc[0][1]].fpr.iloc[-1],
    #              roc_thres.threshold.iloc[-1]))

    binary_amaps[amaps <= roc_thres.iloc[0][1]] = 0
    binary_amaps[amaps > roc_thres.iloc[0][1]] = 1
    for binary_amap, mask in zip(binary_amaps, masks):
      plt.figure(figsize=(5,5))
      plt.subplot(1,2,1)
      plt.imshow(binary_amap)
      plt.axis('off')
      plt.subplot(1,2,2)
      plt.imshow(mask)
      plt.axis('off')
      plt.show()
      for region in measure.regionprops(measure.label(mask)): 
        axes0_ids = region.coords[:, 0] # 連携領域の座標「row」
        axes1_ids = region.coords[:, 1] # 連携領域の座標「column」
        # plt.imshow(region.image)
        # plt.axis('off')
        # plt.show()

    pro_auc = auc(df_less03["fpr"], df_less03["pro"])
    return roc_thres.iloc[0][1]
