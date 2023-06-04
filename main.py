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
from sklearn.metrics import accuracy_score,precision_score,recall_score
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
#import matplotlib
import pickle
import seaborn as sns
import os
import streamlit as st
import test
import time

#@st.cache

def main():
    st.set_page_config(page_icon="🐍", page_title="Streamlit Ev-App")

    with st.sidebar:
        left_column, right_column = st.columns(2)
        left_column.write('Data Initialization')
        ev_bt = left_column.button('Evaluation',key='sbt1')
        data_sb = right_column.selectbox("Choose a data sample",("bottle", "carpet", "grid", "screw"))
        if ev_bt == True:
            #with st.empty():
                #st.text('processing...')
                images, anomaly_maps, gts, labels, pro_ths= test.test(data_sb,mklstflag=True)
                #st.text('finished')
            

    # with st.sidebar:
    #     with st.form('my_form'):
    #         st.write("Inside the form")
    #         slider_val = st.slider("Form slider")
    #         checkbox_val = st.checkbox("Form checkbox")

    #         # Every form must have a submit button.
    #         submitted = st.form_submit_button("Submit")
    #     if submitted:
    #         st.write("slider", slider_val, "checkbox", checkbox_val)

    images, anomaly_maps = [], []
    images, anomaly_maps, gts, labels, pro_ths= test.test(data_sb)

    tab1,tab2=st.tabs(["Detection","Per-Region Overlap"])

    with tab1:
        #gt_list_px, pr_list_px = [], []
        gt_list_sp, pr_list_sp = [], []
        for gt, anomap in zip(gts, anomaly_maps):
            #gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            #pr_list_px.extend(anomap.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomap))

        cold1,cold2 = st.columns(2)
    
        # 最適な閾値で分類
        init_thres_d, fpr_all, tpr_all, th_all = test.Find_Optimal_Cutoff(gt_list_sp,pr_list_sp)
        cold1.markdown(f'Optimal Threshold: **{np.round(init_thres_d,2)}**')
        #st.write(init_thres_d[0])    
        # thres_d = cold1.slider('Threshold value',float(th_all.min()) ,float(th_all.max()),
        #                     init_thres_d,key='sdd1')
        thres_d = cold1.number_input('Threshold Value', min_value=0.00,
                                     max_value=1.50,value=float(init_thres_d[0]),step=0.01,key='ni34')

        thremap = map(lambda x: 1 if x > thres_d else 0, pr_list_sp)
        pr_list_sp_thres = list(thremap)
        sp_class_names = ["Normal", "Anomaly"]
        _, sp_classes = pd.factorize(sp_class_names)
        cf_matrix = pd.crosstab(sp_classes[np.int64(pr_list_sp_thres)],
                                sp_classes[np.int64(gt_list_sp)], 
                                rownames=['Predicted'],
                                colnames=['Actual'])
    
        fig2 = plt.figure()
        sns.heatmap(cf_matrix, cmap = "OrRd", annot = True, fmt = "g")
        cold1.pyplot(fig2)
    
        cold1.markdown(f'Accuracy: :green[**{np.round(accuracy_score(y_true=gt_list_sp,y_pred=pr_list_sp_thres),3)}**]')
        cold1.markdown(f'Precision: :green[**{np.round(precision_score(y_true=gt_list_sp,y_pred=pr_list_sp_thres),3)}**]')
        cold1.markdown(f'Recall: :green[**{np.round(recall_score(y_true=gt_list_sp,y_pred=pr_list_sp_thres),3)}**]')


        # 分類ミスを可視化
        false_indexes  = [i for i,
                          val in enumerate(np.logical_xor(gt_list_sp,pr_list_sp_thres)) if val]
        cold2.markdown(f'Count of Discrepancy: **{len(false_indexes)}**')

        for fi in false_indexes:
            data_img = images[fi].permute(0, 2, 3, 1).cpu().numpy()[0]
            data_img = np.uint8(test.min_max_norm(data_img)*255) 
            anom_img = test.cvt2heatmap(test.min_max_norm(anomaly_maps[fi])*255)
            anom_img = cv2.cvtColor(anom_img, cv2.COLOR_BGR2RGB)
            anom_img = test.show_cam_on_image(data_img, anom_img)
            acflg, preflg = None, None 
            if labels[0][fi] == 1:
                acflg = 'anom'
                prflg = 'norm'
            else:
                acflg = 'norm'
                prflg = 'anom'
            cold2.image(anom_img,caption=f'No. {fi}, Actual: [{acflg}], Pred: [{prflg}]')

    with st.sidebar:
        st.markdown('No.&ensp;:red[Discrepant]&ensp;:blue[Anomaly]&ensp;Normal')
        num_str = ''
        for i, (pr_val, gt_val) in enumerate(zip(pr_list_sp_thres,gt_list_sp)):
            if np.logical_and(pr_val, gt_val):
                num_str += f':blue[**{i}**]&ensp;'
            elif np.logical_xor(pr_val, gt_val):
                num_str += f':red[**{i}**]&ensp;'
            else:
                num_str += f'**{i}**&ensp;'
        num_str = num_str[:-len('&ensp;')]
        st.markdown(num_str)





    with tab2:
        #st.title('画像2値e化アプリ')
        st.write("---")

        col_1,col_2 = st.columns(2)
        num =col_1.number_input('No.',min_value=0,max_value=len(images)-1,step=1,key='number1')
        col_1.markdown(f'Max Value: **{len(images)-1}**')
        goodflg = False if labels[0][num] == 1 else True

        ini_thres = 0
        if goodflg:
            ini_thres = float(pro_ths.median())
        else:
            ini_thres = float(pro_ths.iloc[num])
        th = col_2.slider('Threshold Value',float(pro_ths.min()) ,float(pro_ths.max()),
                        ini_thres,key='sd1')
        col_2.markdown(f"Optimal Threshold: **{np.round(ini_thres,2)}**")

        col1,col2,col3,col4 = st.columns(4)

        if len(images) > 0:           
            data_img = images[num].permute(0, 2, 3, 1).cpu().numpy()[0]
            data_img = np.uint8(test.min_max_norm(data_img)*255) 
            #size=(86,86)
            #data_resize = cv2.resize(data_img,size)
            col1.image(data_img, caption="Input Data")

        if len(anomaly_maps) > 0:
            anom_img = test.cvt2heatmap(test.min_max_norm(anomaly_maps[num])*255)
            anom_img = cv2.cvtColor(anom_img, cv2.COLOR_BGR2RGB)
            anom_img = test.show_cam_on_image(data_img, anom_img)
            col2.image(anom_img,caption='Anomaly Map')

        binary_amap = np.zeros_like(anomaly_maps[num], dtype=np.bool) #全部Falseの配列をつくる
        binary_amap[anomaly_maps[num] <= th] = 0
        binary_amap[anomaly_maps[num] > th] = 1

        pros = []
        fpr = 0
        if not goodflg:
            #st.write(gts[num].squeeze().shape)
            for region in measure.regionprops(measure.label(gts[num].squeeze().cpu().numpy())): 
                axes0_ids = region.coords[:, 0] # 連携領域の座標「row」
                axes1_ids = region.coords[:, 1] # 連携領域の座標「column」
                #col4.image(np.uint8(region.image)*255)
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area) # True Positive Rate
                
                inverse_masks = 1 - gts[num].squeeze().cpu().numpy()
                fp_pixels = np.logical_and(inverse_masks, binary_amap).sum() # 黒(0)の部分なのに、binary_amapsが1の部分
                fpr = fp_pixels / inverse_masks.sum() # False Positive Rate

        # 描画直前で色調補正しないとproのエリア比率計算がおかしなことになる
        #binary_amap = np.uint8(binary_amap*255)
        binary_amap = test.cvt2heatmap(binary_amap*255)
        binary_amap = cv2.cvtColor(binary_amap, cv2.COLOR_BGR2RGB)

        col3.image(binary_amap, caption=f'TPR: {round(np.mean(pros),3)}, FPR: {round(fpr,3)}')

        if len(gts) > 0:
            if goodflg :
                col4.write('No Image')
            else:
                gt_img = gts[num].permute(0, 2, 3, 1).cpu().numpy()[0]
                gt_img = np.uint8(test.min_max_norm(gt_img)*255) 
                col4.image(gt_img, caption='Ground Truth')            

if __name__ == '__main__':
    main()
    #test('grid')