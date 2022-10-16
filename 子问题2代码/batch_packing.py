# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:27:16 2022

@author: yy
"""
import time
import pandas as pd
import numpy as np
import copy
from pack import packing,view
file_num = list(range(1,6))
dataB = None
for f_num in file_num:
    data_file = r'D:/githubWork/方形件组批优化/子问题2-数据集B/dataB{}.csv'.format(f_num)
    if dataB is None:
        dataB = pd.read_csv(data_file,header=0)
    else:
        dataBi = pd.read_csv(data_file,header=0)
        
        dataB = pd.concat((dataB,dataBi),axis=0)

orders = list(dataB['item_order'].value_counts().keys())
dataB['area'] = dataB['item_length'] * dataB['item_width']

order_item_area = dataB.groupby(by='item_order')['area'].sum()
order_item_num = dataB.groupby(by='item_order')['item_num'].count()

materials = list(dataB['item_material'].value_counts().keys())
meaterial_codenum =  dict(zip(materials, list(range(len(materials)))))
# 参数
max_item_area = 250 * 1e6
max_item_num = 1000
plate_length = 2440
plate_width = 1220 
plate_area = plate_length * plate_width

all_need = np.zeros((len(orders),len(materials))) #创建需求矩阵
for i,order in enumerate(orders):
    order_df = dataB[dataB['item_order'] == order]
    order_materials = order_df['item_material'].value_counts()
    for m,num in order_materials.items():
        marea = sum(order_df[order_df['item_material'] == m]['area'])
        all_need[i,meaterial_codenum[m]] = num

        
all_need[all_need > 0] = 1

all_need = all_need.astype(np.int32)
# 杰卡德相似系数 J(A,B) = |A∩B| / |A∪B| = sum(A & B) / sum(A|B)
# 杰卡德距离 1-J(A,B)


def JKDM(batchs_center):
    '''计算类之间的杰卡德距离
    return batchJM
    '''
    batchs_center_np = np.array(batchs_center)
    batch_num = len(batchs_center_np)
    batchJM = np.zeros((batch_num,batch_num))
    for i in range(batch_num):
        AnB = batchs_center_np[i] & batchs_center_np
        AuB = batchs_center_np[i] | batchs_center_np
        batchJM[i,:] = 1 - np.sum(AnB,axis=1) / np.sum(AuB,axis=1)
        batchJM[i,:i+1] = 1
    return batchJM


def JKDMSingle(batch_center,left_orders):
    '''计算杰卡德距离
    batch_center   批次中心
    left_orders    待分配order
    '''
    AnB = batch_center & left_orders
    AuB = batch_center | left_orders
    JKD = 1 -  np.sum(AnB,axis=1) / np.sum(AuB,axis=1)
    best_index = np.argmin(JKD)
    # batch_center += left_orders[batch_center]
    return best_index

class Order(object):
    def __init__(self,order_id,order_needs,item_area,item_num):
        self.id = order_id
        self.needs = order_needs
        self.item_area = item_area
        self.item_num = item_num
    
        
class Batch(object):
    def __init__(self,max_item_area,max_item_num):
        self.left_area = max_item_area
        self.left_num = max_item_num
        self.orders = []
        self.needs = None
        self.area = 0
    
    def addOrder(self,order):
        self.area += order.item_area
        self.left_area -= order.item_area
        self.left_num -= order.item_num
        self.orders.append(order)
        if self.needs is None:
            self.needs = order.needs
        else:
            self.needs = pd.concat((self.needs,order.needs),axis=0)
    


# 聚类
# 批次划分方法1 基于杰卡德距离 2 平方距离 3 贪婪

def batch_division(orders_class,all_need,batch_method=1):
    batchs = []
    labels = []
    left_order_indexs = list(range(len(orders_class)))
    if batch_method in [1,2]:
        while len(labels) < len(orders_class):
            if batch_method == 1:
                batchJM = JKDM(all_need[left_order_indexs])
                rows,cols = np.where(batchJM == np.min(batchJM)) #可能存在多个最小值点，取一个就行了
                batch_first_order_index = rows[0]
            elif batch_method == 2:
                batch_first_order_index = np.argmin(all_need[left_order_indexs].sum(axis=1))
            else:
                break
                
            batch_first_order_index = left_order_indexs[batch_first_order_index]
            labels.append(batch_first_order_index)
            left_order_indexs.remove(batch_first_order_index)
            # 初始化当前batch
            batch = Batch(max_item_area,max_item_num)
            batch.addOrder(orders_class[batch_first_order_index])
            batch_center = all_need[batch_first_order_index]
            
            while batch.left_area > 0 and batch.left_num > 0:
                if left_order_indexs:
                    left_orders = copy.copy(all_need[left_order_indexs])
                    if batch_method == 1:
                        best_index = JKDMSingle(batch_center,left_orders)
                    elif batch_method == 2:
                        left_orders -= batch_center
                        left_orders *= left_orders
                        best_index = np.argmin(left_orders.sum(axis=1))
                    best_index = left_order_indexs[best_index]
                    if batch.left_area >= orders_class[best_index].item_area and  \
                        batch.left_num >= orders_class[best_index].item_num:
                        batch_center += all_need[best_index]
                        batch_center[batch_center>0] = 1
                        batch.addOrder(orders_class[best_index])
                        labels.append(best_index)
                        left_order_indexs.remove(best_index)
                    else:
                        batchs.append(batch)
                        break
                else:
                    batchs.append(batch)
                    break
        
    # 按order顺序贪婪划分批次
    if batch_method == 3:
        orders_list = copy.copy(orders_class)
        while orders_list:
            batch = Batch(max_item_area,max_item_num)
            labels = []
            for i,order in enumerate(orders_list):
                if batch.left_area >= order.item_area and batch.left_num >= order.item_num:
                    batch.addOrder(order)
                    labels.append(i)
    
            if i == len(orders_list)-1:
                labels = sorted(labels,reverse=True)
                batchs.append(batch)
                for label in labels:
                    orders_list.pop(label)
    return batchs


### 检验batch是否合理正确
def checkBatch(batchs,dataB):
    num_item = 0
    flag = True
    for i,batch in enumerate(batchs):
        num_item += len(batch.needs)
        if batch.left_area < 0 or batch.left_num < 0:
            flag = False
            break
    if len(dataB) == num_item and flag:
        print('a nice batch division！')
        
 

            
import random      
# 模拟退火,order的位置是可以调整的
#这里可以做优化
orders_class = [] 
for order_id in orders:
    order_needs = dataB[dataB['item_order'] == order_id]
    item_area = sum(order_needs['area'])
    item_num = sum(order_needs['item_num'])
    order = Order(order_id,order_needs,item_area,item_num)
    orders_class.append(order)
indexs = list(range(len(orders_class)))

def mutation(orders_class):
    '''选择两个位点进行逆转
    '''
    orders_list = orders_class
    a,b = random.sample(indexs,2)
    if a > b:
        a,b = b,a
    orders_list[a:b] = orders_list[a:b][::-1]
    return orders_list
            


def batch_test(batchs,dataB):
    batchs_bin_num = 0
    # start_time = time.time()
    for i,batch in enumerate(batchs):
        batch_df = batch.needs
        data_groupby_material = {}
        materials = batch_df['item_material'].value_counts().keys() 
        for m in materials:
            data_groupby_material[m] = batch_df[batch_df['item_material'] == m]
        batch_bin_num = 0
        for m in materials:
            mdata = data_groupby_material[m]
            mdata.sort_values(by=['item_length','item_width'],axis=0,ascending=False,inplace=True)
            mdata.reset_index(inplace=True,drop=True)
            bin_num,bins = packing(mdata)
            batch_bin_num += bin_num
        batchs_bin_num += batch_bin_num
    total_area = sum(dataB['area'])
    batchs_eff = total_area / (batchs_bin_num * plate_area)
    # print('总体平均利用率:',np.mean(batchs_eff))
    # cost_time = time.time() - start_time
    # print('计算时间:',cost_time) 
    return batchs_eff

# 模拟退火算法求解    
# # random.shuffle(orders_class)
# orders_class = mutation(orders_class)
# batchs = batch_division(orders_class,all_need)
# checkBatch(batchs,dataB)  
# T = 37 #初始温度
# Tfloor = 5 #温度下限
# alpha = 0.8 #降温系数
# iter_count = 5 #每一温度下的迭代次数
# ## 初始解
# best_orders_class = orders_class
# best_batchs = batch_division(orders_class,all_need,batch_method=1)

# best_batchs_eff = batch_test(best_batchs,dataB)
# record = [best_batchs_eff]
# while T > Tfloor:
#     print('当前温度:',T)
#     start_time = time.time()
#     for i in range(iter_count):
#         cand_orders_class = mutation(orders_class)
#         cand_batchs = batch_division(cand_orders_class,all_need,batch_method=3)
#         cand_batchs_eff = batch_test(cand_batchs,dataB)
#         if cand_batchs_eff >= best_batchs_eff:
#             best_orders_class = cand_orders_class
#             orders_class = cand_orders_class
#             best_batchs = cand_batchs
#             best_batchs_eff = cand_batchs_eff
#         else:
#             p = np.exp((best_batchs_eff - cand_batchs_eff)/T)
#             if random.random() < p:
#                 orders_class = cand_orders_class
                
#     cost_time = time.time() - start_time
#     print('迭代时间:',cost_time)
    
#     T *= alpha
                
#     record.append(best_batchs_eff)
    
# print(record)
batchs = batch_division(orders_class,all_need)
batch_eff = []
batchs_bin_num = 0
start_time = time.time()
for i,batch in enumerate(batchs):
    batch_df = batch.needs
    data_groupby_material = {}
    
    materials = batch_df['item_material'].value_counts().keys() 
    for m in materials:
        data_groupby_material[m] = batch_df[batch_df['item_material'] == m]
        
    batch_bin_num = 0
    for m in materials:
        mdata = data_groupby_material[m]
        mdata.sort_values(by=['item_length','item_width'],axis=0,ascending=False,inplace=True)
        mdata.reset_index(inplace=True,drop=True)
        bin_num,bins = packing(mdata)
        batch_bin_num += bin_num
    batch_eff = batch.area / (batch_bin_num * plate_area)
    print('batch{}平均利用率:'.format(i+1),batch_eff)
    batchs_bin_num += batch_bin_num

total_area = sum(dataB['area'])
batchs_eff = total_area / (batchs_bin_num * plate_area)
print('总体平均利用率:',np.mean(batchs_eff))
cost_time = time.time() - start_time
print('计算时间:',cost_time) 













 

