# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 07:55:23 2022

@author: yy
"""

import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# plt.rcParams['savefig.dpi'] = 1000 

from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 5,
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False, # 处理负号，即-号
            'figure.dpi':800,
            'legend.fontsize':5
         }
rcParams.update(config)


plate_length = 2440 #原片长度
plate_width = 1220 #原片宽度
plate_area = plate_length * plate_width


class Item(object):
    def __init__(self,material,item_id,item_length,item_width):
        '''
        material 材料编码,这里请注意，只有满足item_width < plate_width才能被切割，这里在做方向变化的时候一定要注意
        '''
        self.w = item_width
        self.h = item_length
        self.id = item_id
        self.material = material
        
    def __repr__(self):
        return 'the item type ({},{})'.format(self.h,self.w)
    

class Stack(object):
    def __init__(self,stripe,item):
        '''
        item
        '''
        self.w = item.w             #每一个stack的宽有该stack中第一个item的宽决定
        self.uh = stripe.h          #每一个stack的高不超过该stack所属stripe的高
        self.items = []
        
    
    
    def addItem(self,item):
        '''按宽叠加'''      #只有等宽item才能放入一个stack中
        self.uh -= item.h
        self.items.append(item)
    
    
class Stripe(object):
    def __init__(self,plate_width):
        self.uw = plate_width   #stripe的宽度为原片的宽度
        self.w = plate_width
        self.stacks = []
        self.h = 0
        
    def addStack(self,stack):
        self.uw -= stack.w
        self.stacks.append(stack)
    
class Bin(object):
    def __init__(self,bin_id,plate_length,plate_width):
        self.id = bin_id       #原片编号
        self.w = plate_width
        self.h = plate_length
        self.uh = plate_length #原片剩余高
        self.stripes = []
        
    def addStripe(self,stripe):
        self.uh -= stripe.h
        self.stripes.append(stripe)
        
    
            
    def output_result(self):
        curr_h = 0
        out_dict = {"批次序号":[],"原片材质":[],"原片序号":[],"产品id":[],"产品x坐标":[],"产品y坐标":[],"产品x方向长度":[],"产品y方向长度":[]}
        for stripe in self.stripes:
            curr_w = 0    
            for stack in stripe.stacks:
                tmp_h = curr_h
                for item in stack.items:
                    out_dict["批次序号"].append(0)
                    out_dict["原片材质"].append(item.material)
                    out_dict["原片序号"].append(self.id)
                    out_dict["产品id"].append(item.id)
                    out_dict["产品x坐标"].append(curr_w)
                    out_dict["产品y坐标"].append(tmp_h)
                    out_dict["产品x方向长度"].append(item.w)
                    out_dict["产品y方向长度"].append(item.h)
                    tmp_h += item.h
                curr_w += stack.w
            curr_h += stripe.h
        
        
        return out_dict
    
def delete(item_ids,labels):
    for label in labels:
        item_ids.remove(label)
    return item_ids
  

def packing(data):
    '''data :DataFrame
    '''
    # mini_width = min(data['item_width'])
    # mini_height = min(data['item_length'])
    left_items = {}
    for i in range(len(data)):
        item_id = data['item_id'][i]
        length = data['item_length'][i]
        width = data['item_width'][i]
        left_items[item_id] = Item(data['item_material'][i],item_id,length,width)
    
    bin_num = 0
    bins = []
    cur_bin = Bin(bin_num,plate_length,plate_width)
            
    stripe = Stripe(plate_width) #strip.h=0
    item_ids = copy.copy(list(left_items.keys()))
    
    while item_ids:
        mini_width = min(data[data['item_id'].isin(item_ids)]['item_width'])
        mini_height = min(data[data['item_id'].isin(item_ids)]['item_length'])
        
        
        labels = [] #标记是否装载
        # print('bin_num:',bin_num)
        for i,item_id in enumerate(item_ids):
                
            item = left_items[item_id]
            
            flag = False
            first = False
            #当前stripe为空，且当前item可以装入bin
            if stripe.h == 0 and item.h <= cur_bin.uh:
                # 当前stripe左下角第一个
                # 该item的height作为stripe的height
                stripe.h = item.h
                
                stack = Stack(stripe,item) 
                
                stack.addItem(item)
                
                stripe.addStack(stack)
                labels.append(item_id)            
                first = True
                
            # 原有stack中还可以装载
            
            # if not first and len(stripe.stacks) > 1:
            if not first:
                for stack in stripe.stacks:
                    if item.w == stack.w and item.h <= stack.uh+0.001:
                        stack.addItem(item)
                        labels.append(item_id)
                        flag = True
                        break
                
                    
            # 当前bin中的所有stack中装不下,新建stack
            if not first and not flag:
                if item.w <= stripe.uw and item.h <= stripe.h:
                    stack = Stack(stripe,item)
                    stack.addItem(item)
                    stripe.addStack(stack)
                    labels.append(item_id)
                
                elif stripe.uw < mini_width :
                    # 原有stripe装不下任何item将该stripe记录到bin中
                    cur_bin.addStripe(stripe)
                    stripe = Stripe(plate_width)
                    if item.h <= cur_bin.uh:
                        stripe.h = item.h
                        stack = Stack(stripe,item)
                        stack.addItem(item)
                        stripe.addStack(stack)
                        labels.append(item_id)
                        
                    #如果当前bin装不了任何东西了,则新建bin
                    if cur_bin.uh < mini_height:
                        bins.append(cur_bin)
                        bin_num += 1
                        cur_bin = Bin(bin_num,plate_length,plate_width)
                        item_ids = delete(item_ids,labels)
                        break
                        
                        
            
            if len(labels) == 0:
                cur_bin.addStripe(stripe)
                bins.append(cur_bin)
                bin_num += 1
                cur_bin = Bin(bin_num,plate_length,plate_width)
                stripe = Stripe(plate_width)
                item_ids = delete(item_ids,labels)
                break
                

            if i == len(item_ids)-1:
                item_ids = delete(item_ids,labels)
                if len(item_ids) == 0:
                # if len(item_ids) == 0 and cur_bin.stripes:
                    if stripe.stacks:
                        cur_bin.addStripe(stripe)
                    if cur_bin.stripes:
                        bins.append(cur_bin)
                        bin_num += 1
         
    return bin_num,bins
                    
               
                

def view(bin_result,bin_num=0):
    '''
    bin_result: DataFrame
    '''
    fig,ax = plt.subplots(1,1)
    
    for i in range(len(bin_result)):
        x,y = bin_result.loc[i]['产品x坐标'],bin_result.loc[i]['产品y坐标']
        width,height = bin_result.loc[i]['产品x方向长度'],bin_result.loc[i]['产品y方向长度']

        color = np.random.random((1,4))
    
        rect = plt.Rectangle((x,y),width,height,
                             facecolor=color, #rgba
                             edgecolor='red')
        ax.add_patch(rect)
    
    plt.xlim(0,1220)
    plt.ylim(0,2440)
    plt.title('原片{}'.format(bin_num))
    plt.savefig('D:/硕士学习/研数模/B/子问题1图/原片{}.jpg'.format(bin_num))
    plt.close()
    # plt.show()



dataA = pd.read_csv(r'D:/硕士学习/研数模/B/子问题1-数据集A/dataA1.csv',header=0)

# 按item_length 排序
dataA.sort_values(by=['item_length','item_width'],axis=0,ascending=False,inplace=True)
# data = dataA[:12]


dataA.reset_index(inplace=True,drop=True)
data = dataA
data.reset_index(inplace=True,drop=True)
start_time = time.time()
bin_num,bins = packing(data)
print('bin_num:',bin_num)
time_cost = time.time() - start_time
print('time cost :{} seconds'.format(time_cost))

area = sum(data['item_length'] * data['item_width'])
ef = area / (bin_num * plate_width * plate_length)

print('总利用率:',ef)
 
 
 
 
##### 读取文件 #####

def main():
    file_num = list(range(1,5))
    dataA = None
    for f_num in file_num:
        data_file = r'D:/硕士学习/研数模/B/子问题1-数据集A/dataA{}.csv'.format(f_num)
        if dataA is None:
            dataA = pd.read_csv(data_file,header=0)
        else:
            dataAi = pd.read_csv(data_file,header=0)
            
            dataA = pd.concat((dataA,dataAi),axis=0)
    
    
    
    # 按item_length 排序
    dataA.sort_values(by=['item_length','item_width'],axis=0,ascending=False,inplace=True)
    # data = dataA[:12]
    data = dataA
    
    data.reset_index(inplace=True,drop=True)
    start_time = time.time()
    bin_num,bins = packing(data)
    print('bin_num:',bin_num)
    time_cost = time.time() - start_time
    print('time cost :{} seconds'.format(time_cost))
    
    area = sum(data['item_length'] * data['item_width'])
    ef = area / (bin_num * plate_width * plate_length)
    
    print('总利用率:',ef)
    efs = []
    for f_num  in file_num:
        
        data_file = r'D:/硕士学习/研数模/B/子问题1-数据集A/dataA{}.csv'.format(f_num)
        dataA = pd.read_csv(data_file,header=0)
        # 按item_length 排序
        dataA.sort_values(by=['item_length','item_width'],axis=0,ascending=False,inplace=True)
        data = dataA
        data.reset_index(inplace=True,drop=True)
        start_time = time.time()
        bin_num,bins = packing(data)
        time_cost = time.time() - start_time
        print('A{} time cost :{} seconds'.format(f_num,time_cost))
        
        area = sum(data['item_length'] * data['item_width'])
        ef = area / (bin_num * plate_width * plate_length)
        print('A{}利用率:'.format(f_num),ef)
        efs.append(ef)
        
    print('平均利用率:',np.mean(efs))

    # for bin_num,cur_bin in enumerate(bins):
    #     cur_result = cur_bin.output_result()
    #     df = pd.DataFrame(cur_result)
    #     view(df,bin_num)

if __name__ == '__main__':
    main()
    
    
    
    
    

