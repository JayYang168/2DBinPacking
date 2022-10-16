# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:02:34 2022

@author: yy
"""
import copy   #深拷贝不改变原对象
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from packing import packing,view
file_num = list(range(1,5))
dataA = None
for f_num in file_num:
    data_file = r'D:/硕士学习/研数模/B/子问题1-数据集A/dataA{}.csv'.format(f_num)
    if dataA is None:
        dataA = pd.read_csv(data_file,header=0)
    else:
        dataAi = pd.read_csv(data_file,header=0)
        
        dataA = pd.concat((dataA,dataAi),axis=0)


plate_length = 2440
plate_width = 1220
plate_area = plate_length * plate_width
# 按item_length 排序
dataA.sort_values(by=['item_length','item_width'],axis=0,ascending=False,inplace=True)
# data = dataA[:12]
data = dataA

data.reset_index(inplace=True,drop=True)
start_time = time.time()
bin_num,bins = packing(data)
# 确保dataA中所有长大于宽
# length = []
# width = []
# for i in range(len(dataA)):
#     length.append(max(dataA['item_length'][i],dataA['item_width'][i]))
#     width.append(min(dataA['item_length'][i],dataA['item_width'][i]))

# dataA['item_length'] = length
# dataA['item_width'] = width




best_bins = bins

def bin_to_df(best_bins):
    df_result = None
    for bin_num,cur_bin in enumerate(best_bins):
        cur_result = cur_bin.output_result()
        cur_result = pd.DataFrame(cur_result)
        if df_result is None:
            df_result = cur_result
        else:
            df_result = pd.concat((df_result,cur_result),axis=0)
    return df_result
    # view(cur_result,bin_num)
df_result = bin_to_df(best_bins)
df_result.to_csv(r'D:/硕士学习/研数模/B/子问题1-数据集A/dataA_test.csv',encoding='utf-8-sig')

### 编码
'''
[0,1,2,3,....] #放入顺序
[0,1,0,1,....] #item 长宽0对应保持原状，1对应交换长宽
'''
pop_size = 20   #种群规模
N = len(dataA)   
list_seq = np.arange(0,N,1)


def rand01():
    direction = np.random.random(N)
    direction[direction > 0.5] = 1  #交换长宽
    direction[direction <= 0.5] = 0 #保持长宽
    return direction

direction = rand01()




def initial(list_seq):

    chromosomes = [[list_seq,[0] * len(list_seq)]] 
    for i in range(pop_size-1):
        np.random.shuffle(list_seq)
        chromosome = [list_seq,rand01()] #生成染色体
        chromosomes.append(chromosome)

    chromosomes = np.array(chromosomes)
    
    return chromosomes


chromosomes = initial(list_seq)
#sum(list(chromosomes[0][0]) == list_seq)


# 解码
def decoder(chromosome,dataA):
    '''
    df  DataFrame
    [0,1,2,3,....] #放入顺序
    [0,1,0,1,....] #item 长宽0对应保持原状，1对应交换长宽
    '''
    # 交换顺序
    df = copy.copy(dataA)
    width = np.array(df['item_width'])
    tmpw = copy.copy(width)
    length = np.array(df['item_length'])
    # lenth_less_than_plate_width = length <= 1220 
    #需要交换长宽的索引,#只有item长比原片宽短的才能被交换
    idx = (chromosome[1]==1) * (length <= 1220)
    
    
    
    width[idx] = length[idx] 
    length[idx] = tmpw[idx]
    df['item_width'] = width
    df['item_length'] = length
    return df 






# 选择算子
def selection(chromosomes):
    # 选择父代
    parents = []
    for i in range(pop_size):
        if np.random.random() < (1 - i/pop_size):
            parents.append(chromosomes[i])
    return np.array(parents)


# 交叉算子


def crossover(parent1,parent2):
    # 交叉繁殖(只交换顺序)
    child = copy.copy(parent1)
    locs = sorted(np.random.choice(list_seq,2)) #随机抽取两个数的位置保持不变
    cross_points = child[0,locs]
    k = 0
    for i in range(N):
        if i in locs:
            continue
        
        for j in range(k,N):
            if parent2[0,j] in cross_points:
                continue
            else:
                child[0,i] = parent2[0,j]
                child[1,i] = parent2[1,j]
                k = j+1
                break
    assert sum(child[0]) == sum(parent1[0]),'交叉错误'
    
    
    return child

# child = crossover(parent1,parent2)




def mutation(chromosomes,mutation_rate=0.5):
    '''两点变异
    '''
    for chromosome in chromosomes:
        if np.random.random() < mutation_rate:
            index = sorted(np.random.choice(list_seq,2))
            chromosome[0][index[0]:index[1]] = chromosome[0][index[0]:index[1]][::-1]
            chromosome[1][index[0]:index[1]] = chromosome[1][index[0]:index[1]][::-1]
    return chromosomes


    
    
def fitness(chromosome,dataA):
    change_df = decoder(chromosome,dataA)
    
    bin_num,bins = packing(change_df)
    df_result = bin_to_df(bins)

    return bin_num,bins



def get_best_current(chromosomes,dataA):
    '''按bin数量升序排列，并返回当前最佳利用率和方案及其排序后的
        df DataFrame
    '''
    
    bin_nums_record = np.zeros(len(chromosomes)) 
    bins_record = []
    for i,chromosome in enumerate(chromosomes):
        bin_num,bins = fitness(chromosome,dataA)
        bin_nums_record[i] = bin_num
        bins_record.append(bins)

    idxs = np.argsort(bin_nums_record) #获取升序排列的索引
    best_bin_num = bin_nums_record[idxs[0]]
    best_bins = bins_record[idxs[0]]
    chromosomes = chromosomes[idxs]
    
    return best_bin_num,best_bins,chromosomes




def main():

    iter_count = 20
    
    record_num = None
    

    for i in range(iter_count):
        start_time = time.time()
        if record_num is None:
            chromosomes = initial(list_seq)
    
            best_num,best_bins,chromosomes = get_best_current(chromosomes,dataA.copy())
            best_chromosome = chromosomes[0]
            record_num = [best_num]
            time_cost = time.time() - start_time
            print('初代种群消耗时间{}'.format(time_cost))
        print(best_num)
        # 选择繁殖种群
        parents = selection(chromosomes)
        # 交叉繁殖
        target_child_num = pop_size - len(parents)
        childs = []
        pa_ids = list(range(len(parents)))
        while len(childs) < target_child_num:
            parentids = np.random.choice(pa_ids,2)
            parent1,parent2 = parents[parentids[0]],parents[parentids[1]]
            child = crossover(parent1, parent2)
            childs.append(child)
            
        childs = np.array(childs)
        chromosomes = np.vstack((parents,childs))
        # 个体变异
        chromosomes = mutation(chromosomes)
        
        
        current_best_num,current_best_bins,chromosomes = get_best_current(chromosomes,dataA.copy())
        if current_best_num <= best_num:
            best_bins = current_best_bins
            best_num = current_best_num
            best_chromosome = chromosomes[0]
            
        time_cost = time.time() - start_time
        print('第{}代种群消耗时间{}'.format(i+1,time_cost))
        record_num.append(current_best_num)
    
    # 记录最优解
    df_result = None
    for bin_num,cur_bin in enumerate(best_bins):
        cur_result = cur_bin.output_result()
        cur_result = pd.DataFrame(cur_result)
        if df_result is None:
            df_result = cur_result
        else:
            df_result = pd.concat((df_result,cur_result),axis=0)
        # view(cur_result,bin_num)
    df_result.to_csv(r'D:/硕士学习/研数模/B/子问题1-数据集A/dataA_GA.csv',encoding='utf-8-sig',index=None)
    ef = sum(dataA['item_length'] * dataA['item_width']) / (best_num * plate_area )
    print('最优利用率:',ef)
    return record_num,best_bins

# best_num,best_bins,chromosomes = get_best_current(chromosomes,dataA)

if __name__ == '__main__':
    main()



    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    











