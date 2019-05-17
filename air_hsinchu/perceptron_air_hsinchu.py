# encoding=utf-8
# Programmer: 劉志俊
# Date: 2018/03/21
# 感知器 Perceptron 以 Python 語言實作
# 類別實作程式碼參考: 
# https://www.zybuluo.com/hanbingtao/note/433855
# 感知器向量運算參考: 
# https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
# Modified by galileoshen 空氣品質預測->Fail

from numpy import dot, random
#熊貓是python的excel
import pandas as pd

df = pd.read_csv("air_hsinchu_clean_utf8.csv")
training_data = df.values

total_errors = []                               # 各世代的訓練誤差值

unit_step = lambda x: 0 if x < 0 else 1         # 神經元激活函數

class Perceptron(object):
    def __init__(self, input_num):              # 感知器建構子
        print('*init')
        self.weights = random.rand(input_num+1) # 啟始權重向量 w
        print(self)
    def __str__(self):                          # 列印感知器訓練結果
        return '權重向量 weights: %s' % (self.weights)
    def predict(self, input_vec):               # 感知器預測
        return unit_step(dot(self.weights, input_vec))
    def train(self, epochs, eta):               # 感知器訓練
        print('*train')
        print('學習速率: %s\n ' % eta)
        for i in range(epochs):
            print('世代: %d' % i)
            errors = 0
            for (x) in training_data:
                print('%s => %f' % (x[:18], x[18]))
                print(self)                          # 印出感知器訓練後的權重向量
                result = dot(self.weights, x[:18])        # result = weight x
                us = unit_step(result)
                error = x[18] - us                # 計算訓練誤差值
                errors += abs(error)
                y = eta * error * x[:18]
                self.weights += y
                print('目標: %f 輸出: %f => %f  誤差: %f' % (x[18], result, us, error))
                print('修正: %s * %s * %f => %s\n' % (x[:18], eta, error, y)) 
            total_errors.append(errors)              # 記錄此世代的訓練誤差值
            print('錯誤數: %f' %(errors))
            print('=========================================================')

p = Perceptron(17)      # 宣告 2 個輸入的 AND-感知器
p.train(2, 0.1)       # 以資料集 training_data 訓練 10 個世代, 學習速率 0.1
print('訓練誤差值變化', total_errors)   # 印出每個世代的訓練誤差值

# 訓練誤差值變化繪圖
from pylab import plot, ylim
ylim([-1,4])
plot(total_errors)
