import numpy as np
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import tkinter as tk
from tkinter import messagebox
class IV():
    def __init__(self):
        # 设置初始WOE的最大值以及最小值,私有化变量
        self._WOE_MIN = -1
        self._WOE_MAX = 1

    # 设置set,get方法获取以及修改相关的WOE的最大值以及最小值
    def set_WOE_MAX(self,woe_max):
        self._WOE_MAX = woe_max
    def set_WOE_MIN(self,woe_min):
        self._WOE_MIN = woe_min
    def get_WOE_MAX(self):
        return self._WOE_MAX
    def get_WOE_MIN(self):
        return self._WOE_MIN

    def simple_binning(self,x):
        #对传入的x值进行简单的分箱,分箱规则为：空值为一箱,剩下不为空的按照取值的数量进行5等分进行分箱
        #由于过程中可能会用到pandas进行数据的处理,因此在使用之前必须将pandas中的series转换成numpy的array
        x_np = np.array(list(x))
        #不建议直接修改源数据,这样会产生很多不必要的麻烦,使用换元法操作换元后的数组
        res = np.array([0]*x_np.shape[-1],dtype=int)
        #将空值部分填充为-99
        res[np.isnan(x_np)] = -99
        #将不为空的部分取出单独计算
        x_np_copy = x_np[np.isnan(x_np)==False]
        #将剩余的部分分成5箱
        for i in range(5):
            point1 = stats.scoreatpercentile(x_np_copy,i*20)
            point2 = stats.scoreatpercentile(x_np_copy,(i+1)*20)
            x1 = x_np[np.where((x_np>=point1)&(x_np<=point2))]
            mask = np.in1d(x_np,x1)
            res[mask] = (i+1)
        return res

    #将字符型参数粗分箱
    def binning_for_char(self,x,y):
        #x,y都是1-D的数组
        a = pd.DataFrame([x,y]).T
        col1 = a.columns
        try:
            a = a.sort_index(by=col1[0])
        except:
            a = a.sort_index()
        a = a.reset_index(drop=True)
        a["re"+col1[0]]=np.nan
        col2 = a.columns
        # print(col2[2])
        a.ix[a[col1[0]].isnull(), col2[2]] = -99
        a_null = a[a[col2[0]].isnull()]
        a_non_nul = a[~a[col2[0]].isnull()]
        index = a_non_nul.index
        index = np.array(list(index))
        for i in range(5):
            point1 = stats.scoreatpercentile(index, i * 20)
            point2 = stats.scoreatpercentile(index, (i + 1) * 20)
            a_non_nul.ix[(index >= point1)&(index <= point2), col2[2]] = (i + 1)
        a = pd.concat([a_non_nul, a_null])
        return a[col2[2]],a[col2[1]]

    def check_y_binary(self,y):
        #现阶段只能处理二分类的Y变量,用该方法检查Y变量是不是二分类的,如果不是将会抛出异常
        #传入的y值必须为1-D的np.array()
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise Exception('Y Value is not binary! ')
        # else:
        #     print('binary OK!')

    def count_binary(self,y,event=1):
        #计算Y值中为1,0的数量
        event_count = (y==1).sum()
        non_event_count = y.shape[-1] - event_count
        return event_count,non_event_count

    # 计算每个变量的iv值以及缺失率
    # x为输入的参数,1-D数组
    # y为1-D数组
    def iv_and_misrate(self,x,y,event=1):
        #检查y值是否为二元
        self.check_y_binary(y)
        # 计算缺失率
        isnull_num = x.isnull().sum()
        all_num = len(x)
        mis_rate = isnull_num*1.0/all_num

        # 计算y中总的0和1的值
        event_total,non_event_total = self.count_binary(y,event=event)

        #将连续型数值进行粗分箱
        if x.dtype=="O":
            if len(x.unique())>20:
                x,y = self.binning_for_char(x,y)
            else:
                x = x.fillna('miss')
        else:
            if type_of_target(x) in ['continuous']:
                x = self.simple_binning(x)
            else:
                if len(x.unique())>10:
                    x = self.simple_binning(x)
                else:
                    x = x.fillna(-99)
        x = np.array(list(x))
        y = np.array(list(y))
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x==x1)[0]]
            event_count,non_event_count = self.count_binary(y1,event=event)
            rate_event = 1.0*event_count/event_total
            rate_non_event = 1.0*non_event_count/non_event_total
            if rate_event==0:
                woe1 = self._WOE_MIN
            elif rate_non_event==0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event/rate_non_event)
            woe_dict[x1] = woe1
            iv = iv + (rate_event-rate_non_event)*woe1
        return mis_rate,iv,woe_dict
    def iv_mis_result(self,data,flagy):
        col=[]
        mis=[]
        iv=[]
        data_type=[]
        for i in data.columns:
            if list(data.columns).index(i)%100==0:
                print(str('finish: %.2f'%((list(data.columns).index(i)/len(data.columns))*100))+'% of iv step')
            try:
                a,b,c = self.iv_and_misrate(data[i],data[flagy])
                col.append(i)
                mis.append(a)
                iv.append(b)
                data_type.append(data[i].dtype)
            except:
                col.append(i)
                mis.append(0)
                iv.append(0)
                data_type.append(data[i].dtype)
        return pd.DataFrame({'col':col,'iv':iv,'mis':mis,'dtype':data_type})

    def get_afterIV_data(self,data,flagy,iv,mis):
        flag_list = [i for i in data.columns if 'flag_' in i[:6]]
        iv_mis = self.iv_mis_result(data,flagy)
        iv_mis = iv_mis[(iv_mis.iv>=iv)& (iv_mis.mis<=mis)]
        iv_mis.sort_values(by='iv',ascending=False,inplace=True)
        iv_mis.to_csv(os.getcwd()+'\\iv_mis_result.csv',index=None)
        col = list(iv_mis.col)
         #建立窗口对象
        window = tk.Tk()
        window.title('my window')
        #设计窗口的大小
        window.geometry('600x600')

        var1 = tk.StringVar()

        l = tk.Label(window, bg='yellow',width=34,textvariable=var1)
        l.pack()

        #定义button的一个功能
        def print_selection():
            value = lb.get(lb.curselection())
            is_change = messagebox.askyesno(title='Sure to delete?',message='you are going to delete '+value+" ,do it now?")
            if is_change:
                try:
                    col.remove(value)
                    var2.set(col)
                except:
                    pass
            var1.set(value)

        var2 = tk.StringVar()
        var2.set(col)
        lb = tk.Listbox(window,listvariable=var2,width=50)
        lb.pack()


        #生成一个button对象
        b1 = tk.Button(window,text='delete selection',
                      width=15,height=2,command=print_selection)
        b1.pack()

        window.mainloop()
        
        final_list = list(set(col+[flagy]+flag_list))
        return data[final_list]