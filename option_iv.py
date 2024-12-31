from pyarrow import fs
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import exchange_calendars as xcals
import time
import sys

xshg = xcals.get_calendar('XSHG')
from datetime import datetime, timedelta

hdfs = fs.HadoopFileSystem("hdfs://ftxz-hadoop/", user='zli')
import warnings
import math

warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt


def is_weekday(date):
    return date.weekday() < 5  # 周一=0, 周日=6  


def is_session(date):
    """检查给定日期是否是开盘日"""
    if xshg.is_session(date) is False:

        return False
    else:

        return True


def get_third_friday_of_month(year, month):
    """获取指定年月中第三个周五的日期，如果不是工作日则顺延至下一个工作日"""
    # 获取该月的第一天  
    first_day_of_month = datetime(year, month, 1)

    # 找到该月第一个周五  
    # 如果1号是周五，则第一个周五就是1号；否则，找到第一个周五  
    if first_day_of_month.weekday() <= 4:  # 周一到周四  
        first_friday = first_day_of_month + timedelta(days=4 - first_day_of_month.weekday())
    else:  # 周五到周日  
        first_friday = first_day_of_month + timedelta(days=11 - first_day_of_month.weekday())

        # 计算第三个周五
    third_friday = first_friday + timedelta(weeks=2)

    # 如果第三个周五不是工作日，则顺延至下一个工作日  
    while not is_session(third_friday):
        third_friday += timedelta(days=1)

    return third_friday.date()


def get_expired_date(time1):
    return get_third_friday_of_month(year=time1.year, month=time1.month)


def get_expiry_time(date_bin, date_end):
    import exchange_calendars as xcals
    import time
    import sys
    xshg = xcals.get_calendar('XSHG')
    date_format = "%Y-%m-%d"
    # print(date_bin.strftime(date_format),date_end.strftime(date_format))
    trading_days = xshg.schedule.loc[date_bin.strftime(date_format):date_end.strftime(date_format)]
    return len(trading_days)


def N_estimate(d):
    if d > 6:
        return 1.0
    if d < -6:
        return 0.0
    b1 = 0.31938153
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    p = 0.2316419
    c2 = 0.3989423

    a = abs(d)
    t = 1.0 / (1.0 + a * p)
    b = c2 * np.exp((-d) * (d * 0.5))
    n = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t
    n = 1.0 - b * n
    if (d < 0):
        n = 1.0 - n
    return n


def n_estimate(d):
    return 1.0 / np.sqrt(2.0 * math.pi) * np.exp(-d * d * 0.5)


def get_d1(S, X, q, r, sigma, t):
    t_sqrt = np.sqrt(t)
    sigma2 = sigma * sigma
    return (np.log(S / X) + (r - q + sigma2 * 0.5) * t) / (t_sqrt * sigma)


def get_option_value(S, X, q, r, sigma, t, put_call):
    t_sqrt = np.sqrt(t)
    sigma2 = sigma * sigma
    d1 = (np.log(S / X) + (r - q + sigma2 * 0.5) * t) / (t_sqrt * sigma)
    d2 = d1 - t_sqrt * sigma
    # print(d1)
    if put_call == 'c' or put_call == 'call':
        return S * np.exp(-q * t) * N_estimate(d1) - X * np.exp(-r * t) * N_estimate(d2)
    elif put_call == 'p' or put_call == 'put':
        return -S * np.exp(-q * t) * N_estimate(-d1) + X * np.exp(-r * t) * N_estimate(-d2)
    else:
        return 0.0


def get_delta(S, X, q, r, sigma, t, put_call):
    t_sqrt = np.sqrt(t)
    sigma2 = sigma * sigma
    d1 = (np.log(S / X) + (r - q + sigma2 * 0.5) * t) / (t_sqrt * sigma)

    if put_call == 'c' or put_call == 'call':
        return np.exp(-q * t) * N_estimate(d1)
    elif put_call == 'p' or put_call == 'put':
        return -np.exp(-q * t) * N_estimate(-d1)
    else:
        return 0.0


def get_gamma(S, X, q, r, sigma, t):
    t_sqrt = np.sqrt(t)
    sigma2 = sigma * sigma
    d1 = (np.log(S / X) + (r - q + sigma2 * 0.5) * t) / (t_sqrt * sigma)
    return np.exp(-q * t) * n_estimate(d1) / S / t_sqrt / sigma


def get_theta(S, X, q, r, sigma, t, put_call):
    t_sqrt = np.sqrt(t)
    sigma2 = sigma * sigma
    d1 = (np.log(S / X) + (r - q + sigma2 * 0.5) * t) / (t_sqrt * sigma)
    d2 = d1 - t_sqrt * sigma

    part1 = S * sigma * np.exp(-q * t) * n_estimate(d1) / 2.0 / t_sqrt
    part2 = -q * S * np.exp(-q * t)
    part3 = r * X * np.exp(-r * t)
    if put_call == 'c' or put_call == 'call':
        return -part1 - part2 * N_estimate(d1) - part3 * N_estimate(d2)
    elif put_call == 'p' or put_call == 'put':
        return -part1 + part2 * N_estimate(-d1) + part3 * N_estimate(-d2)
    else:
        return 0.0


def get_vega(S, X, q, r, sigma, t):
    t_sqrt = np.sqrt(t)
    sigma2 = sigma * sigma
    d1 = (np.log(S / X) + (r - q + sigma2 * 0.5) * t) / (t_sqrt * sigma)
    return S * np.exp(-q * t) * n_estimate(d1) * t_sqrt


def get_rho(S, X, q, r, sigma, t, put_call):
    t_sqrt = np.sqrt(t)
    sigma2 = sigma * sigma
    d1 = (np.log(S / X) + (r - q + sigma2 * 0.5) * t) / (t_sqrt * sigma)
    d2 = d1 - t_sqrt * sigma

    if put_call == 'c' or put_call == 'call':
        return t * X * np.exp(-r * t) * N_estimate(d2)
    elif put_call == 'p' or put_call == 'put':
        return -t * X * np.exp(-r * t) * N_estimate(-d2)
    else:
        return 0.0


def get_implied_vol(S, X, q, r, option_price, t, put_call, accuracy, max_iterations):
    global sigma
    lower_bound = 0
    upper_bound = 1
    upper_price = get_option_value(S, X, q, r, upper_bound, t, put_call)
    while upper_price < option_price:
        lower_bound = upper_bound
        upper_bound = upper_bound * 2.0
        upper_price = get_option_value(S, X, q, r, upper_bound, t, put_call)
    for i in range(max_iterations):
        sigma = 0.5 * (upper_bound + lower_bound)
        price = get_option_value(S, X, q, r, sigma, t, put_call)
        diff = option_price - price
        if abs(diff) < accuracy:
            return sigma
        if diff > 0:
            lower_bound = sigma
        else:
            upper_bound = sigma

    return sigma


from scipy.interpolate import interp1d
from collections import deque
from sortedcontainers import SortedList


class option_Data_stream:
    def __init__(self):
        self.latest_option_data = {}
        self.option_data_yesterday = {}
        self.skew_deque = deque()
        self.skew_sorted_list = SortedList()

    def update_option_data(self, symbol_id, delivery_date, delivery_price, right, price, ask_price, bid_price, time,
                           T2m, stock_price, order_book):
        option_id = (symbol_id, delivery_date, right, delivery_price)

        iv = get_implied_vol(S=stock_price, X=delivery_price, q=0, r=0.02, option_price=price, t=T2m / 250,
                             put_call=right, accuracy=0.001, max_iterations=100)
        # iv_ask=get_implied_vol(S=stock_price,X=delivery_price,q=0,r=0.02,option_price=ask_price,t=T2m/250,put_call=right,accuracy=0.001,max_iterations=100)
        # iv_bid=get_implied_vol(S=stock_price,X=delivery_price,q=0,r=0.02,option_price=bid_price,t=T2m/250,put_call=right,accuracy=0.001,max_iterations=100)
        delta = get_delta(S=stock_price, X=delivery_price, q=0, r=0.02, sigma=iv, t=T2m / 250, put_call=right)
        gamma = get_gamma(S=stock_price, X=delivery_price, q=0, r=0.02, sigma=iv, t=T2m / 250)
        theta = get_theta(S=stock_price, X=delivery_price, q=0, r=0.02, sigma=iv, t=T2m / 250, put_call=right)
        vega = get_vega(S=stock_price, X=delivery_price, q=0, r=0.02, sigma=iv, t=T2m / 250)
        # option_info={'price':price,'time':time,'T2m':T2m,'stock_price':stock_price,'iv':iv,'iv_ask':iv_ask,'iv_bid':iv_bid,'delta':delta,'gamma':gamma,
        #             'theta':theta,'vega':vega,'order_book':order_book}
        option_info = {'price': price, 'time': time, 'T2m': T2m, 'stock_price': stock_price, 'iv': iv, 'delta': delta,
                       'gamma': gamma,
                       'theta': theta, 'vega': vega, 'order_book': order_book}
        self.latest_option_data[option_id] = option_info

    def save_option_data_yesterday(self):
        self.option_data_yesterday = self.latest_option_data

    def get_skew_quantile_deque(self, time):
        skew = self.latest_option_data['skew']['skew']
        self.skew_deque.append((time, skew))
        self.skew_sorted_list.add(skew)
        if len(self.skew_sorted_list) > 4800:
            ts, value_skew = self.skew_deque.popleft()
            self.skew_sorted_list.remove(value_skew)

    # def delete_option(self,time):
    #     for key in self.latest_option_data.keys():
    #         if key=='skew':
    #             continue
    #         if pd.to_datetime(time+8*3600*1e9)>pd.to_datetime(get_expired_date(pd.to_datetime(key[1].astype(int).astype(str),format='%y%m'))):
    #             del self.latest_option_data[key]
    def get_trade_option1(self, type, delta=0.5, p2m=0, right='c', index_price=4000):
        global closet_option
        if type == 'delta':
            min_diff = 1
            for key, value in self.latest_option_data.items():
                if key == 'skew' or key == 'future':
                    continue
                diff = abs(value['delta'] - delta)
                if diff < min_diff:
                    min_diff = diff
                    closet_option = (key, value)

            return closet_option
        if type == 'index':
            min_diff = 100
            for key, value in self.latest_option_data.items():
                if key == 'skew' or key == 'future':
                    continue
                if key[2] != right:
                    continue
                diff = abs(key[3] - index_price)
                if diff < min_diff:
                    min_diff = diff
                    closet_option = (key, value)

            return closet_option
        if type == 'p2m':
            # print('start1')
            # 过滤和排序逻辑  
            filtered_and_sorted = []
            for k, v in self.latest_option_data.items():
                # 检查键是否为元组且包含四个元素  
                if isinstance(k, tuple) and len(k) == 4:
                    # 检查行权价是否高于现货价格  
                    if k[3] >= index_price and k[2] == right:
                        filtered_and_sorted.append((k, v))
                        # print('start2')
            # 对过滤后的列表按行权价进行排序  
            filtered_and_sorted.sort(key=lambda x: x[0][3])
            # print('start3',filtered_and_sorted,len(filtered_and_sorted),p2m)
            # 检查请求的档位数是否有效  
            if filtered_and_sorted == [] or p2m > len(filtered_and_sorted):
                return None
                # print('start4')
            # 返回第n个期权（基于0索引）  
            return filtered_and_sorted[p2m - 1]

    def get_price_diff(self):
        iv_c = self.get_trade_option1(0.25)[1]['iv']
        iv_p = self.get_trade_option1(-0.25)[1]['iv']
        vega_c = self.get_trade_option1(0.25)[1]['vega']
        vega_p = self.get_trade_option1(-0.25)[1]['vega']
        skew = self.latest_option_data['skew']['skew']
        d_skew_quantile = self.latest_option_data['skew']['quantile_70'] - self.latest_option_data['skew'][
            'quantile_30']
        quantile_70 = self.latest_option_data['skew']['quantile_70']
        quantile_30 = self.latest_option_data['skew']['quantile_30']

        d_skew_30 = skew - self.latest_option_data['skew']['quantile_30']
        d_skew_70 = skew - self.latest_option_data['skew']['quantile_70']
        if abs(d_skew_30) > abs(d_skew_70):
            d_skew = d_skew_30
        else:
            d_skew = d_skew_70
        d_price_quantile_short = (vega_c * 2 * iv_p / (1 - quantile_70) ** 2 + vega_p * 2 * quantile_70 * iv_c / (
                    1 + quantile_70) ** 2) * d_skew_quantile
        d_price_quantile_long = (vega_c * 2 * iv_p / (1 - quantile_30) ** 2 + vega_p * 2 * quantile_30 * iv_c / (
                    1 + quantile_30) ** 2) * d_skew_quantile
        d_price_skew = (vega_c * 2 * iv_p / (1 - skew) ** 2 + vega_p * 2 * skew * iv_c / (1 + skew) ** 2) * d_skew
        return (d_price_quantile_short, d_price_quantile_long, d_price_skew)

    def delete_option(self, time):
        to_delete = []
        for key in self.latest_option_data.keys():
            if key == 'skew':
                continue
            expired_date = get_expired_date(pd.to_datetime(key[1].astype(int).astype(str), format='%y%m'))
            if pd.to_datetime(time + 8 * 3600 * 1e9) > pd.to_datetime(expired_date):
                to_delete.append(key)
        for key in to_delete:
            del self.latest_option_data[key]

    def calculate_skew(self, time):

        deltas = []
        ivs = []
        iv_asks = []
        iv_bids = []
        for key, value in self.latest_option_data.items():
            try:
                deltas.append(value['delta'])
            except:
                continue
            ivs.append(value['iv'])
            # iv_asks.append(value['iv_ask']) 
            # iv_bids.append(value['iv_bid']) 

        # 使用插值  

        in1 = interp1d(deltas, ivs, kind='linear', bounds_error=False, fill_value="extrapolate")
        # in2 = interp1d(deltas, iv_asks, kind='linear', bounds_error=False, fill_value="extrapolate") 
        # in3 = interp1d(deltas, iv_bids, kind='linear', bounds_error=False, fill_value="extrapolate") 
        # 计算skew  
        iv_at_025 = in1(0.25)
        iv_at_minus_025 = in1(-0.25)
        skew = (iv_at_025 - iv_at_minus_025) / (iv_at_025 + iv_at_minus_025)
        # skew_ask=(in2(0.25)-in3(-0.25))/(in2(0.25)+in3(-0.25))
        # skew_bid=(in3(0.25)-in2(-0.25))/(in3(0.25)+in2(-0.25))

        quantile = 0.7
        index = int(quantile * len(self.skew_sorted_list))
        index = min(max(index, 0), len(self.skew_sorted_list) - 1)
        # print(index)
        try:
            quantile_70 = self.skew_sorted_list[index]
        except:
            quantile_70 = 0
        quantile = 0.3
        index = int(quantile * len(self.skew_sorted_list))
        index = min(max(index, 0), len(self.skew_sorted_list) - 1)
        try:
            quantile_30 = self.skew_sorted_list[index]
        except:
            quantile_30 = 0
        trade = 0
        if skew > quantile_70:
            trade = -1
        if skew < quantile_30:
            trade = 1
        if len(self.skew_sorted_list) < 240 * 10:
            trade = 0
        # self.latest_option_data['skew']={'time':time,'skew':skew,'skew_ask':skew_ask,'skew_bid':skew_bid,'quantile_70':quantile_70,'quantile_30':quantile_30,'trade':trade}
        self.latest_option_data['skew'] = {'time': time, 'skew': skew, 'quantile_70': quantile_70,
                                           'quantile_30': quantile_30, 'trade': trade}
        return (skew, quantile_70, quantile_30, trade)


import signal
import sys


def signal_handler(sig, frame):
    print('你按下了 Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
date_now = '20240403'
pnl = []
op_stream = option_Data_stream()
import os


def delete_file(file_path):
    """删除指定路径的文件"""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"文件 {file_path} 已删除。")
    else:
        print(f"文件 {file_path} 不存在。")


import subprocess


def uploading_file(path):
    # 定义Hadoop命令  
    hadoop_command = ["hadoop", "fs", "-put", "-f", path, "/user/zli/tyy_d/option_300_data_future"]

    # 设置环境变量（注意：这种方法在subprocess.run中设置环境变量可能不直接有效，需要特殊处理）  
    # 一种方法是在Python脚本外部设置环境变量，或者在调用subprocess之前使用os.environ  
    import os
    os.environ['HADOOP_USER_NAME'] = 'zli'

    # 执行Hadoop命令  
    try:
        result = subprocess.run(hadoop_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Hadoop命令执行成功:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Hadoop命令执行失败:", e.stderr)

        # 注意：这里我们假设Hadoop命令会在标准输出（stdout）中输出有用的信息，
    # 如果出错，则会在标准错误（stderr）中输出错误信息。  
    # 根据你的Hadoop配置和版本，输出可能会有所不同。


def get_vol(args):
    start_date, end_date = args
    for date_now in xshg.schedule.loc[start_date:end_date].index.strftime('%Y%m%d'):
        print(date_now)
        # 读入期权数据
        try:
            f = pq.read_table('/user/zli/lyt/data/OptionData/DepthData/' + date_now, filesystem=hdfs).to_pandas()
        except:
            continue
        f['date_today'] = pd.to_datetime(date_now)
        f['time_datetime'] = pd.to_datetime(f['time']) + pd.Timedelta(
            hours=8)  # .tz_localize(tz)#.apply(lambda x: datetime.datetime.fromtimestamp(x/1e9))
        f.index = f['time_datetime']
        f = f.loc[f['symbol_id'] == 300]
        f['expired_date'] = pd.to_datetime(f['delivery_date'].astype(int).astype(str), format='%y%m').apply(
            get_expired_date)

        # 读入指数数据并处理
        try:
            data_index = pq.read_table('/user/zli/lyt/data/StockIndexFutures/DepthData/' + date_now,
                                       filesystem=hdfs).to_pandas()
        except:
            continue
        month_min = data_index['delivery_code'].min()
        data_index_50 = data_index.loc[(data_index['symbol_id'] == 3693) & (data_index[
                                                                                'delivery_code'] == month_min)]  # [(future['symbol_id']==3693)&(future['delivery_code']==month_min)]
        data_index_50.index = pd.to_datetime(data_index_50['time'] + 8 * 3600 * 1e9)

        # path_stock='/user/zli/lyt/data/StockIndexFutures/DepthData/'+date_now
        # stock=pq.read_table(path_stock, filesystem=hdfs).to_pandas()
        input_times = pd.to_datetime(f['time'] + 8 * 3600 * 1e9)
        try:
            nearest_indices = data_index_50.index.get_indexer(input_times, method='nearest')
        except:
            if not data_index_50.index.is_unique:
                print("索引不是唯一的，需要先处理", date_now)
                # 这里可以添加处理代码，比如重置索引  
                # data_index_50.reset_index(drop=True, inplace=True)
                data_index_50 = data_index_50[~data_index_50.index.duplicated(keep='first')]
                nearest_indices = data_index_50.index.get_indexer(input_times, method='nearest')

        # 使用找到的索引来获取指数价格
        nearest_prices = data_index_50.iloc[nearest_indices]['price'].tolist()
        f.index = input_times
        f['stock_price'] = nearest_prices
        f['right'] = f['is_call'].apply(lambda x: 'c' if x == 1 else 'p')

        # 计算T2m
        def get_expiry_time(date_bin, date_end):
            import exchange_calendars as xcals
            import time
            import sys
            xshg = xcals.get_calendar('XSHG')
            date_format = "%Y-%m-%d"
            # print(date_bin.strftime(date_format),date_end.strftime(date_format))
            trading_days = xshg.schedule.loc[date_bin.strftime(date_format):date_end.strftime(date_format)]
            return len(trading_days)

        # np.datetime_as_string(f['expired_date'].values)
        f['expired_date'] = pd.to_datetime(f['delivery_date'].astype(int).astype(str), format='%y%m').apply(
            get_expired_date)
        # a=f['expired_date'].values
        b = pd.to_datetime(date_now)
        T2m_list = [get_expiry_time(b, i) for i in f['expired_date'].unique()]
        end_time_list = f['expired_date'].unique()
        result_list = [list(item) for item in zip(end_time_list, T2m_list)]

        def get_T2m_day(date_end):
            for item in result_list:
                if item[0] == date_end:
                    return item[1]

        T2m_outday = np.array([get_T2m_day(i) for i in f['expired_date']])
        start_morning = pd.to_datetime(date_now + ' 09:30:00').timestamp() * 1e9
        end_morning = pd.to_datetime(date_now + ' 11:30:00').timestamp() * 1e9
        start_noon = pd.to_datetime(date_now + ' 13:00:00').timestamp() * 1e9
        end_noon = pd.to_datetime(date_now + ' 15:00:00').timestamp() * 1e9
        T2m_day = ((f['time'].values + 8 * 3600 * 1e9 - start_morning) * (
                    f['time'].values + 8 * 3600 * 1e9 - end_morning < 0)
                   + 2 * 3600 * 1e9 * (f['time'].values + 8 * 3600 * 1e9 - end_morning >= 0)
                   + (f['time'].values + 8 * 3600 * 1e9 - start_noon) * (
                               f['time'].values + 8 * 3600 * 1e9 - start_noon > 0)
                   * (f['time'].values + 8 * 3600 * 1e9 - end_noon < 0)
                   + 2 * 3600 * 1e9 * (f['time'].values + 8 * 3600 * 1e9 - end_noon >= 0)) / (4 * 3600 * 1e9)
        T2m = T2m_outday - T2m_day
        f['T2m'] = T2m

        # 把部分pandas转成numpy
        f = f[(f['time'] + 8 * 3600 * 1e9 >= pd.to_datetime(date_now + ' 09:35:00').timestamp() * 1e9) & (
                    f['time'] + 8 * 3600 * 1e9 <= pd.to_datetime(date_now + ' 14:55:00').timestamp() * 1e9)]
        # print(f)
        f1 = f[(f['delivery_date'] == f['delivery_date'].unique().min()) & (f['symbol_id'] == 300)]
        symbol_id_list = f1['symbol_id'].values
        delivery_date_list = f1['delivery_date'].values
        delivery_price_list = f1['delivery_price'].values
        right_list = f1['is_call'].values
        right_list = np.where(right_list == 1, 'c', 'p')
        price_list = f1['price'].values
        ask_price_list = f1['asks1p'].values
        bid_price_list = f1['bids1p'].values
        time_list = f1['time'].values
        time_skew_quantile = time_list[0]
        T2m_list = f1['T2m'].values
        stock_price_list = f1['stock_price'].values
        time_pamda_list = f1.index
        orderbook = {}
        order_list = []
        ask1p = f1['asks1p'].values
        ask1v = f1['asks1v'].values
        ask2p = f1['asks2p'].values
        ask2v = f1['asks2v'].values
        ask3p = f1['asks3p'].values
        ask3v = f1['asks3v'].values
        ask4p = f1['asks4p'].values
        ask4v = f1['asks4v'].values
        ask5p = f1['asks5p'].values
        ask5v = f1['asks5v'].values
        bid1p = f1['bids1p'].values
        bid1v = f1['bids1v'].values
        bid2p = f1['bids2p'].values
        bid2v = f1['bids2v'].values
        bid3p = f1['bids3p'].values
        bid3v = f1['bids3v'].values
        bid4p = f1['bids4p'].values
        bid4v = f1['bids4v'].values
        bid5p = f1['bids5p'].values
        bid5v = f1['bids5v'].values
        skew_list = []
        quantile_30_list = []
        quantile_70_list = []
        trade_list = []
        for i in range(f1.shape[0]):  # f1.shape[0]
            orderbook = {('ask', 1): {'price': ask1p[i], 'volume': ask1v[i]},
                         ('ask', 2): {'price': ask2p[i], 'volume': ask2v[i]},
                         ('ask', 3): {'price': ask3p[i], 'volume': ask3v[i]},
                         ('ask', 4): {'price': ask4p[i], 'volume': ask4v[i]},
                         ('ask', 5): {'price': ask5p[i], 'volume': ask5v[i]},
                         ('bid', 1): {'price': bid1p[i], 'volume': bid1v[i]},
                         ('bid', 2): {'price': bid2p[i], 'volume': bid2v[i]},
                         ('bid', 3): {'price': bid3p[i], 'volume': bid3v[i]},
                         ('bid', 4): {'price': bid4p[i], 'volume': bid4v[i]},
                         ('bid', 5): {'price': bid5p[i], 'volume': bid5v[i]}}
            order_list.append(orderbook)
        iv_list = []
        delta_list = []
        gamma_list = []
        theta_list = []
        vega_list = []
        for i in range(f1.shape[0]):
            stock_price = stock_price_list[i]
            delivery_price = delivery_price_list[i]
            price = price_list[i]
            right = right_list[i]
            T2m = T2m_list[i]

            # 计算iv及希腊字母
            iv = get_implied_vol(S=stock_price, X=delivery_price, q=0, r=0.02, option_price=price, t=T2m / 250,
                                 put_call=right, accuracy=0.001, max_iterations=100)
            # iv_ask=get_implied_vol(S=stock_price,X=delivery_price,q=0,r=0.02,option_price=ask_price,t=T2m/250,put_call=right,accuracy=0.001,max_iterations=100)
            # iv_bid=get_implied_vol(S=stock_price,X=delivery_price,q=0,r=0.02,option_price=bid_price,t=T2m/250,put_call=right,accuracy=0.001,max_iterations=100)
            delta = get_delta(S=stock_price, X=delivery_price, q=0, r=0.02, sigma=iv, t=T2m / 250, put_call=right)
            gamma = get_gamma(S=stock_price, X=delivery_price, q=0, r=0.02, sigma=iv, t=T2m / 250)
            theta = get_theta(S=stock_price, X=delivery_price, q=0, r=0.02, sigma=iv, t=T2m / 250, put_call=right)
            vega = get_vega(S=stock_price, X=delivery_price, q=0, r=0.02, sigma=iv, t=T2m / 250)
            iv_list.append(iv)
            delta_list.append(delta)
            theta_list.append(theta)
            vega_list.append(vega)
            gamma_list.append(gamma)

            if i == 0 and len(op_stream.latest_option_data) > 0:
                op_stream.delete_option(time_list[i])
            op_stream.update_option_data(symbol_id=symbol_id_list[i], delivery_date=delivery_date_list[i],
                                         delivery_price=delivery_price_list[i], right=right_list[i], price=price_list[i]
                                         , ask_price=ask_price_list[i], bid_price=bid_price_list[i], time=time_list[i],
                                         T2m=T2m_list[i], stock_price=stock_price_list[i], order_book=order_list[i])
            skew, quantile_70, quantile_30, trade = op_stream.calculate_skew(time_list[i])
            skew_list.append(skew)
            quantile_30_list.append(quantile_30)
            quantile_70_list.append(quantile_70)
            trade_list.append(trade)
            if i < f1.shape[0] - 2 and time_list[i + 1] > time_list[i] and time_list[i] - time_skew_quantile > 60 * 1e9:
                op_stream.get_skew_quantile_deque(time_list[i])
                time_skew_quantile = time_list[i]
        f1['iv'] = iv_list
        f1['delta'] = delta_list
        f1['theta'] = theta_list
        f1['vega'] = vega_list
        f1['gamma'] = gamma_list
        f1['skew'] = skew_list
        f1['quantile_30'] = quantile_30_list
        f1['quantile_70'] = quantile_70_list
        f1['trade'] = trade_list
        path_option = '/home/tyy/python/0711/data_option/data_option_' + date_now + '.csv'
        f1.to_csv(path_option)
        uploading_file(path_option)
        delete_file(path_option)


params = [['2022-01-01', '2022-02-01'], ['2022-02-01', '2022-03-01'], ['2022-03-01', '2022-04-01'],
          ['2022-05-01', '2022-06-01'], ['2022-06-01', '2022-07-01'],
          ['2022-07-01', '2022-08-01'], ['2022-08-01', '2022-09-01'], ['2022-09-01', '2022-10-01'],
          ['2022-10-01', '2022-11-01'], ['2022-11-01', '2022-12-01'], ['2022-12-01', '2022-12-31'],
          ['2023-01-01', '2023-02-01'], ['2023-02-01', '2023-03-01'], ['2023-03-01', '2023-04-01'],
          ['2023-04-01', '2023-05-01'], ['2023-05-01', '2023-06-01'], ['2023-06-01', '2023-07-01'],
          ['2023-07-01', '2023-08-01'], ['2023-08-01', '2023-09-01'], ['2023-09-01', '2023-10-01'],
          ['2023-10-01', '2023-11-01'], ['2023-11-01', '2023-12-01'], ['2023-12-01', '2023-12-31'],
          ['2024-01-01', '2024-02-01'], ['2024-02-01', '2024-03-01'], ['2024-03-01', '2024-04-01'],
          ['2024-04-01', '2024-05-01'], ['2024-05-01', '2024-05-31']]
# params=[['2022-04-01','2022-04-05'],['2022-04-06','2022-04-10'],['2022-04-11','2022-04-15'],['2022-04-16','2022-04-20'],['2022-04-21','2022-04-25'],['2022-04-26','2022-04-30']]
# params=[['2024-01-20','2024-01-24'],['2024-01-25','2024-01-29'],['2024-01-30','2024-02-01'],['2024-02-02','2024-02-06'],['2024-02-07','2024-02-12'],['2024-02-13','2024-02-16']
#         ,['2024-02-17','2024-02-19']]
from joblib import Parallel, delayed

resu = Parallel(n_jobs=30)(delayed(get_vol)(args) for args in params)
