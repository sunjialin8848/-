"""
工具模块：数据加载、距离矩阵、公共工具函数
华东杯 B题 医药物流安排问题
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, os, re

# ========== 城市间公路距离字典（南通到各城市） ==========
# 发货地固定为南通市崇川区
NT_TO_CITY_KM = {
    '太原': 1050, '杭州': 320, '沈阳': 1550, '吉林': 1750,
    '长春': 1700, '天津': 850, '济南': 580, '北京': 950,
    '重庆': 1500, '武汉': 720, '正定': 980, '呼和浩特': 1500,
    '乌鲁木齐': 3500, '上海': 120, '石家庄': 980,
}

# 城市间距离矩阵（用于问题2路径规划，仅包括附件2中出现的目的地城市）
CITY_LIST = ['南通', '太原', '杭州', '沈阳', '吉林', '长春', '天津',
             '济南', '北京', '重庆', '武汉', '正定', '呼和浩特', '乌鲁木齐', '上海', '石家庄']

# 城市间公路距离（单位：km）——基于主要高速公路的近似值
# 索引与 CITY_LIST 一致
CITY_DIST_KM = np.array([
    #   南通  太原  杭州  沈阳  吉林  长春  天津  济南  北京  重庆  武汉  正定 呼和  乌市  上海  石家庄
    [   0, 1050,  320, 1550, 1750, 1700,  850,  580,  950, 1500,  720,  980, 1500, 3500,  120,  980],  # 南通
    [1050,    0, 1200, 1200, 1300, 1200,  500,  500,  500, 1100,  800,  100,  500, 2500, 1200,  100],  # 太原
    [ 320, 1200,    0, 1400, 1600, 1500,  800,  600, 1100, 1400,  700, 1100, 1600, 3500,  150, 1100],  # 杭州
    [1550, 1200, 1400,    0,  300,  300,  700,  900,  700, 2500, 1900, 1100, 1200, 3500, 1600, 1100],  # 沈阳
    [1750, 1300, 1600,  300,    0,  100,  900, 1100,  900, 2600, 2000, 1200, 1300, 3500, 1800, 1200],  # 吉林
    [1700, 1200, 1500,  300,  100,    0,  900, 1100,  800, 2600, 2000, 1200, 1300, 3500, 1800, 1200],  # 长春
    [ 850,  500,  800,  700,  900,  900,    0,  300,  100, 1600, 1000,  300,  600, 3000,  800,  300],  # 天津
    [ 580,  500,  600,  900, 1100, 1100,  300,    0,  400, 1300,  700,  400,  800, 3200,  600,  400],  # 济南
    [ 950,  500, 1100,  700,  900,  800,  100,  400,    0, 1600, 1100,  250,  500, 3000, 1000,  250],  # 北京
    [1500, 1100, 1400, 2500, 2600, 2600, 1600, 1300, 1600,    0,  800, 1200, 1200, 2500, 1600, 1200],  # 重庆
    [ 720,  800,  700, 1900, 2000, 2000, 1000,  700, 1100,  800,    0,  800, 1200, 3000,  700,  800],  # 武汉
    [ 980,  100, 1100, 1100, 1200, 1200,  300,  400,  250, 1200,  800,    0,  500, 2600, 1100,   50],  # 正定
    [1500,  500, 1600, 1200, 1300, 1300,  600,  800,  500, 1200, 1200,  500,    0, 2400, 1800,  500],  # 呼和浩特
    [3500, 2500, 3500, 3500, 3500, 3500, 3000, 3200, 3000, 2500, 3000, 2600, 2400,    0, 3500, 2600],  # 乌鲁木齐
    [ 120, 1200,  150, 1600, 1800, 1800,  800,  600, 1000, 1600,  700, 1100, 1800, 3500,    0, 1100],  # 上海
    [ 980,  100, 1100, 1100, 1200, 1200,  300,  400,  250, 1200,  800,   50,  500, 2600, 1100,    0],  # 石家庄
])

# ========== 车型参数 ==========
VEHICLE_PARAMS = {
    '4.2M': {
        'fixed_cost': 284.99,    # 日均固定成本(折旧+保险+年审)
        'var_cost_per_km': 1.70, # 每公里变动成本(油耗+路桥)
        'labor_cost': 600.00,    # 日均人工成本(2人)
        'other_cost': 460.00,    # 日均其他成本(胎耗+维保+违章)
        'pallet_cap': 5,         # 托盘数容量
        'vol_cap': 12.2,         # 容积(m³)
        'max_range': 300,        # 建议最大单程(km)
    },
    '7.6M': {
        'fixed_cost': 412.95,
        'var_cost_per_km': 2.40,
        'labor_cost': 600.00,
        'other_cost': 533.33,
        'pallet_cap': 11,
        'vol_cap': 22.0,
        'max_range': 3500,
    },
    '9.6M': {
        'fixed_cost': 505.74,
        'var_cost_per_km': 3.00,
        'labor_cost': 600.00,
        'other_cost': 533.33,
        'pallet_cap': 14,
        'vol_cap': 26.0,
        'max_range': 4500,
    }
}

VEHICLE_TYPES = ['4.2M', '7.6M', '9.6M']


def extract_city(address):
    """从收货地址中提取城市名"""
    if not isinstance(address, str):
        return '未知'
    # 优先匹配已知城市
    for city in ['乌鲁木齐', '呼和浩特', '石家庄', '正定', '太原', '杭州', '沈阳',
                 '吉林', '长春', '天津', '济南', '北京', '重庆', '武汉', '上海']:
        if city in address:
            return city
    # 尝试匹配'市'字前的两个字符
    m = re.search(r'([^\s]+市)', address)
    if m:
        return m.group(1)[:-1]
    return '未知'


def get_distance(city_from, city_to):
    """获取两城市间的公路距离"""
    try:
        idx_from = CITY_LIST.index(city_from)
        idx_to = CITY_LIST.index(city_to)
        return float(CITY_DIST_KM[idx_from][idx_to])
    except (ValueError, IndexError):
        return None


def get_distance_from_nt(city):
    """获取南通到指定城市的距离"""
    return NT_TO_CITY_KM.get(city, None)


def calc_transport_days(pickup_date, arrive_date):
    """计算运输天数"""
    if pd.isna(pickup_date) or pd.isna(arrive_date):
        return 1
    diff = (arrive_date - pickup_date).days
    return max(1, diff + 1)


def load_attachment1():
    """加载附件1：各型号车辆的成本和托数"""
    # 使用预定义的VEHICLE_PARAMS（核心参数从附件1提取后计算得到）
    return VEHICLE_PARAMS.copy()


def load_attachment2(filepath='user_data/附件2.排货信息表.xlsx'):
    """
    加载附件2：排货信息表
    返回清洗后的DataFrame
    """
    df = pd.read_excel(filepath, header=None, skiprows=2)
    df.columns = ['月份', '填写日期', '随货通行单号', '温度计编号', '运输交接单号',
                   '盒数', '箱数', '托装', '托盘数', '收货方地址', '运输时效',
                   '预计提货日期', '预计到货日期', '车型', '运输方式',
                   '车辆数目', '始发地天气', '目的地天气', '实际提货日期',
                   '实际到货日期', '签收状态', '随货单据是否齐全', '备注']
    # 删除全空行
    df = df.dropna(subset=['填写日期'], how='all').reset_index(drop=True)

    # 类型转换
    df['填写日期'] = pd.to_datetime(df['填写日期'], errors='coerce')
    df['预计提货日期'] = pd.to_datetime(df['预计提货日期'], errors='coerce')
    df['预计到货日期'] = pd.to_datetime(df['预计到货日期'], errors='coerce')
    df['盒数'] = pd.to_numeric(df['盒数'], errors='coerce').fillna(0)
    df['箱数'] = pd.to_numeric(df['箱数'], errors='coerce').fillna(0)
    df['托盘数'] = pd.to_numeric(df['托盘数'], errors='coerce').fillna(0)
    df['运输时效'] = pd.to_numeric(df['运输时效'], errors='coerce').fillna(2)

    # 标准化车型
    df['车型'] = df['车型'].astype(str).str.strip().str.upper()
    # 修复车型格式：如 '7.6M' -> '7.6M'
    df['车型'] = df['车型'].apply(lambda x: x if x in VEHICLE_TYPES else
                                  ('7.6M' if '7.6' in x else
                                   ('9.6M' if '9.6' in x else
                                    ('4.2M' if '4.2' in x else '7.6M'))))

    # 提取城市
    df['城市'] = df['收货方地址'].apply(extract_city)
    df['运输距离_km'] = df['城市'].apply(get_distance_from_nt)
    df['运输天数'] = df.apply(
        lambda r: calc_transport_days(r['预计提货日期'], r['预计到货日期']), axis=1)

    return df


def load_week_dispatch(date_str):
    """加载指定日期的一周派车单数据"""
    filepath = f'user_data/一周派车单数据（{date_str}）.xlsx'
    df = pd.read_excel(filepath)
    # 去掉首列无名列
    if '   ' in df.columns:
        df = df.drop(columns=['   '])
    df['启运时间'] = pd.to_datetime(df['启运时间'], errors='coerce')
    df['签收时间'] = pd.to_datetime(df['签收时间'], errors='coerce')
    return df


def load_week_waybill(date_str):
    """加载指定日期的一周运单数据"""
    filepath = f'user_data/一周运单数据（{date_str}）.xlsx'
    df = pd.read_excel(filepath)
    if '   ' in df.columns:
        df = df.drop(columns=['   '])
    return df


def load_all_week_data():
    """加载全部7天的派车单和运单数据"""
    date_ranges = [
        '20181210-20181211', '20181211-20181212', '20181212-20181213',
        '20181213-20181214', '20181214-20181215', '20181215-20181216',
        '20181216-20181217'
    ]
    all_dispatch = []
    all_waybill = []
    for dr in date_ranges:
        try:
            dd = load_week_dispatch(dr)
            dd['数据日期'] = dr
            all_dispatch.append(dd)
        except Exception as e:
            print(f"  加载派车单 {dr} 失败: {e}")
        try:
            wb = load_week_waybill(dr)
            wb['数据日期'] = dr
            all_waybill.append(wb)
        except Exception as e:
            print(f"  加载运单 {dr} 失败: {e}")

    dispatch = pd.concat(all_dispatch, ignore_index=True) if all_dispatch else pd.DataFrame()
    waybill = pd.concat(all_waybill, ignore_index=True) if all_waybill else pd.DataFrame()
    return dispatch, waybill


def save_results(data, filename):
    """保存结果到 figures/ 目录"""
    os.makedirs('figures', exist_ok=True)
    filepath = os.path.join('figures', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✅ 结果已保存: {filepath} ({os.path.getsize(filepath)} bytes)")
    return filepath
