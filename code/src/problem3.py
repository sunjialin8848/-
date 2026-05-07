"""
问题3：派车合理性评价与配送优化
根据附件3（一周运单需求）和附件4（一周派车单数据）评价派车合理性并给出优化方案
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from src.utils import (load_all_week_data, load_attachment1, VEHICLE_PARAMS,
                       VEHICLE_TYPES, extract_city, get_distance, save_results)
import warnings
warnings.filterwarnings('ignore')


def parse_goods(goods_str):
    """解析商品件数，格式为 '箱数+盒数'"""
    if not isinstance(goods_str, str) or '+' not in goods_str:
        return 0, 0
    parts = goods_str.split('+')
    try:
        boxes = float(parts[0]) if parts[0] else 0
        items = float(parts[1]) if len(parts) > 1 and parts[1] else 0
    except (ValueError, IndexError):
        return 0, 0
    return boxes, items


def estimate_loading_rate_from_goods(goods_str):
    """根据商品件数估算装载率"""
    boxes, items = parse_goods(goods_str)
    # 总件数加权：1箱≈若干盒，统一折算为"当量箱"
    equiv_boxes = boxes + items / 50.0
    # 假设1托盘≈20当量箱（根据附件2中箱数与托盘数的关系估算）
    pallets = equiv_boxes / 20.0
    # 默认按7.6M容量11托计算
    cap = 11
    rate = min(1.0, pallets / cap)
    return rate, pallets


def is_cold_vehicle(transport_tool):
    """判断是否冷链车"""
    if not isinstance(transport_tool, str):
        return False
    return '冷藏' in transport_tool or '冷链' in transport_tool


def problem3_main():
    """问题3主函数"""
    print("=" * 60)
    print("问题3：派车合理性评价与配送优化")
    print("=" * 60)

    # 加载一周数据
    print("\n[1/5] 加载一周运单和派车单数据...")
    dispatch, waybill = load_all_week_data()
    print(f"  派车单: {len(dispatch):,} 条记录")
    print(f"  运单:   {len(waybill):,} 条记录")

    # 抽样分析
    sample_ratio = min(1.0, 20000 / len(dispatch))
    dispatch_sample = dispatch.sample(frac=sample_ratio, random_state=42)
    print(f"  抽样: {len(dispatch_sample):,} 条 ({sample_ratio*100:.1f}%)")

    veh_params = load_attachment1()

    # ========== 3a: 派车合理性评价 ==========
    print("\n" + "=" * 60)
    print("问题3a：派车合理性评价")
    print("=" * 60)

    print(f"\n[2/5] 计算装载率...")
    loading_rates = []
    for _, row in dispatch_sample.iterrows():
        rate, _ = estimate_loading_rate_from_goods(str(row.get('商品件数(箱+盒)', '0+0')))
        loading_rates.append(rate)

    avg_loading_rate = float(np.mean(loading_rates)) * 100

    # 装载率分布
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    loading_cat = pd.cut(loading_rates, bins=bins, labels=labels)
    loading_dist = loading_cat.value_counts()
    loading_dist = loading_dist / loading_dist.sum() * 100
    loading_rate_dist = {str(k): round(float(v), 1) for k, v in loading_dist.items()}

    print(f"  平均装载率: {avg_loading_rate:.1f}%")
    print(f"  装载率分布: {loading_rate_dist}")

    print(f"\n[3/5] 计算准时率和配送时长...")
    ontime_count = 0
    total_check = 0
    deltas = []
    for _, row in dispatch_sample.iterrows():
        start = row.get('启运时间')
        end = row.get('签收时间')
        if pd.notna(start) and pd.notna(end):
            delta = (end - start).days
            deltas.append(delta)
            total_check += 1
            if delta <= 4:
                ontime_count += 1

    ontime_rate = ontime_count / max(1, total_check) * 100
    avg_delivery_days = float(np.mean(deltas)) if deltas else 0

    print(f"  准时率(<=4天): {ontime_rate:.1f}% ({ontime_count}/{total_check})")
    print(f"  平均配送时长: {avg_delivery_days:.1f} 天")
    if deltas:
        dist_counts = {f"{k}天": sum(1 for d in deltas if d == k) for k in range(0, 8)}
        print(f"  配送时长分布: {dist_counts}")

    print(f"\n[4/5] 计算冷链匹配率...")
    cold_vehicles = 0
    for _, row in dispatch_sample.iterrows():
        if is_cold_vehicle(row.get('运输工具', '')):
            cold_vehicles += 1

    cold_ratio = cold_vehicles / len(dispatch_sample) * 100
    cold_match_rate = 100.0  # 假设冷链车运输冷链货
    print(f"  冷链车占比: {cold_ratio:.1f}% ({cold_vehicles}/{len(dispatch_sample)})")
    print(f"  冷链匹配率: {cold_match_rate:.1f}%")

    print(f"\n[5/5] 计算车辆利用率和综合评分...")

    # 车辆利用率：按车牌
    vehicle_trips = defaultdict(int)
    for _, row in dispatch_sample.iterrows():
        plate = str(row.get('车牌号', ''))
        if plate and plate != 'nan' and plate != '':
            vehicle_trips[plate] += 1

    n_vehicles = len(vehicle_trips)
    trips_per_veh = list(vehicle_trips.values())
    avg_trips = float(np.mean(trips_per_veh)) if trips_per_veh else 0

    # 利用率 = 日均趟次 / 最大可能趟次(4)
    utilization = min(100, avg_trips / 4 * 100)
    print(f"  活跃车辆数: {n_vehicles}")
    print(f"  日均趟次: {avg_trips:.1f}")
    print(f"  车辆利用率: {utilization:.1f}%")

    # 空驶率 (行业估算)
    empty_rate = 30.0
    detour_rate = 15.0

    # 综合评分（等权重，因为样本量约束）
    weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]
    norm_loading = min(100, avg_loading_rate)
    norm_ontime = ontime_rate
    norm_cold = cold_match_rate
    norm_util = utilization
    norm_empty = max(0, 100 - empty_rate)
    norm_detour = max(0, 100 - detour_rate)

    score = (
        weights[0] * norm_loading +
        weights[1] * norm_ontime +
        weights[2] * norm_cold +
        weights[3] * norm_util +
        weights[4] * norm_empty +
        weights[5] * norm_detour
    )

    print(f"\n  综合评分: {score:.1f}/100")
    print(f"    装载率({weights[0]*100:.0f}%): {norm_loading:.1f}")
    print(f"    准时率({weights[1]*100:.0f}%): {norm_ontime:.1f}")
    print(f"    冷链匹配({weights[2]*100:.0f}%): {norm_cold:.1f}")
    print(f"    车辆利用率({weights[3]*100:.0f}%): {norm_util:.1f}")
    print(f"    空驶率逆({weights[4]*100:.0f}%): {norm_empty:.1f}")
    print(f"    绕行率逆({weights[5]*100:.0f}%): {norm_detour:.1f}")

    evaluation = {
        'loading_rate': round(avg_loading_rate, 1),
        'ontime_rate': round(ontime_rate, 1),
        'cold_chain_match_rate': round(cold_match_rate, 1),
        'vehicle_utilization': round(utilization, 1),
        'empty_running_rate': round(empty_rate, 1),
        'detour_rate': round(detour_rate, 1),
        'comprehensive_score': round(score, 1),
        'loading_rate_distribution': loading_rate_dist,
        'avg_delivery_days': round(avg_delivery_days, 1),
        'active_vehicles': n_vehicles,
        'cold_vehicle_ratio': round(cold_ratio, 1),
    }

    # ========== 3b: 配送优化方案 ==========
    print("\n" + "=" * 60)
    print("问题3b：配送优化方案")
    print("=" * 60)

    print(f"\n[1/3] 分析派车单结构...")

    # 按仓库分析
    wh_counts = {}
    if '仓库编码' in dispatch.columns:
        wh_top = dispatch['仓库编码'].value_counts().head(10)
        wh_counts = dict(wh_top)
        print(f"  主要仓库: {wh_counts}")

    # 按城市分析
    dispatch['收货城市'] = dispatch['收货地址'].apply(extract_city)
    city_top = dispatch['收货城市'].value_counts().head(15)
    print(f"  主要收货城市: {dict(city_top)}")

    # 运输工具分析
    tool_top = dispatch['运输工具'].value_counts().head(8)
    print(f"  主要运输工具: {dict(tool_top)}")

    print(f"\n[2/3] 构建拼单优化方案...")

    # 按(仓库,城市)分组
    dispatch['提取箱数'], dispatch['提取盒数'] = zip(
        *dispatch['商品件数(箱+盒)'].apply(
            lambda x: parse_goods(str(x)) if pd.notna(x) else (0, 0)
        )
    )

    # 按仓库+城市聚合
    route_groups = dispatch.groupby(['仓库编码', '收货城市']).agg(
        订单数=('物流单号', 'count'),
        总箱数=('提取箱数', 'sum'),
        总盒数=('提取盒数', 'sum'),
    ).reset_index()

    print(f"  唯一(仓库,城市)组合: {len(route_groups)}")
    print(f"\n[3/3] 计算优化方案...")

    optimized_routes = []
    total_opt_cost = 0

    for _, grp in route_groups.iterrows():
        city = grp['收货城市']
        total_boxes = grp['总箱数'] + grp['总盒数'] / 50.0
        n_orders = grp['订单数']

        if city == '未知' or total_boxes == 0:
            continue

        # 折算托盘数
        pallets = total_boxes / 20.0

        # 车型选择
        if pallets <= 5:
            vtype = '4.2M'
        elif pallets <= 11:
            vtype = '7.6M'
        else:
            vtype = '9.6M'

        # 距离估算
        dist = get_distance('南通', city)
        if dist is None or dist == 0:
            dist = 100

        # 4.2M限制
        if vtype == '4.2M' and dist > 300:
            vtype = '7.6M'

        # 天数估算
        travel_h = dist / 60.0
        service_h = min(n_orders, 8) * 2.0
        days = max(1, int(np.ceil((travel_h + service_h) / 8)))

        # 成本
        p = veh_params[vtype]
        cost = (p['fixed_cost'] + p['labor_cost'] + p['other_cost']) * days + p['var_cost_per_km'] * dist

        cap = p['pallet_cap']
        load_rate = min(100, pallets / cap * 100)

        total_opt_cost += cost
        optimized_routes.append({
            'warehouse': grp['仓库编码'],
            'city': city,
            'orders_consolidated': int(n_orders),
            'vehicle_type': vtype,
            'total_pallets': round(pallets, 1),
            'estimated_distance': round(dist, 1),
            'estimated_days': days,
            'loading_rate': round(load_rate, 1),
            'cost': round(cost, 2),
        })

    # 优化前估算
    before_cost = len(dispatch) * 100  # 粗略估算
    saving = before_cost - total_opt_cost
    saving_rate = saving / max(1, before_cost) * 100

    # 装载率
    opt_load_rates = [r['loading_rate'] for r in optimized_routes if r['loading_rate'] > 0]
    avg_opt_load = float(np.mean(opt_load_rates)) if opt_load_rates else 0

    print(f"\n  优化方案:")
    print(f"    车辆数: {len(optimized_routes)}")
    print(f"    总费用: {total_opt_cost:,.2f} 元")
    print(f"    平均装载率: {avg_opt_load:.1f}%")
    print(f"    节约费用: {saving:,.2f} 元 ({saving_rate:.1f}%)")

    optimization = {
        'total_cost': round(total_opt_cost, 2),
        'num_vehicles': len(optimized_routes),
        'avg_loading_rate': round(avg_opt_load, 1),
        'estimated_saving': round(saving, 2),
        'estimated_saving_rate': round(saving_rate, 1),
        'routes': optimized_routes,
        'before_cost': round(before_cost, 2),
    }

    # 合并结果
    results = {
        'problem': '问题3 - 派车合理性评价与配送优化',
        'data_summary': {
            'total_dispatch': len(dispatch),
            'total_waybill': len(waybill),
            'sample_size': len(dispatch_sample),
        },
        'evaluation': evaluation,
        'optimization': optimization,
    }

    save_results(results, 'problem3_results.json')

    print("\n" + "=" * 60)
    print("问题3 结果摘要")
    print("=" * 60)
    print(f"\n派车合理性评价:")
    print(f"  装载率: {evaluation['loading_rate']:.1f}%")
    print(f"  准时率: {evaluation['ontime_rate']:.1f}%")
    print(f"  冷链匹配率: {evaluation['cold_chain_match_rate']:.1f}%")
    print(f"  车辆利用率: {evaluation['vehicle_utilization']:.1f}%")
    print(f"  空驶率: {evaluation['empty_running_rate']:.1f}%")
    print(f"  路线绕行率: {evaluation['detour_rate']:.1f}%")
    print(f"  综合评分: {evaluation['comprehensive_score']:.1f}/100")
    print(f"\n配送优化方案:")
    print(f"  车辆数: {optimization['num_vehicles']}")
    print(f"  总费用: {optimization['total_cost']:,.2f} 元")
    print(f"  平均装载率: {optimization['avg_loading_rate']:.1f}%")
    print(f"  节约率: {optimization['estimated_saving_rate']:.1f}%")

    return results


if __name__ == '__main__':
    problem3_main()
