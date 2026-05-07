"""
问题1：运输成本核算模型
根据附件1的车辆运营成本和附件2的排货信息表，计算当前运输总费用
"""
import numpy as np
import pandas as pd
from src.utils import (load_attachment1, load_attachment2, VEHICLE_PARAMS,
                        VEHICLE_TYPES, save_results)


def calc_order_cost(row, veh_params):
    """
    计算单个订单的运输费用
    Cost_i = C_fixed * days + V_var * dist + (C_labor + C_other) * days
    """
    vtype = row['车型']
    if vtype not in veh_params:
        raise ValueError(f"未知车型: {vtype}")

    p = veh_params[vtype]
    days = row['运输天数']
    dist = row['运输距离_km']

    if pd.isna(dist) or dist is None:
        dist = 0

    fixed_cost = p['fixed_cost'] * days
    var_cost = p['var_cost_per_km'] * dist
    labor_cost = p['labor_cost'] * days
    other_cost = p['other_cost'] * days

    total = fixed_cost + var_cost + labor_cost + other_cost

    return {
        '订单索引': int(row.name) if hasattr(row, 'name') else 0,
        '车型': vtype,
        '收货城市': row.get('城市', '未知'),
        '运输天数': int(days),
        '运输距离_km': float(dist),
        '托盘数': float(row['托盘数']),
        '箱数': float(row['箱数']),
        '固定成本': round(fixed_cost, 2),
        '变动成本': round(var_cost, 2),
        '人工成本': round(labor_cost, 2),
        '其他成本': round(other_cost, 2),
        '总费用': round(total, 2),
    }


def problem1_main():
    """问题1主函数：运输成本核算"""
    print("=" * 60)
    print("问题1：运输成本核算")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载附件2排货信息表...")
    orders = load_attachment2()
    print(f"  有效订单数: {len(orders)}")
    print(f"  车型分布: {orders['车型'].value_counts().to_dict()}")

    veh_params = load_attachment1()

    # 2. 逐单计算费用
    print("\n[2/5] 逐单计算运输费用...")
    cost_details = []
    for idx, row in orders.iterrows():
        detail = calc_order_cost(row, veh_params)
        detail['订单索引'] = int(idx)
        cost_details.append(detail)

    df_costs = pd.DataFrame(cost_details)

    # 3. 汇总
    print("\n[3/5] 汇总运输总费用...")
    total_cost = df_costs['总费用'].sum()

    # 分车型汇总
    by_type = df_costs.groupby('车型').agg(
        订单数=('总费用', 'count'),
        总费用=('总费用', 'sum'),
        平均费用=('总费用', 'mean'),
        总里程=('运输距离_km', 'sum'),
    ).reset_index()

    # 分时效汇总
    by_days = df_costs.groupby('运输天数').agg(
        订单数=('总费用', 'count'),
        总费用=('总费用', 'sum'),
    ).reset_index()

    # 按城市汇总
    by_city = df_costs.groupby('收货城市').agg(
        订单数=('总费用', 'count'),
        总费用=('总费用', 'sum'),
        平均费用=('总费用', 'mean'),
    ).reset_index().sort_values('总费用', ascending=False)

    print(f"\n  运输总费用: {total_cost:,.2f} 元")
    print(f"  订单数: {len(df_costs)}")
    print(f"  平均单次费用: {df_costs['总费用'].mean():,.2f} 元")

    print("\n  分车型汇总:")
    for _, r in by_type.iterrows():
        print(f"    {r['车型']}: {int(r['订单数'])}单, "
              f"总费用 {r['总费用']:,.2f}元, "
              f"平均 {r['平均费用']:,.2f}元/单")

    # 4. 成本结构分析
    print("\n[4/5] 成本结构分析...")
    total_fixed = df_costs['固定成本'].sum()
    total_var = df_costs['变动成本'].sum()
    total_labor = df_costs['人工成本'].sum()
    total_other = df_costs['其他成本'].sum()

    cost_structure = {
        '固定成本': round(total_fixed, 2),
        '变动成本(油耗+路桥)': round(total_var, 2),
        '人工成本': round(total_labor, 2),
        '其他成本': round(total_other, 2),
    }
    print(f"  固定成本: {total_fixed:,.2f} ({total_fixed/total_cost*100:.1f}%)")
    print(f"  变动成本: {total_var:,.2f} ({total_var/total_cost*100:.1f}%)")
    print(f"  人工成本: {total_labor:,.2f} ({total_labor/total_cost*100:.1f}%)")
    print(f"  其他成本: {total_other:,.2f} ({total_other/total_cost*100:.1f}%)")

    # 5. 保存结果
    print("\n[5/5] 保存结果...")
    results = {
        'problem': '问题1 - 运输成本核算',
        'total_cost': round(total_cost, 2),
        'order_count': len(df_costs),
        'avg_cost_per_order': round(float(df_costs['总费用'].mean()), 2),
        'cost_by_type': by_type.to_dict(orient='records'),
        'cost_by_days': by_days.to_dict(orient='records'),
        'cost_by_city': by_city.to_dict(orient='records'),
        'cost_structure': cost_structure,
        'orders': df_costs.sort_values('总费用', ascending=False).to_dict(orient='records'),
        'pallet_stats': {
            'total_pallets': float(orders['托盘数'].sum()),
            'avg_pallets': float(orders['托盘数'].mean()),
            'max_pallets': float(orders['托盘数'].max()),
            'min_pallets': float(orders['托盘数'].min()),
        },
        'veh_usage': {
            '7.6M_orders': int((orders['车型'] == '7.6M').sum()),
            '9.6M_orders': int((orders['车型'] == '9.6M').sum()),
            '4.2M_orders': 0,
        },
        'loading_rate_analysis': {
            '7.6M_avg_pallets': float(orders[orders['车型']=='7.6M']['托盘数'].mean()),
            '9.6M_avg_pallets': float(orders[orders['车型']=='9.6M']['托盘数'].mean()),
            '7.6M_capacity': 11,
            '9.6M_capacity': 14,
            '7.6M_avg_loading_rate': float(orders[orders['车型']=='7.6M']['托盘数'].mean() / 11 * 100),
            '9.6M_avg_loading_rate': float(orders[orders['车型']=='9.6M']['托盘数'].mean() / 14 * 100),
        }
    }

    save_results(results, 'problem1_results.json')
    return results


if __name__ == '__main__':
    problem1_main()
