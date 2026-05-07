"""
灵敏度分析：关键参数波动对运输总费用的影响
- 油耗成本 ±20%
- 路桥费 ±20%
- 人工成本 ±30%
- 车价(折旧) ±15%
- 行驶里程 ±10%
- 双参数联合分析：油耗×路桥费
"""
import numpy as np
import pandas as pd
from src.utils import load_attachment2, VEHICLE_PARAMS, save_results


def calc_total_cost_with_params(orders, fuel_factor=1.0, toll_factor=1.0,
                                labor_factor=1.0, price_factor=1.0,
                                ins_factor=1.0, dist_factor=1.0):
    """在给定参数系数下计算总运输费用"""
    total = 0
    for _, row in orders.iterrows():
        vtype = row['车型']
        p = VEHICLE_PARAMS[vtype]
        days = row['运输天数']
        dist = row['运输距离_km']

        if pd.isna(dist) or dist is None:
            dist = 0

        # 固定成本（受车价和保险影响）
        # 基准固定成本 = 折旧 + 保险+年审
        # 折旧与车价成正比，保险+年审与保险成正比
        fixed = p['fixed_cost']
        # 拆分: 折旧 ≈ 232.36/284.99=81.5%, 保险年审≈18.5%
        dep_ratio = 0.815
        ins_ratio = 0.185
        fixed_adj = fixed * (dep_ratio * price_factor + ins_ratio * ins_factor)

        # 变动成本 = 油耗 + 路桥
        var = p['var_cost_per_km']
        # 拆分: 油耗占比 ≈ 1.1/1.7=64.7%, 路桥≈35.3%
        fuel_ratio = {'4.2M': 0.647, '7.6M': 0.625, '9.6M': 0.633}
        fr = fuel_ratio.get(vtype, 0.63)
        var_adj = var * (fr * fuel_factor + (1 - fr) * toll_factor)

        labor = p['labor_cost'] * labor_factor
        other = p['other_cost']

        adj_dist = dist * dist_factor

        cost = fixed_adj * days + var_adj * adj_dist + (labor + other) * days
        total += cost

    return total


def sensitivity_main(q1_results=None):
    """执行灵敏度分析"""
    print("\n" + "=" * 60)
    print("灵敏度分析")
    print("=" * 60)

    orders = load_attachment2()
    base_cost = q1_results['total_cost'] if q1_results else calc_total_cost_with_params(orders)

    print(f"\n  基准总费用: {base_cost:,.2f} 元")
    print()

    results = {
        'problem': '灵敏度分析',
        'base_cost': round(base_cost, 2),
        'single_factor': {},
        'dual_factor': {},
    }

    # 单因素灵敏度
    params = {
        '油耗成本': {'factor_range': np.arange(0.8, 1.21, 0.05), 'fn_args': {}},
        '路桥费': {'factor_range': np.arange(0.8, 1.21, 0.05), 'fn_args': {}},
        '人工成本': {'factor_range': np.arange(0.7, 1.31, 0.05), 'fn_args': {}},
        '车价(折旧)': {'factor_range': np.arange(0.85, 1.16, 0.025), 'fn_args': {}},
        '行驶里程': {'factor_range': np.arange(0.8, 1.21, 0.05), 'fn_args': {}},
    }

    param_map = {
        '油耗成本': {'fuel_factor': None},
        '路桥费': {'toll_factor': None},
        '人工成本': {'labor_factor': None},
        '车价(折旧)': {'price_factor': None},
        '行驶里程': {'dist_factor': None},
    }

    for pname, pinfo in params.items():
        values = []
        costs = []
        sensitivity_coeffs = []

        for factor in pinfo['factor_range']:
            kwargs = {'fuel_factor': 1.0, 'toll_factor': 1.0, 'labor_factor': 1.0,
                      'price_factor': 1.0, 'ins_factor': 1.0, 'dist_factor': 1.0}

            if pname == '油耗成本':
                kwargs['fuel_factor'] = factor
            elif pname == '路桥费':
                kwargs['toll_factor'] = factor
            elif pname == '人工成本':
                kwargs['labor_factor'] = factor
            elif pname == '车价(折旧)':
                kwargs['price_factor'] = factor
            elif pname == '行驶里程':
                kwargs['dist_factor'] = factor

            cost = calc_total_cost_with_params(orders, **kwargs)
            values.append(round(float(factor), 3))
            costs.append(round(float(cost), 2))

            # 灵敏度系数
            if factor != 1.0:
                delta_cost = (cost - base_cost) / base_cost
                delta_param = factor - 1.0
                sensitivity_coeffs.append(round(float(delta_cost / delta_param), 4))

        # 平均灵敏度系数(绝对值)
        avg_sensitivity = round(float(np.mean(np.abs(sensitivity_coeffs))), 4) if sensitivity_coeffs else 0

        results['single_factor'][pname] = {
            'values': values,
            'costs': costs,
            'sensitivity_coefficient': avg_sensitivity,
            'interpretation': f'参数每变化1%，总费用变化约{avg_sensitivity:.4f}%'
        }

        # 计算影响百分比
        base_v = costs[len(costs) // 2]  # factor=1.0时的值
        min_cost = min(costs)
        max_cost = max(costs)
        impact_pct = (max_cost - min_cost) / base_v * 100

        print(f"  {pname}:")
        print(f"    波动范围: {values[0]:.2f} ~ {values[-1]:.2f}")
        print(f"    费用范围: {min_cost:,.0f} ~ {max_cost:,.0f} 元")
        print(f"    影响幅度: {impact_pct:.1f}%")
        print(f"    灵敏度系数: {avg_sensitivity:.4f}")

    # 双参数联合分析：油耗 × 路桥费
    print(f"\n  双参数联合分析: 油耗 × 路桥费")
    fuel_range = np.arange(0.8, 1.21, 0.1)
    toll_range = np.arange(0.8, 1.21, 0.1)
    joint_costs = []

    for f_factor in fuel_range:
        row_costs = []
        for t_factor in toll_range:
            cost = calc_total_cost_with_params(orders, fuel_factor=f_factor, toll_factor=t_factor)
            row_costs.append(round(float(cost), 2))
        joint_costs.append(row_costs)

    results['dual_factor'] = {
        'fuel_factors': [round(float(x), 2) for x in fuel_range.tolist()],
        'toll_factors': [round(float(x), 2) for x in toll_range.tolist()],
        'costs': joint_costs,
        'description': '油耗和路桥费同时波动时总费用的变化（用于等高线图）'
    }

    print(f"    油耗范围: 0.8~1.2, 路桥费范围: 0.8~1.2")
    print(f"    联合影响矩阵: {len(fuel_range)}×{len(toll_range)}")

    # 保存
    save_results(results, 'sensitivity_results.json')

    print(f"\n  灵敏度分析完成。结果已保存。")
    return results


if __name__ == '__main__':
    sensitivity_main()
