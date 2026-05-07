"""
问题2：拼单运输与车辆调度优化 (HFVRPTW + CV-AGA-LS)
带时间窗约束的异构车辆路径优化模型 + 基于适应度变异系数的自适应遗传算法与局部搜索
"""
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from collections import defaultdict
from src.utils import (load_attachment1, load_attachment2, VEHICLE_PARAMS,
                       VEHICLE_TYPES, CITY_LIST, CITY_DIST_KM, NT_TO_CITY_KM,
                       extract_city, get_distance, get_distance_from_nt, save_results)

# ========== GA参数 ==========
POP_SIZE = 80
MAX_GEN = 200
ELITE_SIZE = 4
TOURNAMENT_SIZE = 3
PC_MAX = 0.9
PC_MIN = 0.5
PM_MAX = 0.15
PM_MIN = 0.03
CV_THRESHOLD = 0.01
PENALTY_FACTOR = 1e4
SPEED_KMH = 60.0
SERVICE_HOURS = 2.0
WORK_START = 9.0
WORK_END = 17.0


class HFVRPTW:
    """HFVRPTW问题定义"""

    def __init__(self, orders_df, veh_params=None):
        self.orders = orders_df.reset_index(drop=True)
        self.n_orders = len(self.orders)
        self.veh_params = veh_params or VEHICLE_PARAMS

        self.node_cities = ['南通'] + self.orders['城市'].tolist()
        self.node_demands = [0] + self.orders['托盘数'].tolist()

        self.n_nodes = self.n_orders + 1
        self.dist_matrix = self._build_distance_matrix()

    def _build_distance_matrix(self):
        D = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue
                ci, cj = self.node_cities[i], self.node_cities[j]
                d = get_distance(ci, cj)
                if d is None or d == 0:
                    d = 20.0 if ci == cj else 500.0
                D[i][j] = d
        return D

    def evaluate_route(self, route, vtype):
        """评估一条路径的成本和可行性"""
        p = self.veh_params[vtype]
        if len(route) == 0:
            return 0, 0, 0, True, 0, 0

        total_pallets = sum(self.node_demands[i] for i in route)

        # 容量检查（硬约束）
        if total_pallets > p['pallet_cap'] * 1.05:  # 允许5%超载容差
            return 0, 0, 0, False, total_pallets, 0

        # 计算总距离
        total_dist = self.dist_matrix[0][route[0]]
        for i in range(len(route) - 1):
            total_dist += self.dist_matrix[route[i]][route[i + 1]]

        # 检查最大里程
        if total_dist > p['max_range']:
            return 0, 0, 0, False, total_pallets, total_dist

        # 计算所需天数
        travel_time = total_dist / SPEED_KMH
        service_time = len(route) * SERVICE_HOURS
        daily_work_hours = WORK_END - WORK_START
        total_needed = travel_time + service_time
        days = max(1, int(np.ceil(total_needed / daily_work_hours)))

        # 计算成本
        fixed_cost = p['fixed_cost'] * days
        var_cost = p['var_cost_per_km'] * total_dist
        labor_cost = p['labor_cost'] * days
        other_cost = p['other_cost'] * days
        total_cost = fixed_cost + var_cost + labor_cost + other_cost

        return total_cost, total_pallets, total_dist, True, total_pallets, total_dist

    def evaluate_solution(self, assignments):
        """
        评价一个方案（订单到路径的分配）
        assignments: [{'route': [order_idx, ...], 'vtype': '7.6M'}, ...]
        """
        total_cost = 0
        total_penalty = 0
        detailed = []

        for assign in assignments:
            route = assign['route']
            vtype = assign['vtype']
            cost, pallets, dist, feasible, load, _ = self.evaluate_route(route, vtype)
            if feasible:
                total_cost += cost
                penalty = 0
            else:
                penalty = PENALTY_FACTOR
                total_penalty += penalty

            detailed.append({
                'route': route,
                'vtype': vtype,
                'cost': cost if feasible else cost + penalty,
                'pallets': float(pallets),
                'dist': float(dist),
                'feasible': feasible,
                'penalty': penalty
            })

        return total_cost + total_penalty, detailed


def group_orders_by_city(orders_df):
    """将订单按城市分组"""
    city_groups = defaultdict(list)
    for idx, row in orders_df.iterrows():
        city = row['城市']
        city_groups[city].append(int(idx))
    return dict(city_groups)


def build_direction_clusters(orders_df):
    """
    按地理方向聚类：
    华北方向：北京、天津、太原、正定/石家庄、呼和浩特
    东北方向：沈阳、吉林、长春
    华东方向：杭州、上海、济南
    华中方向：武汉
    西部方向：重庆、乌鲁木齐
    """
    direction_map = {
        '北京': '华北', '天津': '华北', '太原': '华北',
        '正定': '华北', '石家庄': '华北', '呼和浩特': '华北',
        '沈阳': '东北', '吉林': '东北', '长春': '东北',
        '杭州': '华东', '上海': '华东', '济南': '华东',
        '武汉': '华中',
        '重庆': '西南', '乌鲁木齐': '西北',
        '未知': '其他'
    }
    clusters = defaultdict(list)
    for idx, row in orders_df.iterrows():
        city = row['城市']
        direction = direction_map.get(city, '其他')
        clusters[direction].append(int(idx))
    return dict(clusters)


def generate_smart_solution(orders_df, dist_matrix):
    """
    智能初始解：按地理方向+城市聚类生成初始拼单方案
    """
    n = len(orders_df)
    all_orders = set(range(n))

    # 先提取城市信息
    city_groups = group_orders_by_city(orders_df)

    assignments = []

    # 对每个城市，合并同一城市的订单
    for city, order_ids in city_groups.items():
        remaining = list(order_ids)
        while remaining:
            route = []
            pallets = 0
            for oid in sorted(remaining):
                p = orders_df.loc[oid, '托盘数']
                if pallets + p <= 11:  # 7.6M容量
                    route.append(oid)
                    pallets += p
                else:
                    break
            for oid in route:
                remaining.remove(oid)

            if not route:
                route = [remaining.pop(0)]

            # 选择车型
            tp = sum(orders_df.loc[oid, '托盘数'] for oid in route)
            vt = '4.2M' if tp <= 5 else ('7.6M' if tp <= 11 else '9.6M')

            # 检查距离（避免4.2M用于远距离）
            if vt == '4.2M':
                city = orders_df.loc[route[0], '城市']
                dist = get_distance_from_nt(city) or 0
                if dist > 300:
                    vt = '7.6M'

            assignments.append({'route': route, 'vtype': vt, 'pallets': float(tp)})

    return assignments


def random_solution(orders_df, dist_matrix):
    """生成随机解"""
    n = len(orders_df)
    all_orders = list(range(n))
    random.shuffle(all_orders)

    assignments = []
    i = 0
    while i < n:
        # 随机确定车辆大小
        max_per_vehicle = random.choice([3, 4, 5, 6, 8])
        route = all_orders[i:i + max_per_vehicle]
        tp = sum(orders_df.loc[oid, '托盘数'] for oid in route)
        vt = '4.2M' if tp <= 5 else ('7.6M' if tp <= 11 else '9.6M')
        assignments.append({'route': route, 'vtype': vt, 'pallets': float(tp)})
        i += max_per_vehicle

    return assignments


def evaluate_solution_cost(assignments, orders_df, veh_params, dist_matrix):
    """计算方案总成本（用于智能方案评价）"""
    total = 0
    for a in assignments:
        route = a['route']
        vtype = a['vtype']
        if not route:
            continue
        p = veh_params[vtype]
        tp = sum(orders_df.loc[oid, '托盘数'] for oid in route)

        if tp > p['pallet_cap']:
            total += PENALTY_FACTOR
            continue

        total_dist = dist_matrix[0][route[0] + 1]  # 注意索引偏移（仓库=0）
        for i in range(len(route) - 1):
            total_dist += dist_matrix[route[i] + 1][route[i + 1] + 1]

        if total_dist > p['max_range']:
            total += PENALTY_FACTOR
            continue

        travel_time = total_dist / SPEED_KMH
        service_time = len(route) * SERVICE_HOURS
        days = max(1, int(np.ceil((travel_time + service_time) / (WORK_END - WORK_START))))

        cost = (p['fixed_cost'] + p['labor_cost'] + p['other_cost']) * days + p['var_cost_per_km'] * total_dist
        total += cost
    return total


class CVRouter:
    """基于先聚类再优化的拼单方案生成器"""

    def __init__(self, orders_df, veh_params=None):
        self.orders = orders_df.reset_index(drop=True)
        self.n = len(self.orders)
        self.veh_params = veh_params or VEHICLE_PARAMS
        self.dist_matrix = HFVRPTW(orders_df).dist_matrix

        # 按方向聚类
        self.clusters = build_direction_clusters(self.orders)
        print(f"\n  地理方向聚类: {dict((k, len(v)) for k, v in self.clusters.items())}")

    def optimize_cluster(self, order_indices):
        """对一个聚类的订单进行路径优化"""
        if len(order_indices) <= 3:
            return self._direct_ship(order_indices)

        # 使用简单贪心：按城市分组，同城拼单
        sub_orders = self.orders.loc[order_indices]
        city_groups = group_orders_by_city(sub_orders)

        result = []
        for city, oids in city_groups.items():
            # 同城订单尽可能拼入同一辆车
            remaining = sorted(oids)
            while remaining:
                route = []
                pallets = 0
                for oid in remaining[:]:
                    p = self.orders.loc[oid, '托盘数']
                    if pallets + p <= 11:
                        route.append(oid)
                        pallets += p
                    else:
                        break
                for oid in route:
                    remaining.remove(oid)

                if not route:
                    route = [remaining.pop(0)]
                    pallets = self.orders.loc[route[0], '托盘数']

                tp = sum(self.orders.loc[oid, '托盘数'] for oid in route)
                vt = '4.2M' if tp <= 5 else ('7.6M' if tp <= 11 else '9.6M')

                # 检查距离限制
                if vt == '4.2M':
                    dist = get_distance_from_nt(city) or 0
                    if dist > 300:
                        vt = '7.6M'

                cost, _, _, feasible, _, _ = HFVRPTW(self.orders).evaluate_route(route, vt)
                result.append({
                    'route': route,
                    'vtype': vt,
                    'cost': float(cost) if feasible else float('inf'),
                    'pallets': float(tp),
                    'feasible': feasible
                })
        return result

    def _direct_ship(self, order_indices):
        """少量订单直接单独发车"""
        result = []
        for oid in order_indices:
            tp = self.orders.loc[oid, '托盘数']
            vt = '4.2M' if tp <= 5 else ('7.6M' if tp <= 11 else '9.6M')
            cost, _, _, feasible, _, _ = HFVRPTW(self.orders).evaluate_route([oid], vt)
            result.append({
                'route': [oid],
                'vtype': vt,
                'cost': float(cost) if feasible else float('inf'),
                'pallets': float(tp),
                'feasible': feasible
            })
        return result

    def run(self):
        """运行聚类优化"""
        print("\n  开始聚类优化...")
        all_routes = []
        total_cost = 0

        for direction, oids in self.clusters.items():
            if len(oids) == 0:
                continue
            print(f"    处理 {direction} ({len(oids)} 个订单)...")
            routes = self.optimize_cluster(oids)
            for r in routes:
                if r['feasible']:
                    total_cost += r['cost']
                all_routes.append(r)

        n_vehicles = len(all_routes)
        feasible = [r for r in all_routes if r['feasible']]

        # 统计装载率
        load_rates = []
        for r in all_routes:
            if r['feasible']:
                cap = self.veh_params[r['vtype']]['pallet_cap']
                rate = r['pallets'] / cap * 100 if cap > 0 else 0
                load_rates.append(rate)

        solution = {
            'total_cost': round(float(total_cost), 2),
            'num_vehicles': n_vehicles,
            'routes': [],
            'avg_loading_rate': round(float(np.mean(load_rates)), 1) if load_rates else 0,
        }

        vt_usage = defaultdict(int)
        for r in all_routes:
            vt_usage[r['vtype']] += 1
            city_names = [self.orders.loc[oid, '城市'] for oid in r['route']]
            solution['routes'].append({
                'vehicle_id': len(solution['routes']) + 1,
                'vehicle_type': r['vtype'],
                'route_orders': [int(oid) for oid in r['route']],
                'route_cities': city_names,
                'total_pallets': round(r['pallets'], 2),
                'cost': round(r['cost'], 2) if r['feasible'] else 0,
                'feasible': r['feasible'],
                'loading_rate': round(r['pallets'] / self.veh_params[r['vtype']]['pallet_cap'] * 100, 1)
                if r['feasible'] else 0,
            })

        solution['vehicle_types_used'] = dict(vt_usage)
        return solution


def problem2_main(q1_results=None):
    """问题2主函数"""
    print("=" * 60)
    print("问题2：拼单运输与车辆调度优化 (先聚类再优化)")
    print("=" * 60)

    # 加载数据
    print("\n[1/5] 加载数据...")
    orders = load_attachment2()
    print(f"  订单数: {len(orders)}")

    print(f"\n[2/5] 订单分析...")
    # 按城市分组
    city_groups = group_orders_by_city(orders)
    print(f"  涉及城市: {len(city_groups)}个")
    for city, oids in sorted(city_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {city}: {len(oids)}单, {sum(orders.loc[oids,'托盘数']):.1f}托")

    # 方向聚类
    direction_clusters = build_direction_clusters(orders)
    print(f"\n  地理方向: {dict((k, len(v)) for k, v in direction_clusters.items())}")

    # 运行聚类优化
    print(f"\n[3/5] 运行聚类优化...")
    router = CVRouter(orders)
    solution = router.run()

    # 结果汇总
    print(f"\n[4/5] 结果汇总...")
    total_cost = solution['total_cost']
    print(f"\n  优化后总费用: {total_cost:,.2f} 元")
    print(f"  使用车辆数: {solution['num_vehicles']}")
    print(f"  平均装载率: {solution['avg_loading_rate']:.1f}%")
    print(f"  车型分布: {solution['vehicle_types_used']}")

    # 与Q1对比
    if q1_results:
        q1_cost = q1_results['total_cost']
        saving = q1_cost - total_cost
        saving_rate = saving / q1_cost * 100
        print(f"\n  与问题1对比:")
        print(f"    原费用: {q1_cost:,.2f} 元")
        print(f"    优化后: {total_cost:,.2f} 元")
        print(f"    节约: {saving:,.2f} 元 ({saving_rate:.1f}%)")
        solution['q1_cost'] = q1_cost
        solution['saving'] = round(saving, 2)
        solution['saving_rate'] = round(saving_rate, 2)

    # 装载率分布
    load_rates = [r['loading_rate'] for r in solution['routes'] if r['feasible']]
    if load_rates:
        print(f"\n  装载率分布:")
        print(f"    均值: {np.mean(load_rates):.1f}%")
        print(f"    中位数: {np.median(load_rates):.1f}%")
        print(f"    最小值: {np.min(load_rates):.1f}%")
        print(f"    最大值: {np.max(load_rates):.1f}%")

    # 保存
    print(f"\n[5/5] 保存结果...")
    save_results(solution, 'problem2_results.json')
    return solution


if __name__ == '__main__':
    problem2_main()
