"""
主程序：华东杯 B题 医药物流安排问题
串联问题1(成本核算) → 问题2(拼单优化) → 问题3(派车评价) → 灵敏度分析
"""
import sys, os, time
# 获取项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from src.problem1 import problem1_main
from src.problem2 import problem2_main
from src.problem3 import problem3_main
from src.sensitivity_analysis import sensitivity_main
from src.utils import save_results


def main():
    print("=" * 70)
    print("  华东杯 B题 医药物流安排问题 — 编程实现")
    print("=" * 70)

    start_time = time.time()

    # ========== 问题1 ==========
    t0 = time.time()
    print("\n" + "■" * 70)
    print("  【问题1】运输成本核算")
    print("■" * 70)
    q1_results = problem1_main()
    t1 = time.time()
    print(f"  ⏱ 耗时: {t1-t0:.1f}s")

    # ========== 问题2 ==========
    print("\n" + "■" * 70)
    print("  【问题2】拼单运输与车辆调度优化")
    print("■" * 70)
    q2_results = problem2_main(q1_results)
    t2 = time.time()
    print(f"  ⏱ 耗时: {t2-t1:.1f}s")

    # ========== 问题3 ==========
    print("\n" + "■" * 70)
    print("  【问题3】派车合理性评价与配送优化")
    print("■" * 70)
    q3_results = problem3_main()
    t3 = time.time()
    print(f"  ⏱ 耗时: {t3-t1:.1f}s")

    # ========== 灵敏度分析 ==========
    print("\n" + "■" * 70)
    print("  【灵敏度分析】")
    print("■" * 70)
    sensitivity_results = sensitivity_main(q1_results)
    t4 = time.time()
    print(f"  ⏱ 耗时: {t4-t3:.1f}s")

    # ========== 汇总 ==========
    print("\n" + "■" * 70)
    print("  【结果汇总】")
    print("■" * 70)

    all_results = {
        'competition': '华东杯 B题',
        'pipeline_step': '编程实现',
        'problem1': q1_results,
        'problem2': q2_results,
        'problem3': q3_results,
        'sensitivity': sensitivity_results,
        'summary': {
            'q1_total_cost': q1_results['total_cost'],
            'q2_total_cost': q2_results['total_cost'],
            'q2_saving_rate': q2_results.get('saving_rate', 0),
            'q3_score': q3_results.get('evaluation', {}).get('comprehensive_score', 0),
            'q3_opt_cost': q3_results.get('optimization', {}).get('total_cost', 0),
            'total_time': round(t4 - start_time, 1),
        }
    }

    print(f"\n  问题1 当前运输总费用: {q1_results['total_cost']:>10,.2f} 元")
    print(f"  问题2 优化后总费用:   {q2_results['total_cost']:>10,.2f} 元")
    if 'saving_rate' in q2_results:
        print(f"  问题2 成本节约率:       {q2_results['saving_rate']:>10.1f}%")
    print(f"  问题3 派车综合评分:     {q3_results.get('evaluation',{}).get('comprehensive_score',0):>10.1f}/100")
    print(f"  问题3 优化方案费用:     {q3_results.get('optimization',{}).get('total_cost',0):>10,.2f} 元")
    print(f"\n  总耗时: {t4-start_time:.1f}s")

    save_results(all_results, 'all_results.json')

    print("\n" + "=" * 70)
    print("  编程实现完成！所有结果已保存至 figures/ 目录")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
