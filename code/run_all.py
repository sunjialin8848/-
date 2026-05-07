"""全流程运行脚本"""
import sys, os, time
sys.path.insert(0, './src')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.problem1 import problem1_main
from src.problem2 import problem2_main
from src.problem3 import problem3_main
from src.sensitivity_analysis import sensitivity_main
from src.utils import save_results

t0 = time.time()

print('=' * 70)
print('  华东杯 B题 医药物流安排问题 — 编程实现')
print('=' * 70)

# Q1
print('\n' + '■' * 70)
print('  【问题1】运输成本核算')
print('■' * 70)
q1 = problem1_main()
t1 = time.time()
print(f'  耗时: {t1-t0:.1f}s')

# Q2
print('\n' + '■' * 70)
print('  【问题2】拼单运输与车辆调度优化')
print('■' * 70)
q2 = problem2_main(q1)
t2 = time.time()
print(f'  耗时: {t2-t1:.1f}s')

# Q3
print('\n' + '■' * 70)
print('  【问题3】派车合理性评价与配送优化')
print('■' * 70)
q3 = problem3_main()
t3 = time.time()
print(f'  耗时: {t3-t2:.1f}s')

# Sensitivity
print('\n' + '■' * 70)
print('  【灵敏度分析】')
print('■' * 70)
sens = sensitivity_main(q1)
t4 = time.time()
print(f'  耗时: {t4-t3:.1f}s')

# Summary
all_results = {
    'competition': '华东杯 B题',
    'pipeline_step': '编程实现',
    'problem1': q1,
    'problem2': q2,
    'problem3': q3,
    'sensitivity': sens,
    'summary': {
        'q1_total_cost': q1['total_cost'],
        'q2_total_cost': q2['total_cost'],
        'q2_saving_rate': q2.get('saving_rate', 0),
        'q3_score': q3.get('evaluation', {}).get('comprehensive_score', 0),
        'q3_opt_cost': q3.get('optimization', {}).get('total_cost', 0),
        'total_time': round(t4 - t0, 1),
    }
}

save_results(all_results, 'all_results.json')

print('\n' + '=' * 70)
print('  编程实现完成！总耗时: {:.1f}s'.format(t4 - t0))
print('=' * 70)
print(f'\n  问题1 当前运输总费用: {q1["total_cost"]:>10,.2f} 元')
print(f'  问题2 优化后总费用:   {q2["total_cost"]:>10,.2f} 元')
if 'saving_rate' in q2:
    print(f'  问题2 成本节约率:       {q2["saving_rate"]:>10.1f}%')
print(f'  问题3 派车综合评分:     {q3.get("evaluation",{}).get("comprehensive_score",0):>10.1f}/100')
print(f'  问题3 优化方案费用:     {q3.get("optimization",{}).get("total_cost",0):>10,.2f} 元')
