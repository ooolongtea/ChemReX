#!/usr/bin/env python3
"""
中英文NER性能对比雷达图
用法: python radar_comparison.py

请在下方 JSON_FILES 列表中填入你的6个模型结果文件路径
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ============================================================
# 【在这里填入你的JSON文件路径】
# ============================================================
_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
JSON_FILES = [
    # 格式: ("模型名称", "JSON文件路径")
    ("Qwen3-8B Base", os.path.join(_results, "external_Qwen3-8B_20260630-115257", "eval_metrics_language.json")),
    ("DeepSeek-V3", os.path.join(_results, "external_deepseek-v3_20260201-155435", "eval_metrics_language.json")),
    ("GPT-4o", os.path.join(_results, "external_gpt-4o_20260131-205535", "eval_metrics_language.json")),
    ("Qwen-Plus", os.path.join(_results, "external_qwen-plus_20260201-145935", "eval_metrics_language.json")),
    ("ChemReX", os.path.join(_results, "external_qwen3-lora-08_20260630-121627", "eval_metrics_language.json")),
    ("ReactionSeek", os.path.join(_results, "reactionseek_Qwen3-8B_20260701_174918", "eval_metrics_language.json")),
]

# 输出文件路径（不含扩展名，脚本会自动生成 .png 和 .svg）
OUTPUT_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radar_en_zh_comparison_all")

# 图例文字（可自定义）
LEGEND_LABELS = {
    "EN": "English-F1(%)",
    "ZH": "Chinese-F1(%)",
}

# ============================================================
# 以下代码无需修改
# ============================================================

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

COLORS = {
    'EN': '#2E86AB',  # radar_comparison_v2 经典蓝
    'ZH': '#E94F37',  # radar_comparison_v2 经典红
}

# 实体类型映射
FIELDS = ['REACTION_PRODUCT', 'STARTING_MATERIAL', 'REAGENT_CATALYST', 
          'SOLVENT', 'TEMPERATURE', 'TIME', 'YIELD_PERCENT']
FIELD_LABELS = ['Prod.', 'Start.Mat.', 'Rea/Cat.', 'Solv.', 'Temp.', 'Time', 'Yield']

TITLE_FONT_SIZE = 17
LEGEND_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 10
GAP_TEXT_FONT_SIZE = 10
IOU_FONT_SIZE = 12


def load_json_data(json_path):
    """加载JSON文件并提取中英文NER F1分数"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = {'EN': {}, 'ZH': {}}
    
    # 提取英文数据
    if 'by_language' in data and 'EN' in data['by_language']:
        en_data = data['by_language']['EN']['ner']['field_metrics']
        for field in FIELDS:
            if field in en_data:
                result['EN'][field] = en_data[field]['f1']
            else:
                result['EN'][field] = 0.0
    
    # 提取中文数据
    if 'by_language' in data and 'ZH' in data['by_language']:
        zh_data = data['by_language']['ZH']['ner']['field_metrics']
        for field in FIELDS:
            if field in zh_data:
                result['ZH'][field] = zh_data[field]['f1']
            else:
                result['ZH'][field] = 0.0
    
    return result


def compute_polygon_area(values):
    """将雷达图数值转成平面多边形面积，用于衡量整体能力覆盖。"""
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    x = np.array(values) * np.cos(angles)
    y = np.array(values) * np.sin(angles)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_overlap_iou(data):
    en_values = np.array([data['EN'].get(f, 0.0) for f in FIELDS], dtype=float)
    zh_values = np.array([data['ZH'].get(f, 0.0) for f in FIELDS], dtype=float)
    overlap_values = np.minimum(en_values, zh_values)
    union_values = np.maximum(en_values, zh_values)
    union_area = compute_polygon_area(union_values)
    if union_area <= 1e-12:
        return 1.0
    overlap_area = compute_polygon_area(overlap_values)
    return float(overlap_area / union_area)

def plot_radar_comparison1(models_data, output_path):
    """
    表格化雷达图：一个模型 = 一个单元格
    """
    n_models = len(models_data)
    n_cols = 3                      # 固定 3 列（论文常用）
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        subplot_kw=dict(projection='polar')
    )

    axes = np.array(axes).reshape(n_rows, n_cols)

    # 雷达角度
    angles = np.linspace(0, 2 * np.pi, len(FIELDS), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    for idx, (model_name, data) in enumerate(models_data):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        en = [data['EN'].get(f, 0) for f in FIELDS] + [data['EN'].get(FIELDS[0], 0)]
        zh = [data['ZH'].get(f, 0) for f in FIELDS] + [data['ZH'].get(FIELDS[0], 0)]

        ax.plot(angles, en, color=COLORS['EN'], linewidth=2)
        ax.fill(angles, en, color=COLORS['EN'], alpha=0.25)

        ax.plot(angles, zh, color=COLORS['ZH'], linewidth=2)
        ax.fill(angles, zh, color=COLORS['ZH'], alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FIELD_LABELS, fontsize=9)

        ax.set_ylim(0, 1.0)
        # ax.set_yticks([])  # 关键：像表格一样，去掉刻度
        ax.grid(True, linewidth=0.6, alpha=0.6)

        ax.set_title(model_name, fontsize=12, pad=12, fontweight='bold')

    # 隐藏空单元格（像空表格）
    for idx in range(n_models, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    # 全局图例（只出现一次）
    handles = [
        plt.Line2D([0], [0], color=COLORS['EN'], lw=2, label=LEGEND_LABELS['EN']),
        plt.Line2D([0], [0], color=COLORS['ZH'], lw=2, label=LEGEND_LABELS['ZH'])
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=2,
        frameon=False,
        fontsize=11
    )

    plt.subplots_adjust(
        left=0.05, right=0.95,
        top=0.92, bottom=0.08,
        wspace=0.25, hspace=0.35
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ 表格化雷达图已保存: {output_path}")


def plot_radar_comparison(models_data, output_path):
    """
    绘制雷达图对比
    models_data: list of (model_name, data_dict)
    """
    n_models = len(models_data)
    
    # 根据模型数量调整布局
    if n_models <= 3:
        n_rows, n_cols = 1, n_models
        fig_width = 5 * n_models
        fig_height = 5
    elif n_models <= 6:
        n_rows, n_cols = 2, 3
        fig_width = 15.8
        fig_height = 11.4
    else:
        n_rows = (n_models + 2) // 3
        n_cols = 3
        fig_width = 15.8
        fig_height = 5.6 * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                              subplot_kw=dict(projection='polar'))
    
    # 确保axes是二维数组
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(FIELDS), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for idx, (model_name, data) in enumerate(models_data):
        ax = axes_flat[idx]
        
        # 获取数据
        en_values = [data['EN'].get(f, 0) for f in FIELDS]
        zh_values = [data['ZH'].get(f, 0) for f in FIELDS]
        en_values += en_values[:1]  # 闭合
        zh_values += zh_values[:1]
        
        # 绘制
        ax.plot(angles, en_values, 'o-', linewidth=2, label=LEGEND_LABELS["EN"], 
                color=COLORS['EN'], markersize=5)
        ax.fill(angles, en_values, alpha=0.25, color=COLORS['EN'])
        ax.plot(angles, zh_values, 's-', linewidth=2, label=LEGEND_LABELS["ZH"], 
                color=COLORS['ZH'], markersize=5)
        ax.fill(angles, zh_values, alpha=0.25, color=COLORS['ZH'])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FIELD_LABELS, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.tick_params(axis="y", width=4)
        # 显示子图标题（模型名）
        overlap_iou = compute_overlap_iou(data)
        ax.set_title(model_name, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=30)
        ax.text(
            0.5,
            -0.11,
            f"IoU: {overlap_iou * 100:.1f}%",
            transform=ax.transAxes,
            fontsize=IOU_FONT_SIZE,
            fontweight='normal',
            ha='center',
            va='center',
            color='#333333',
        )
        ax.grid(True, alpha=0.55, linewidth=0.8, color='gray')
    
    # 隐藏多余的子图
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # 不显示总标题
    # 全局图例放在所有子图下方
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE
    )

    plt.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.12, wspace=0.32, hspace=0.32)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 图片已保存: {output_path}")


def main():
    print("="*60)
    print("中英文NER性能对比雷达图生成器")
    print("="*60)
    
    # 检查文件是否存在
    models_data = []
    for model_name, json_path in JSON_FILES:
        if not os.path.exists(json_path):
            print(f"✗ 文件不存在: {json_path}")
            print(f"  请检查路径是否正确")
            continue
        
        try:
            data = load_json_data(json_path)
            models_data.append((model_name, data))
            print(f"✓ 已加载: {model_name}")
        except Exception as e:
            print(f"✗ 加载失败 {model_name}: {e}")
    
    if len(models_data) == 0:
        print("\n错误: 没有成功加载任何模型数据!")
        print("请检查 JSON_FILES 中的文件路径是否正确")
        return
    
    print(f"\n共加载 {len(models_data)} 个模型")
    print("正在生成雷达图...")

    plot_radar_comparison(models_data, OUTPUT_PREFIX + ".png")
    plot_radar_comparison(models_data, OUTPUT_PREFIX + ".svg")

    print("\n" + "="*60)
    print("完成!")
    print("="*60)


if __name__ == '__main__':
    main()
