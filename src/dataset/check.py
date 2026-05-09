import json
import os

# 定义图片路径前缀
IMAGE_PREFIX = "/c20250505/xtr/Visual-CoT/cot_images_tar_split/"

def _get_image_path(image_value):
    """从image字段值中提取图片路径"""
    if isinstance(image_value, str):
        return image_value
    elif isinstance(image_value, list) and len(image_value) > 0:
        # 如果是列表，取第一个元素
        if isinstance(image_value[0], str):
            return image_value[0]
        else:
            return None
    return None

def _check_image_field(data, line_number, stats):
    """检查image字段的路径是否存在，处理列表格式"""
    if 'image' not in data:
        stats['missing_image'] += 1
        return
    
    image_value = data['image']
    stats['has_image'] += 1
    
    # 检查image字段类型
    if not isinstance(image_value, (str, list)):
        stats['image_format_error'] += 1
        stats['problem_lines'].append(f"行 {line_number}: image字段不是字符串或列表，类型={type(image_value)}")
        if stats['image_format_error'] <= 5:
            print(f"⚠️  行 {line_number}: image字段不是字符串或列表，类型={type(image_value)}")
        return
    
    # 从image值中提取路径
    image_path = _get_image_path(image_value)
    
    if image_path is None:
        stats['image_format_error'] += 1
        stats['problem_lines'].append(f"行 {line_number}: 无法从image字段提取路径，值={image_value}")
        if stats['image_format_error'] <= 5:
            print(f"⚠️  行 {line_number}: 无法从image字段提取路径，值={image_value}")
        return
    
    # 构建完整图片路径（添加前缀）
    if not image_path.startswith(IMAGE_PREFIX):
        full_image_path = IMAGE_PREFIX + image_path
    else:
        full_image_path = image_path
    
    # 检查路径是否存在
    if os.path.exists(full_image_path):
        stats['image_path_exists'] += 1
        # 打印前3个存在的图片路径
        if stats['image_path_exists'] <= 3:
            print(f"✅ 行 {line_number}: 图片路径存在 - {full_image_path}")
    else:
        stats['image_path_not_exists'] += 1
        stats['problem_lines'].append(f"行 {line_number}: 图片路径不存在 - {full_image_path}")
        if stats['image_path_not_exists'] <= 5:
            print(f"❌ 行 {line_number}: 图片路径不存在 - {full_image_path}")
        
        # 尝试查找可能的路径
        _try_find_image_path(full_image_path, line_number, image_path)

def _try_find_image_path(full_image_path, line_number, original_path):
    """尝试查找图片文件的可能位置"""
    # 检查是否为常见的图片扩展名
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    _, ext = os.path.splitext(full_image_path)
    if ext.lower() not in valid_extensions:
        print(f"  警告: 文件扩展名 {ext} 可能不是有效的图片格式")
    
    # 尝试在不添加前缀的情况下查找
    if not original_path.startswith(IMAGE_PREFIX):
        # 尝试在标准前缀下查找
        print(f"  提示: 尝试使用标准前缀 {IMAGE_PREFIX} + {original_path}")

def _find_image_file(image_path):
    """
    尝试查找图片文件的可能位置，自动添加前缀
    返回找到的路径或None
    """
    # 构建完整路径（添加前缀）
    if not image_path.startswith(IMAGE_PREFIX):
        full_path = IMAGE_PREFIX + image_path
    else:
        full_path = image_path
    
    # 如果完整路径存在，直接返回
    if os.path.exists(full_path):
        return full_path
    
    # 如果已经是绝对路径且存在，直接返回
    if os.path.exists(image_path):
        return image_path
    
    # 提取文件名
    filename = os.path.basename(full_path)
    
    # 在常见目录查找
    common_dirs = [
        IMAGE_PREFIX,  # 标准前缀目录
        '.',  # 当前目录
        './images',
        './data/images',
        '../images',
        os.path.expanduser('~/images')
    ]
    
    for dir_path in common_dirs:
        if os.path.exists(dir_path):
            # 首先尝试查找原始文件名
            test_path = os.path.join(dir_path, filename)
            if os.path.exists(test_path):
                return test_path
            
            # 尝试查找不带前缀的文件名
            if image_path != full_path:
                test_path = os.path.join(dir_path, os.path.basename(image_path))
                if os.path.exists(test_path):
                    return test_path
    
    return None

def check_all_embeddings(data_path, check_image_path=False):
    """
    检查整个数据集中所有emb字段的完整性和一致性
    参数:
        data_path: 数据文件路径
        check_image_path: 是否检查image字段的路径是否存在
    """
    print("="*60)
    print("完整数据集检查工具")
    print(f"检查文件: {data_path}")
    print(f"图片路径前缀: {IMAGE_PREFIX}")
    print(f"文件大小: {os.path.getsize(data_path) / 1024 / 1024:.2f} MB")
    print("="*60)
    
    # 统计信息
    stats = {
        'total_lines': 0,
        'valid_4096': 0,
        'missing_emb': 0,
        'wrong_format': 0,
        'wrong_dimension': 0,
        'parsing_errors': 0,
        'dimensions': {},
        'problem_lines': [],
        # 新增image字段统计
        'has_image': 0,
        'missing_image': 0,
        'image_path_exists': 0,
        'image_path_not_exists': 0,
        'image_format_error': 0,
        # 新增image格式统计
        'image_string': 0,
        'image_list': 0,
        'image_other': 0,
    }
    
    # 读取整个文件
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                stats['total_lines'] += 1
                
                try:
                    data = json.loads(line.strip())
                    
                    # 1. 检查emb字段是否存在
                    if 'emb' not in data:
                        stats['missing_emb'] += 1
                        stats['problem_lines'].append(f"行 {line_number}: 没有emb字段")
                        if stats['missing_emb'] <= 5:
                            print(f"❌ 行 {line_number}: 没有emb字段")
                        continue
                    
                    emb = data['emb']
                    
                    # 检查emb是否为None
                    if emb is None:
                        stats['missing_emb'] += 1
                        stats['problem_lines'].append(f"行 {line_number}: emb字段为None")
                        if stats['missing_emb'] <= 5:
                            print(f"❌ 行 {line_number}: emb字段为None")
                        continue
                    
                    # 检查emb是否为列表
                    if not isinstance(emb, list):
                        stats['wrong_format'] += 1
                        stats['problem_lines'].append(f"行 {line_number}: emb不是列表，类型={type(emb)}")
                        if stats['wrong_format'] <= 5:
                            print(f"❌ 行 {line_number}: emb不是列表，类型={type(emb)}")
                        continue
                    
                    # 检查维度
                    dim = len(emb)
                    
                    # 记录维度分布
                    if dim not in stats['dimensions']:
                        stats['dimensions'][dim] = 0
                    stats['dimensions'][dim] += 1
                    
                    # 检查是否为4096维
                    if dim != 4096:
                        stats['wrong_dimension'] += 1
                        stats['problem_lines'].append(f"行 {line_number}: 维度={dim}, 应为4096")
                        if stats['wrong_dimension'] <= 5:
                            print(f"❌ 行 {line_number}: 维度={dim}, 应为4096")
                        continue
                    
                    # 检查数据类型（是否为数值）
                    if len(emb) > 0:
                        first_val = emb[0]
                        if not isinstance(first_val, (int, float)):
                            stats['wrong_format'] += 1
                            stats['problem_lines'].append(f"行 {line_number}: emb[0]不是数值，类型={type(first_val)}")
                            if stats['wrong_format'] <= 5:
                                print(f"❌ 行 {line_number}: emb[0]不是数值，类型={type(first_val)}")
                            continue
                    
                    # 所有emb检查通过
                    stats['valid_4096'] += 1
                    
                    # 2. 检查image字段（如果启用）
                    if check_image_path:
                        _check_image_field(data, line_number, stats)
                        
                        # 统计image字段格式
                        if 'image' in data:
                            image_value = data['image']
                            if isinstance(image_value, str):
                                stats['image_string'] += 1
                            elif isinstance(image_value, list):
                                stats['image_list'] += 1
                            else:
                                stats['image_other'] += 1
                    
                    # 打印前3个有效样本的信息
                    if stats['valid_4096'] <= 3:
                        print(f"✅ 行 {line_number}: 维度正确 [4096], 数据类型: {type(emb[0])}")
                        if check_image_path and 'image' in data:
                            # 显示完整路径
                            image_path = _get_image_path(data['image'])
                            if image_path:
                                if not image_path.startswith(IMAGE_PREFIX):
                                    full_path = IMAGE_PREFIX + image_path
                                else:
                                    full_path = image_path
                                print(f"   图片路径: {full_path}")
                                print(f"   image字段格式: {type(data['image'])}")
                
                except json.JSONDecodeError as e:
                    stats['parsing_errors'] += 1
                    stats['problem_lines'].append(f"行 {line_number}: JSON解析错误 - {str(e)}")
                    if stats['parsing_errors'] <= 5:
                        print(f"❌ 行 {line_number}: JSON解析错误 - {str(e)}")
                except Exception as e:
                    stats['parsing_errors'] += 1
                    stats['problem_lines'].append(f"行 {line_number}: 处理错误 - {str(e)}")
                    if stats['parsing_errors'] <= 5:
                        print(f"❌ 行 {line_number}: 处理错误 - {str(e)}")
                
                # 每10000行打印一次进度
                if line_number % 10000 == 0:
                    progress_msg = f"已检查 {line_number} 行... (有效emb: {stats['valid_4096']}, 问题: {stats['total_lines'] - stats['valid_4096'] - stats['parsing_errors']})"
                    if check_image_path:
                        progress_msg += f", 有image: {stats['has_image']}, 图片存在: {stats['image_path_exists']}"
                    print(progress_msg)
                    
    except Exception as e:
        print(f"打开文件时出错: {e}")
        return None
    
    # 打印详细统计结果
    print("\n" + "="*60)
    print("完整统计结果:")
    print("="*60)
    print(f"总行数: {stats['total_lines']}")
    print(f"有效4096维emb: {stats['valid_4096']} ({stats['valid_4096']/stats['total_lines']*100:.2f}%)")
    print(f"缺少emb字段: {stats['missing_emb']} ({stats['missing_emb']/stats['total_lines']*100:.2f}%)")
    print(f"格式错误: {stats['wrong_format']} ({stats['wrong_format']/stats['total_lines']*100:.2f}%)")
    print(f"维度错误: {stats['wrong_dimension']} ({stats['wrong_dimension']/stats['total_lines']*100:.2f}%)")
    print(f"解析错误: {stats['parsing_errors']} ({stats['parsing_errors']/stats['total_lines']*100:.2f}%)")
    
    # 打印维度分布
    if stats['dimensions']:
        print(f"\n维度分布:")
        for dim, count in sorted(stats['dimensions'].items()):
            percentage = count/stats['total_lines']*100
            print(f"  {dim}维: {count}行 ({percentage:.2f}%)")
    
    # 打印image字段统计（如果检查了）
    if check_image_path:
        print(f"\n{'='*60}")
        print("Image字段统计:")
        print(f"有image字段: {stats['has_image']} ({stats['has_image']/stats['total_lines']*100:.2f}%)")
        print(f"无image字段: {stats['missing_image']} ({stats['missing_image']/stats['total_lines']*100:.2f}%)")
        print(f"图片路径存在: {stats['image_path_exists']} ({stats['image_path_exists']/max(1, stats['has_image'])*100:.2f}%)")
        print(f"图片路径不存在: {stats['image_path_not_exists']} ({stats['image_path_not_exists']/max(1, stats['has_image'])*100:.2f}%)")
        print(f"图片格式错误: {stats['image_format_error']} ({stats['image_format_error']/max(1, stats['has_image'])*100:.2f}%)")
        
        # 打印image格式分布
        if stats['has_image'] > 0:
            print(f"\nImage字段格式分布:")
            print(f"  字符串格式: {stats['image_string']} ({stats['image_string']/stats['has_image']*100:.2f}%)")
            print(f"  列表格式: {stats['image_list']} ({stats['image_list']/stats['has_image']*100:.2f}%)")
            if stats['image_other'] > 0:
                print(f"  其他格式: {stats['image_other']} ({stats['image_other']/stats['has_image']*100:.2f}%)")
    
    # 总结和建议
    print("\n" + "="*60)
    print("检查结果总结:")
    
    if stats['valid_4096'] == stats['total_lines']:
        print("✅ 完美！所有行都有正确的4096维emb字段！")
    elif stats['valid_4096'] > 0 and stats['valid_4096'] == stats['total_lines'] - stats['parsing_errors']:
        print("✅ 所有可解析的行都有正确的4096维emb字段！")
    else:
        print("❌ 发现问题！")
        if stats['missing_emb'] > 0:
            print(f"  - {stats['missing_emb']} 行缺少emb字段")
        if stats['wrong_format'] > 0:
            print(f"  - {stats['wrong_format']} 行格式错误")
        if stats['wrong_dimension'] > 0:
            print(f"  - {stats['wrong_dimension']} 行维度不是4096")
        if stats['parsing_errors'] > 0:
            print(f"  - {stats['parsing_errors']} 行解析错误")
        
        print(f"\n问题行总数: {stats['total_lines'] - stats['valid_4096']}")
    
    # 保存问题行到文件
    if stats['problem_lines']:
        problem_file = data_path.replace('.jsonl', '_problems.txt')
        with open(problem_file, 'w', encoding='utf-8') as pf:
            pf.write(f"数据文件: {data_path}\n")
            pf.write(f"图片路径前缀: {IMAGE_PREFIX}\n")
            pf.write(f"检查时间: {os.path.getsize(data_path) / 1024 / 1024:.2f} MB\n")
            pf.write(f"总行数: {stats['total_lines']}\n")
            pf.write(f"有效行: {stats['valid_4096']}\n")
            pf.write(f"问题行: {stats['total_lines'] - stats['valid_4096']}\n\n")
            pf.write("详细问题:\n")
            pf.write("="*50 + "\n")
            for problem in stats['problem_lines']:
                pf.write(problem + "\n")
        print(f"\n问题行详细记录已保存到: {problem_file}")
    
    return stats

def fix_all_embeddings(input_path, output_path, fix_image_paths=False):
    """
    修复整个数据集的emb字段
    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        fix_image_paths: 是否修复不存在的图片路径
    """
    print("="*60)
    print("数据集修复工具")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"图片路径前缀: {IMAGE_PREFIX}")
    print("="*60)
    
    stats = {
        'total': 0,
        'fixed_emb': 0,
        'fixed_image': 0,
        'skipped': 0,
        'errors': 0,
        'image_not_found': [],
        'image_format_converted': 0
    }
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        line_number = 0
        for line in fin:
            line_number += 1
            stats['total'] += 1
            
            try:
                data = json.loads(line.strip())
                needs_fix_emb = False
                needs_fix_image = False
                
                # 1. 检查并修复emb字段
                if 'emb' not in data or data['emb'] is None:
                    # 创建全零的4096维向量
                    data['emb'] = [0.0] * 4096
                    needs_fix_emb = True
                else:
                    emb = data['emb']
                    if not isinstance(emb, list):
                        # 如果不是列表，创建全零向量
                        data['emb'] = [0.0] * 4096
                        needs_fix_emb = True
                    elif len(emb) != 4096:
                        # 调整维度到4096
                        if len(emb) > 4096:
                            # 截断
                            data['emb'] = emb[:4096]
                        else:
                            # 填充
                            data['emb'] = emb + [0.0] * (4096 - len(emb))
                        needs_fix_emb = True
                
                if needs_fix_emb:
                    stats['fixed_emb'] += 1
                
                # 2. 检查并修复image字段（如果启用）
                if fix_image_paths and 'image' in data:
                    image_value = data['image']
                    
                    if isinstance(image_value, str):
                        # 如果是字符串，转换为列表格式
                        if not image_value.startswith(IMAGE_PREFIX):
                            full_path = IMAGE_PREFIX + image_value
                        else:
                            full_path = image_value
                        
                        if not os.path.exists(full_path):
                            # 尝试查找图片文件
                            found_path = _find_image_file(image_value)
                            if found_path:
                                # 如果找到了，更新为列表格式
                                data['image'] = [found_path]
                                needs_fix_image = True
                                stats['fixed_image'] += 1
                                stats['image_format_converted'] += 1
                            else:
                                stats['image_not_found'].append(f"行 {line_number}: {full_path}")
                        # 如果路径已经是完整路径但相对，转换为带前缀的路径并保持列表格式
                        elif not image_value.startswith(IMAGE_PREFIX) and not os.path.isabs(image_value):
                            data['image'] = [IMAGE_PREFIX + image_value]
                            needs_fix_image = True
                            stats['fixed_image'] += 1
                            stats['image_format_converted'] += 1
                        # 确保是列表格式
                        elif isinstance(image_value, str):
                            data['image'] = [image_value]
                            needs_fix_image = True
                            stats['fixed_image'] += 1
                            stats['image_format_converted'] += 1
                    
                    elif isinstance(image_value, list) and len(image_value) > 0:
                        # 如果是列表，处理第一个元素
                        image_path = image_value[0] if isinstance(image_value[0], str) else str(image_value[0])
                        
                        if not image_path.startswith(IMAGE_PREFIX):
                            full_path = IMAGE_PREFIX + image_path
                        else:
                            full_path = image_path
                        
                        if not os.path.exists(full_path):
                            # 尝试查找图片文件
                            found_path = _find_image_file(image_path)
                            if found_path:
                                # 更新列表中的路径
                                data['image'][0] = found_path
                                needs_fix_image = True
                                stats['fixed_image'] += 1
                            else:
                                stats['image_not_found'].append(f"行 {line_number}: {full_path}")
                        # 如果路径已经是完整路径但相对，转换为带前缀的路径
                        elif not image_path.startswith(IMAGE_PREFIX) and not os.path.isabs(image_path):
                            data['image'][0] = IMAGE_PREFIX + image_path
                            needs_fix_image = True
                            stats['fixed_image'] += 1
                
                if not needs_fix_emb and not needs_fix_image:
                    stats['skipped'] += 1
                
                # 写入修复后的数据
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # 每10000行打印一次进度
                if line_number % 10000 == 0:
                    progress_msg = f"已处理 {line_number} 行... "
                    if needs_fix_emb or needs_fix_image:
                        progress_msg += f"(修复emb: {stats['fixed_emb']}, 修复image: {stats['fixed_image']}, 跳过: {stats['skipped']})"
                    else:
                        progress_msg += f"(跳过: {stats['skipped']})"
                    print(progress_msg)
                    
            except Exception as e:
                stats['errors'] += 1
                print(f"❌ 行 {line_number}: 处理错误 - {str(e)}")
                continue
    
    print("\n" + "="*60)
    print("修复完成:")
    print(f"总行数: {stats['total']}")
    print(f"修复emb行数: {stats['fixed_emb']}")
    if fix_image_paths:
        print(f"修复image行数: {stats['fixed_image']}")
        print(f"image格式转换: {stats['image_format_converted']}")
        if stats['image_not_found']:
            print(f"未找到的图片: {len(stats['image_not_found'])}")
            # 保存未找到的图片列表
            not_found_file = output_path.replace('.jsonl', '_images_not_found.txt')
            with open(not_found_file, 'w', encoding='utf-8') as nf:
                nf.write(f"图片路径前缀: {IMAGE_PREFIX}\n")
                nf.write(f"未找到的图片列表 (共{len(stats['image_not_found'])}个):\n")
                nf.write("="*60 + "\n")
                for item in stats['image_not_found'][:100]:  # 只保存前100个
                    nf.write(item + '\n')
                if len(stats['image_not_found']) > 100:
                    nf.write(f"... 还有 {len(stats['image_not_found']) - 100} 个未列出\n")
            print(f"未找到的图片列表已保存到: {not_found_file}")
    print(f"跳过行数: {stats['skipped']}")
    print(f"错误行数: {stats['errors']}")
    
    # 验证修复后的文件
    print("\n验证修复后的文件...")
    check_result = check_all_embeddings(output_path, check_image_path=fix_image_paths)
    
    if check_result and check_result['valid_4096'] == check_result['total_lines'] - check_result['parsing_errors']:
        print("✅ 修复成功！所有可解析的行都有正确的4096维emb字段！")
    else:
        print("⚠️  修复可能不完全，请检查输出文件")
    
    return stats

def check_specific_lines(data_path, line_numbers, check_image_path=False):
    """
    检查指定行号的emb和image字段
    """
    print("="*60)
    print(f"检查指定行: {line_numbers}")
    print(f"图片路径前缀: {IMAGE_PREFIX}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i in line_numbers:
                print(f"\n行 {i}:")
                try:
                    data = json.loads(line.strip())
                    
                    # 检查emb字段
                    if 'emb' not in data:
                        print("  ❌ 没有emb字段")
                        print(f"  可用字段: {list(data.keys())}")
                    else:
                        emb = data['emb']
                        if emb is None:
                            print("  ❌ emb字段为None")
                        elif not isinstance(emb, list):
                            print(f"  ❌ emb不是列表，类型: {type(emb)}")
                        else:
                            dim = len(emb)
                            print(f"  ✅ emb维度: {dim}")
                            if dim != 4096:
                                print(f"  ⚠️  警告: 维度不是4096")
                            
                            # 打印前5个值
                            if len(emb) > 5:
                                print(f"  emb前5个值: {emb[:5]}")
                            
                            # 检查数据类型
                            if len(emb) > 0:
                                print(f"  emb数据类型: {type(emb[0])}")
                    
                    # 检查image字段
                    if check_image_path:
                        if 'image' not in data:
                            print("  ❌ 没有image字段")
                        else:
                            image_value = data['image']
                            print(f"  ✅ image字段类型: {type(image_value)}")
                            print(f"  ✅ image字段值: {image_value}")
                            
                            # 提取图片路径
                            image_path = _get_image_path(image_value)
                            
                            if image_path is None:
                                print(f"  ❌ 无法提取图片路径")
                            else:
                                # 构建完整路径
                                if not image_path.startswith(IMAGE_PREFIX):
                                    full_path = IMAGE_PREFIX + image_path
                                else:
                                    full_path = image_path
                                
                                print(f"  ✅ image提取路径: {image_path}")
                                print(f"  ✅ image完整路径: {full_path}")
                                if os.path.exists(full_path):
                                    print(f"  ✅ 图片文件存在")
                                    file_size = os.path.getsize(full_path)
                                    print(f"     文件大小: {file_size/1024:.2f} KB")
                                    # 检查文件类型
                                    if full_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                                        print(f"     文件类型: 图片文件")
                                    else:
                                        print(f"     ⚠️ 文件类型: 可能不是标准图片格式")
                                else:
                                    print(f"  ❌ 图片文件不存在")
                    
                except Exception as e:
                    print(f"  ❌ 解析错误: {e}")

if __name__ == "__main__":
    data_path = "/c20250505/xtr/output2_strict_with_emb_and_image_modified.jsonl"
    
    print(f"默认图片路径前缀: {IMAGE_PREFIX}")
    
    # 询问是否检查image路径
    check_image = input("是否检查image字段路径是否存在？(y/n): ").lower() == 'y'
    
    # 1. 完整检查整个数据集
    print("开始完整检查整个数据集...")
    stats = check_all_embeddings(data_path, check_image_path=check_image)
    
    if stats:
        # 2. 询问是否修复
        if stats['valid_4096'] < stats['total_lines'] - stats['parsing_errors']:
            print("\n" + "="*60)
            response = input("发现有问题行，是否修复数据集？(y/n): ")
            
            if response.lower() == 'y':
                output_path = data_path.replace('.jsonl', '_fixed.jsonl')
                
                # 确认是否覆盖
                if os.path.exists(output_path):
                    overwrite = input(f"文件 {output_path} 已存在，是否覆盖？(y/n): ")
                    if overwrite.lower() != 'y':
                        timestamp = os.path.splitext(output_path)[0] + f"_{int(os.path.getmtime(data_path))}.jsonl"
                        output_path = timestamp
                
                # 询问是否修复image路径
                fix_images = False
                if check_image and stats['image_path_not_exists'] > 0:
                    fix_images = input(f"发现 {stats['image_path_not_exists']} 个图片路径不存在，是否尝试修复？(y/n): ").lower() == 'y'
                
                # 执行修复
                fix_all_embeddings(data_path, output_path, fix_image_paths=fix_images)
        else:
            print("\n✅ 数据集已完美，无需修复！")
    
    # 3. 询问是否检查特定行
    print("\n" + "="*60)
    check_specific = input("是否检查特定行号？(输入行号，多个用逗号分隔，或输入n跳过): ")
    if check_specific.lower() != 'n':
        try:
            line_numbers = [int(num.strip()) for num in check_specific.split(',')]
            check_specific_lines(data_path, line_numbers, check_image_path=check_image)
        except ValueError:
            print("输入格式错误，请使用逗号分隔的数字")