import re
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import RGBColor

def create_use_case_table(document, use_cases):
    """
    创建用例表格
    """
    # 添加标题
    heading = document.add_heading('5. 用例设计', level=1)
    
    # 为每个用例创建表格
    for i, use_case in enumerate(use_cases, 1):
        # 添加用例标题
        subtitle = document.add_heading(f'5.{i} {use_case["name"]}', level=2)
        
        # 创建表格 (5列: 项目, 描述, 前置条件, 主要流程, 后置条件)
        table = document.add_table(rows=1, cols=5)
        table.style = 'Table Grid'
        
        # 设置表头
        hdr_cells = table.rows[0].cells
        headers = ['项目', '描述', '前置条件', '主要流程', '后置条件']
        for j, header in enumerate(headers):
            hdr_cells[j].text = header
            # 设置表头格式
            for paragraph in hdr_cells[j].paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.size = Pt(10)
        
        # 填充表格内容
        row_cells = table.add_row().cells
        
        # 项目 (用例名称)
        row_cells[0].text = use_case.get('name', '')
        
        # 描述
        row_cells[1].text = use_case.get('description', '')
        
        # 前置条件
        preconditions = use_case.get('preconditions', [])
        row_cells[2].text = '\n'.join([f"{idx}. {cond}" for idx, cond in enumerate(preconditions, 1)])
        
        # 主要流程
        main_flow = use_case.get('main_flow', [])
        row_cells[3].text = '\n'.join([f"{idx}. {step}" for idx, step in enumerate(main_flow, 1)])
        
        # 后置条件
        row_cells[4].text = use_case.get('postconditions', '')
        
        # 设置表格字体大小
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
        
        # 设置表格列宽
        widths = [Inches(1.0), Inches(1.5), Inches(1.5), Inches(2.0), Inches(1.5)]
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = width

def parse_use_cases(text):
    """
    解析用例文本
    """
    use_cases = []
    
    # 按用例分割文本
    case_pattern = r'5\.\d+ ([^\n]+)\s+用例名称\s+(.+?)\s+用例描述\s+(.+?)\s+前置条件\s+(.+?)\s+主要流程\s+(.+?)\s+后置条件\s+(.+?)(?=\s+5\.\d+|\Z)'
    cases = re.findall(case_pattern, text, re.DOTALL)
    
    for case in cases:
        use_case = {
            'section': case[0].strip(),
            'name': case[1].strip(),
            'description': case[2].strip(),
            'preconditions': [line.strip() for line in case[3].strip().split('\n') if line.strip() and not line.strip().isdigit()],
            'main_flow': [line.strip() for line in case[4].strip().split('\n') if line.strip() and not line.strip().isdigit()],
            'postconditions': case[5].strip()
        }
        use_cases.append(use_case)
    
    # 特殊处理最后一段文本格式
    if not cases:
        # 如果正则没有匹配到，尝试另一种解析方式
        sections = re.split(r'5\.\d+', text)
        for section in sections:
            if section.strip():
                lines = section.strip().split('\n')
                if len(lines) >= 15:  # 确保有足够的行数
                    use_case = {
                        'section': '',  # 章节标题
                        'name': lines[1].strip() if len(lines) > 1 else '',
                        'description': lines[3].strip() if len(lines) > 3 else '',
                        'preconditions': [line.strip() for line in lines[5:7] if line.strip() and not line.strip().isdigit()],
                        'main_flow': [line.strip() for line in lines[8:13] if line.strip() and not line.strip().isdigit()],
                        'postconditions': lines[14].strip() if len(lines) > 14 else ''
                    }
                    use_cases.append(use_case)
    
    return use_cases

def main():
    # 读取用例文本
    use_case_text = """
5. 用例设计
5.1 特征准入控制用例
用例名称
特征准入控制
用例描述
系统根据特征的历史访问次数决定是否将特征加载到Embedding Cache中
前置条件
2.FeatureFilter已初始化并配置了准入阈值
2. 存在待加载的特征数据
主要流程
3.系统接收特征加载请求
2. FeatureFilter统计特征访问次数
3. FeatureFilter检查特征是否满足准入条件
4. 如果满足准入条件，特征被加载到Embedding Cache
5. 如果不满足准入条件，特征被标记为无效
后置条件
特征根据准入策略被正确处理
5.2 特征淘汰用例
用例名称
特征淘汰
用例描述
系统根据特征的时间戳信息淘汰长时间未使用的特征
前置条件
4.FeatureFilter已初始化并配置了淘汰阈值
2. 系统中存在已加载的特征
主要流程
5.系统定期检查特征使用情况
2. FeatureFilter记录特征时间戳
3. FeatureFilter识别需要淘汰的特征
4. 将待淘汰特征记录到EvictFeatureRecord
5. 在适当时机从Embedding Cache中移除特征
后置条件
长时间未使用的特征被正确淘汰，释放内存空间
5.3 特征记录加载用例
用例名称
特征记录加载
用例描述
系统支持从持久化存储中加载特征访问记录和时间戳记录
前置条件
6.FeatureFilter已初始化
2. 存在持久化的特征记录数据
主要流程
7.系统启动或恢复时加载特征记录
2. FeatureFilter加载特征访问次数记录
3. FeatureFilter加载特征时间戳记录
4. 系统基于加载的记录继续特征过滤操作
后置条件
特征记录被成功加载并可用于后续的过滤操作
5.4 特征计数统计用例
用例名称
特征计数统计
用例描述
系统统计特征的访问次数，用于准入控制决策
前置条件
8.FeatureFilter已初始化并配置了准入阈值
2. 存在携带计数信息的KeyedJaggedTensorWithCount数据
主要流程
9.系统接收携带特征计数信息的KeyedJaggedTensorWithCount数据
2. FeatureFilter通过StatisticsKeyCount方法统计特征访问次数
3. 访问次数信息被存储在featureRecordMap中
4. 后续的准入控制将基于这些统计信息进行决策
后置条件
特征访问次数被正确统计并存储，可用于后续的准入控制
5.5 时间戳处理用例
用例名称
时间戳处理
用例描述
系统记录特征的时间戳信息，用于淘汰长时间未使用的特征
前置条件
10.FeatureFilter已初始化并配置了淘汰阈值
2. 存在携带时间戳信息的KeyedJaggedTensorWithTimestamp数据
主要流程
11.系统接收携带时间戳信息的KeyedJaggedTensorWithTimestamp数据
2. FeatureFilter通过RecordTimestamp方法记录特征时间戳
3. 时间戳信息被存储在timestampRecordMap中
4. 在适当的时机，系统根据时间戳判断是否需要淘汰特征
后置条件
特征时间戳被正确记录并存储，可用于后续的淘汰策略
"""
    
    # 解析用例
    use_cases = parse_use_cases(use_case_text)
    
    # 创建Word文档
    document = Document()
    
    # 设置页面布局
    section = document.sections[0]
    section.page_height = Inches(11)
    section.page_width = Inches(8.5)
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    
    # 创建用例表格
    create_use_case_table(document, use_cases)
    
    # 保存文档
    document.save('use_case_table.docx')
    print("用例表格文档已保存为 use_case_table.docx")

if __name__ == "__main__":
    main()