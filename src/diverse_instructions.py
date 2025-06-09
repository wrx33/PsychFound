import json
import os
import numpy as np
import random
from anonymization import get_anony_names

doctor_names, patient_names = get_anony_names()

def anony(content):
    for name in doctor_names:
        content = content.replace(name, "")
    
    for name in patient_names:
        content = content.replace(name, "")
    
    return content

# Task1:summarization
instructions_task1 = [
    "根据以下患者的详细病历信息，提炼主诉、病史概要及标准化的病例特点。",
    "请阅读以下患者病历记录，并总结主诉、主要病史以及临床病例特点。",
    "请基于患者的住院病历，按要求总结以下内容：主诉、病史简要和病例特点。",
    "结合提供的病历信息，总结患者的核心病情表现和病例特点。",
    "从以下患者病历中提取诊断相关信息，包括患者主诉、关键病史和病例特点。",
    "根据患者的完整病历，提炼以下几点：主诉、病史要点和病例特点。",
    
    "精神科医生：根据以下住院病历内容，生成病例总结，包括患者的主诉、简要病史及病例特点。",
    "以下是患者的详细病历记录，请为病例讨论会准备一份简要总结，重点包括主诉、病史概要及病例特点。",
    "患者住院记录如下，请帮助总结核心内容用于电子病历，包括主诉、病史摘要和病例特点。",
    "这是患者的住院病历，请提取主要信息生成一份病例总结，用于交班报告。",
    "阅读病历后，生成一份诊疗概要报告，内容包括主诉、病史重点和病例特点。",
    
    """根据患者提供的病历内容，完成以下任务：
        - 提炼患者的主诉。
        - 总结病史中的重要内容。
        - 提取病例特点（包括病程、症状、严重程度和排除标准）。""",
    """请按以下步骤分析患者病历：
        - 记录患者主诉。
        - 总结现病史和既往病史中的核心内容。
        - 提炼病例特点，涵盖病程、症状、严重程度及排除标准。""",
    """请阅读以下病历信息，并逐项总结以下内容：
        - 患者的主诉。
        - 简要的病史（现病史、既往史、家族史等）。
        - 标准化病例特点（包括病程、症状学、严重程度和排除标准）。""",
    """病历信息如下，请完成以下工作：
        - 提取患者主要症状和病史重点。
        - 按以下标准总结病例特点：
            - 病程标准
            - 症状学标准
            - 严重程度标准
            - 排除标准""",

    """根据以下详细病历记录，完成以下内容的总结：
        - 提炼患者的主诉。
        - 总结关键病史信息。
        - 按四个标准总结病例特点（病程标准、症状学标准、严重程度标准和排除标准）。""",
    """患者病历如下，请提炼主诉，结合病史记录生成简要总结，并按以下标准梳理病例特点：
        - 病程：症状持续的时间、阶段和变化。
        - 症状：具体的精神症状、行为表现等。
        - 严重程度：对患者和家人的实际影响。
        - 排除标准：不支持该诊断的排除依据。""",
    """请将以下病历内容总结为诊断报告，内容包括：
        - 主诉（患者当前的核心问题）。
        - 病史概要（现病史、既往史及其他相关信息）。
        - 病例特点（结合病程、症状、严重程度及排除标准）。""",

    """详细病历记录如下，请从中提炼以下几点：
        - 主诉：患者描述的主要症状。
        - 病史概要：包括现病史、既往史及其他重要信息。
        - 病例特点：按病程、症状、严重程度和排除标准分类总结。""",
    """以下是患者病历内容，请从以下方面进行总结：
        - 主诉：即患者的核心诉求或主要症状。
        - 病史：提取现病史、既往史和家族史中的重要信息。
        - 病例特点：结合患者病史及检查结果，归纳病程、症状和严重程度。""",
    """请将以下病历内容总结为病例分析报告，具体包括：
        - 患者的主诉。
        - 病史要点（现病史、既往史及其他相关记录）。
        - 临床特点（按病程、症状、严重程度和排除标准分类）。""",
]

# Task2:diagnosis
instructions_task2 = [
    "根据以下患者病历信息，给出明确的精神疾病诊断结论，并完成支持诊断的分析及鉴别诊断。",
    "请根据患者病历内容，提供诊断结论、支持诊断的分析及鉴别诊断的说明。",
    "阅读以下病历记录，诊断患者可能的精神疾病，并分析诊断依据，同时完成鉴别诊断。",
    "结合患者的详细病历内容，确定精神疾病诊断结论，分析支持诊断的证据，并进行鉴别诊断。",
    "请根据患者的完整病历，给出诊断结论并完成详细的鉴别诊断分析。",
    
    """作为精神科医生，请根据患者的病历内容完成以下任务：
        - 提出明确的诊断结论。
        - 提供支持诊断的分析。
        - 进行必要的鉴别诊断。""",
    "以下是患者的详细病历记录，请为临床讨论会准备诊断报告，包括诊断结论、支持分析及鉴别诊断。",
    "患者入院病历如下，请帮助完成精神疾病诊断分析，并提供鉴别诊断的依据。",
    "请基于患者的住院病历内容，为病例讨论会议生成诊断总结和鉴别诊断报告。",
    "请以精神科医生的身份对以下病历内容进行分析，明确诊断，支持诊断依据，并完成鉴别诊断。",
    
    """根据以下病历内容，逐步完成以下任务：
        - 提出明确的精神疾病诊断结论。
        - 分析支持诊断的核心依据。
        - 完成鉴别诊断，说明排除其他疾病的理由。""",
    """请阅读以下病历信息，并完成以下步骤：
        - 给出诊断结论。
        - 提取支持该诊断的病历依据。
        - 提供详细的鉴别诊断，排除其他可能的诊断。""",
    """以下是患者的病历记录，请分以下步骤进行诊断分析：
        诊断明确的精神疾病。
        提供支持诊断的具体分析，包括病史、症状和检查结果。
        列出可能需要鉴别的其他诊断，并说明排除理由。""",
    """患者的病历内容如下，请完成以下任务：
        根据患者的病历内容明确诊断。
        结合患者病程和临床表现，提供支持诊断的分析。
        逐项分析可能的鉴别诊断，并说明排除依据。""",
        
    """请根据患者的详细病历内容，完成以下工作：
        确定诊断结论（例如未分化型精神分裂症）。
        提供详细的诊断依据，包括症状、病程及辅助检查结果。
        提出需要考虑的鉴别诊断，并分析如何排除其他可能疾病。""",
    """以下是患者的住院病历，请完成精神疾病诊断和鉴别诊断分析，包括：
        - 明确诊断（例如双相情感障碍抑郁发作）。
        - 提出支持诊断的病史、临床表现及辅助检查依据。
        - 提供鉴别诊断，说明如何排除其他可能的疾病。""",
    """患者病历如下，请完成诊断分析报告，具体包括：
        - 诊断结论。
        - 支持诊断的临床和检查依据。
        - 鉴别诊断及其排除分析。""",
        
    """根据以下病历内容，完成以下内容：
        - 明确精神疾病的诊断，并提供详细的支持依据。
        - 完成与其他疾病的鉴别诊断，并说明排除理由。""",
    """以下是患者的住院记录，请根据病史和检查结果完成以下任务：
        - 提出诊断结论。
        - 提供支持该诊断的病程、症状及检查依据。
        - 完成鉴别诊断，说明如何得出最终诊断。""",
    """请分析以下病历内容，并生成诊断报告，报告内容包括：
        精神疾病诊断（例如精神分裂症）。
        支持诊断的分析（病史、症状及其他依据）。
        鉴别诊断和排除分析。""",
    
    "请根据以下病历信息，诊断患者的精神疾病（如偏执型精神分裂症），分析支持诊断的依据，并说明与其他精神疾病的区别。",
    "患者病历如下，请完成诊断分析，包括明确诊断、支持诊断的病史依据及鉴别诊断。",
    """请基于患者详细的病历内容，完成诊断报告，报告内容包括：
        - 精神疾病诊断。
        - 支持诊断的具体分析。
        - 必要的鉴别诊断说明。""",
]

# Task3:treatment
instructions_task3 = [
    "根据以下患者信息，制定详细的诊疗计划，包括检查、护理、药物治疗及家属沟通等内容。",
    "请根据患者的临床信息，设计一个全面的诊疗计划，涵盖检查、治疗及护理措施。",
    "阅读以下患者信息，为患者提供针对性的诊疗计划，具体包括药物选择、护理措施及必要的辅助检查。",
    "结合患者当前病情，制定诊疗计划，包括检查项目、治疗建议及风险评估措施。",
    "请根据以下患者的病历信息，提供合理的诊疗计划，重点关注治疗目标和安全措施。",
    
    "作为精神科主治医师，请根据以下患者信息制定初步的诊疗计划，包括检查项目、用药方案、护理措施及家属教育内容。",
    """以下是患者的临床资料，请为其制定一份全面的诊疗计划，内容需涵盖：
        检查和评估项目
        治疗目标与具体措施
        护理方案及注意事项
        家属沟通建议""",
    """患者当前症状如下，请为住院期间制定诊疗计划，需包含以下内容：
        - 精神科专科护理措施
        - 药物治疗建议
        - 检查项目安排及评估方法
        - 家属教育计划""",
    "请以临床医生的身份，根据以下患者的病史和当前表现，设计一个多维度的诊疗计划，包括治疗策略和风险评估。",
    "阅读以下患者信息，请为其提供一个完整的诊疗计划，覆盖检查、治疗、护理和随访建议。",
    
    """根据以下患者信息，逐步完成以下任务：
        列出需要完善的检查和评估项目。
        制定针对患者当前症状的药物治疗计划。
        提出护理建议及风险管理策略。
        提供家属教育和随访计划。""",
    """请根据以下患者信息，完成以下任务：
        - 拟定必要的辅助检查项目及评估量表。
        - 提供合理的药物治疗方案。
        - 制定详细的护理计划和家属沟通策略。""",
    """患者当前病情如下，请分步骤完成诊疗计划的制定：
        明确诊疗目标。
        制定完善的检查计划。
        提出具体的治疗方案，包括药物及非药物干预措施。
        安排护理和家属沟通计划。""",
    """结合以下患者信息，请制定一个包括以下内容的诊疗计划：
        - 诊疗目标及治疗方向。
        - 检查和评估量表建议。
        - 药物治疗及其他干预措施。
        - 护理和安全措施。
        - 家属教育与随访安排。""",

    """请根据患者当前的病历信息，完成诊疗计划，具体包括：
        - 检查和评估项目：列出所有必要的辅助检查及评估量表。
        - 药物治疗方案：明确用药种类、剂量及目标。
        - 护理计划：制定住院期间的护理措施和注意事项。
        - 风险管理：说明自杀风险及攻击风险评估措施。
        - 家属沟通：设计家属教育内容及知情同意流程。""",
    """患者病情如下，请为其制定全面的诊疗计划，包括：
        - 完善检查项目。
        - 制定药物和非药物治疗策略。
        - 设计护理及安全管理措施。
        - 提出家属教育及随访计划。""",
    """以下是患者的住院记录，请制定一份具体的诊疗计划，包含以下内容：
        检查和评估：需要完成哪些检查及评估项目。
        治疗措施：药物治疗的种类、剂量及疗效评估方法。
        护理与安全：制定护理措施及风险防控策略。
        家属沟通与教育：如何向家属交代注意事项并签署知情同意书。""",

    """根据以下患者的病历内容，完成以下内容：
        - 设计检查计划：包括必要的辅助检查及评估量表。
        - 药物治疗：提出合理的药物选择及治疗方案。
        - 护理计划：针对患者的护理需求制定详细的措施。
        - 风险评估：分析患者的自杀或攻击风险并提出管理策略。""",
    """阅读患者当前的病历信息，请完成以下任务：
        - 给出具体的检查项目安排。
        - 提供药物治疗方案及疗效评估策略。
        - 制定护理计划及安全防范措施。
        - 安排家属教育与沟通计划。""",
    """请根据以下患者信息，生成诊疗计划，需包含：
        精神科特级护理及风险评估。
        用药建议及剂量调整方案。
        检查和量表评估的具体安排。
        向家属交代的注意事项及随访计划。""",
]

# Task4:management
instructions_task4 = [
    "根据以下患者的病程记录，给出当日诊疗安排。",
    "请阅读患者的病程记录，结合检查结果和临床表现，制定当日诊疗计划。",
    "根据患者的当日病情进展，提供下一步诊疗安排，重点说明异常指标的应对措施。",
    "请结合以下患者的检查和临床数据，提出当日的治疗和护理调整建议。",
    "患者病程记录如下，请基于当前病情给出具体的诊疗计划调整。",
    
    "患者当前病程记录如下，请作为主管医生制定今日的诊疗计划，包括用药调整、检查安排及护理策略。",
    """以下是患者的住院记录和检查数据，请制定今日诊疗安排，需涵盖：
        异常指标的处理方案
        重要药物的调整计划
        辅助检查的复查安排
        护理措施更新""",
    "结合以下患者住院期间的最新病程记录，制定今日诊疗计划，重点关注危险值指标和临床表现的改善情况。",
    "作为病区医生，请根据以下病程记录更新诊疗计划，明确调整的治疗措施和监测方案。",
    "患者病情进展如下，请为今日诊疗安排提供指导，包括检查、用药和护理重点。",
    
    """根据患者以下病程记录，完成以下任务：
        列出需重点关注的异常检查指标及处理方案。
        明确当日药物治疗调整建议。
        制定护理和风险管理计划。""",
    """请根据以下患者病程记录，逐步完成诊疗安排的制定：
        异常指标的监测和应对。
        药物治疗方案的调整。
        辅助检查复查计划。
        护理措施更新及安全防护。""",
    """根据患者当日病情进展，完成以下任务：
        说明当日需要调整或停止的药物。
        列出需要复查的辅助检查项目。
        制定患者的护理重点和安全管理策略。""",
    """患者住院期间病情进展如下，请分步骤完成诊疗安排：
        结合病情和检查结果调整治疗方案。
        针对异常检查指标提出应对措施。
        明确当日护理目标和随访重点。""",

    """阅读以下患者的住院病程记录，请逐步完成当日诊疗安排，需包括以下内容：
        异常指标处理：列出需重点监测的指标，并给出相应的调整或处理措施。
        药物治疗调整：说明需更改或维持的药物及治疗方案。
        辅助检查复查：根据病情安排后续检查。
        护理更新：根据患者当前状况制定护理重点。
        风险管理：判断患者当前是否存在高风险情况，并提出预防措施。""",
    """患者病情记录如下，请制定当日诊疗安排，具体包括：
        异常检查指标的应对措施。
        药物治疗方案调整建议。
        辅助检查和评估量表的复查计划。
        护理和安全防控措施。""",
    """请根据以下患者的病程记录，完成今日诊疗安排：
        明确异常指标的临床意义及应对策略。
        说明药物治疗的调整计划及目标。
        制定当日护理和观察要点。
        安排必要的检查及随访计划。""",
        
    "以下是患者的初始病程记录，请给出诊疗安排，并记住患者信息以便后续调整。",
    "患者当前病程记录如下，请制定今日的诊疗安排，并在后续对话中随时根据新增信息调整计划。",
    "患者住院期间每日病情进展如下，请针对当前病情制定诊疗计划，并保持连续对话以跟进调整。",
]

path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X/task1-summary.jsonl'
path2 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X/task2-diagnosis.jsonl'
path3 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X/task3-treatment.jsonl'
# path4 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F2X/task4-management.jsonl'

data1 = []
with open(path1, 'r') as file1:
    for line in file1:
        newline = eval(line)
        data1.append(newline)
file1.close()

data2 = []
with open(path2, 'r') as file2:
    for line in file2:
        newline = eval(line)
        data2.append(newline)
file2.close()
     
data3 = []
with open(path3, 'r') as file3:
    for line in file3:
        newline = eval(line)
        data3.append(newline)
file3.close()
        
# data4 = []
# with open(path4, 'r') as file4:
#     for line in file4:
#         newline = eval(line)
#         data4.append(newline)
# file4.close()

exist_in_data1 = False
exist_in_data2 = False
exist_in_data3 = False
# exist_in_data4 = False

selected_data1 = []
selected_data2 = []
selected_data3 = []
# selected_data4 = []

cnt = 0
for item in data1:
    ipid = item['conversations'][0]['value']
    
    for item2 in data2:
        if item2['conversations'][0]['value'] == ipid:
            exist_in_data2 = True
            break
    
    for item3 in data3:
        if item3['conversations'][0]['value'] == ipid:
            exist_in_data3 = True
            break
    
    # for item4 in data4:
    #     if item4['conversations'][0]['value'] == ipid:
    #         exist_in_data4 = True
    #         break
    
    
    if exist_in_data2 == True and exist_in_data3 == True:# and exist_in_data4 == True:
        print(cnt)
        # current_his = re.search(r'简要病史：(.*?)(?=查体及精神检查)', item['conversations'][1]['value'], re.DOTALL).group(1).strip()
   

        input1 = "{}\n 患者信息：{}\n".format(random.choice(instructions_task1), item['conversations'][1]['value'])
        input1 = anony(input1)
        label1 = item['conversations'][2]['value']
        label1 = anony(label1)
        conversation1 = []
        conversation1.append({'from':'human','value': input1})
        conversation1.append({'from':'gpt','value': label1})
        selected_data1.append({'conversations': conversation1})
        
        input2 = "{}\n 患者信息：{}\n".format(random.choice(instructions_task2), item2['conversations'][1]['value'])
        input2 = anony(input2)
        label2 = item2['conversations'][2]['value']
        label2 = anony(label2)
        conversation2 = []
        conversation2.append({'from':'human','value': input2})
        conversation2.append({'from':'gpt','value': label2})
        selected_data2.append({'conversations': conversation2})
        
        input3 = "{}\n 患者信息：{}\n".format(random.choice(instructions_task3), item3['conversations'][1]['value'])
        input3 = anony(input3)
        label3 = item3['conversations'][2]['value']
        label3 = anony(label3)
        conversation3 = []
        conversation3.append({'from':'human','value': input3})
        conversation3.append({'from':'gpt','value': label3})
        selected_data3.append({'conversations': conversation3})

        # input4 = "{}\n 患者信息：{}\n".format(random.choice(instructions_task4), item4['conversations'][1]['value'])
        # input4 = anony(input4)
        # label4 = item4['conversations'][2]['value']
        # label4 = anony(label4)
        # conversation4 = []
        # conversation4.append({'from':'human','value': input4})
        # conversation4.append({'from':'gpt','value': label4})

        # for i in range(3, len(item4['conversations']), 2):
        #     if i < (len(item4['conversations'])-1):
        #         conversation4.append({'from':'human','value': anony(item4['conversations'][i]['value'])})
        #         conversation4.append({'from':'gpt','value': anony(item4['conversations'][i+1]['value'])})
        # selected_data4.append({'conversations': conversation4})
        
        
        cnt += 1
        exist_in_data1 = False
        exist_in_data2 = False
        exist_in_data4 = False

os.makedirs('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X_instruct', exist_ok=True)

with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X_instruct/task1-summary.jsonl', 'w') as file1:
    for item in selected_data1:
        # 将每个字典转换为JSON字符串并写入文件
        json_str = json.dumps(item, ensure_ascii=False)
        file1.write(json_str + '\n')
file1.close()


with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X_instruct/task2-diagnosis.jsonl', 'w') as file2:
    for item in selected_data2:
        # 将每个字典转换为JSON字符串并写入文件
        json_str = json.dumps(item, ensure_ascii=False)
        file2.write(json_str + '\n')
file2.close()


with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X_instruct/task3-treatment.jsonl', 'w') as file3:
    for item in selected_data3:
        # 将每个字典转换为JSON字符串并写入文件
        json_str = json.dumps(item, ensure_ascii=False)
        file3.write(json_str + '\n')
file3.close()


# with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F2X_instruct/task4-management.jsonl', 'w') as file4:
#     for item in selected_data4:
#         # 将每个字典转换为JSON字符串并写入文件
#         json_str = json.dumps(item, ensure_ascii=False)
#         file4.write(json_str + '\n')
# file4.close()