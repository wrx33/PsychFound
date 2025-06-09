import re
import random
import ast
import operator
import difflib
from ollama import Client

client = Client(
    host='127.0.0.1:11435',
)

ICD_LIST = [
    "F00.0 阿尔茨海默病性痴呆，早发型",
    "F00.1 阿尔茨海默病性痴呆，晚发型",
    "F00.2 阿尔茨海默病性痴呆，非典型或混合型",
    "F00.9 阿尔茨海默病性痴呆，未特指",
    "F01.0 血管性痴呆，急性发作",
    "F01.1 多发脑梗死性痴呆",
    "F01.2 皮层下血管性痴呆",
    "F01.3 混合型皮层和皮层下血管性痴呆",
    "F01.8 其他血管性痴呆",
    "F01.9 血管性痴呆，未特指",
    "F02.0 匹克病性痴呆",
    "F02.1 克雅病性痴呆",
    "F02.2 亨廷顿病性痴呆",
    "F02.3 帕金森病性痴呆",
    "F02.4 人类免疫缺陷病毒[HIV]病性痴呆",
    "F02.8 其他疾病分类中分类的其他疾病引起的痴呆",
    "F03 未特指的痴呆",
    "F04 器质性遗忘综合征，非由酒精和其他精神活性物质引起",
    "F05.0 谵妄，非叠加于痴呆",
    "F05.1 谵妄，叠加于痴呆",
    "F05.8 其他谵妄",
    "F05.9 谵妄，未特指",
    "F06.0 器质性幻觉症",
    "F06.1 器质性紧张性障碍",
    "F06.2 器质性妄想性[精神分裂样]障碍",
    "F06.3 器质性心境[情感]障碍",
    "F06.4 器质性焦虑障碍",
    "F06.5 器质性分离性障碍",
    "F06.6 器质性情绪不稳定[衰弱]障碍",
    "F06.7 轻度认知障碍",
    "F06.8 由脑损害和功能紊乱及躯体疾病引起的其他精神障碍",
    "F06.9 由脑损害和功能紊乱及躯体疾病引起的未特指的精神障碍",
    "F07.0 器质性人格障碍",
    "F07.1 脑炎后综合征",
    "F07.2 脑震荡后综合征",
    "F07.8 由脑疾病、脑损害和脑功能紊乱引起的其他人格和行为障碍",
    "F07.9 由脑疾病、脑损害和脑功能紊乱引起的未特指的人格和行为障碍",
    "F09 未特指的器质性或症状性精神障碍",
    
    "F10.0 急性酒精中毒",
    "F10.1 有害性酒精使用",
    "F10.2 酒精依赖综合征",
    "F10.3 酒精戒断状态",
    "F10.4 酒精戒断状态伴谵妄",
    "F10.5 酒精性精神病性障碍",
    "F10.6 酒精性遗忘综合征[科尔萨科夫综合征]",
    "F10.7 酒精性残留性和迟发性精神病性障碍",
    "F10.8 其他酒精引起的精神和行为障碍",
    "F10.9 酒精引起的精神和行为障碍，未特指",
    "F11-F19 其他精神活性物质引起的精神和行为障碍（结构与F10类似，具体物质不同）",
    
    "F20.0 偏执型精神分裂症",
    "F20.1 青春型精神分裂症",
    "F20.2 紧张型精神分裂症",
    "F20.3 未分化型精神分裂症",
    "F20.4 精神分裂症后抑郁",
    "F20.5 残留型精神分裂症",
    "F20.6 单纯型精神分裂症",
    "F20.8 其他精神分裂症",
    "F20.9 精神分裂症，未特指",
    "F21 分裂型障碍",
    "F22.0 妄想性障碍",
    "F22.8 其他持续性妄想性障碍",
    "F22.9 持续性妄想性障碍，未特指",
    "F23.0 急性多形性精神病性障碍，不伴精神分裂症症状",
    "F23.1 急性多形性精神病性障碍，伴精神分裂症症状",
    "F23.2 急性精神分裂症样精神病性障碍",
    "F23.3 其他急性以妄想为主的精神病性障碍",
    "F23.8 其他急性短暂性精神病性障碍",
    "F23.9 急性短暂性精神病性障碍，未特指",
    "F24 感应性妄想性障碍",
    "F25.0 分裂情感性障碍，躁狂型",
    "F25.1 分裂情感性障碍，抑郁型",
    "F25.2 分裂情感性障碍，混合型",
    "F25.8 其他分裂情感性障碍",
    "F25.9 分裂情感性障碍，未特指",
    "F28 其他非器质性精神病性障碍",
    "F29 未特指的非器质性精神病",
    
    "F30.0 轻躁狂",
    "F30.1 躁狂，不伴精神病性症状",
    "F30.2 躁狂，伴精神病性症状",
    "F30.8 其他躁狂发作",
    "F30.9 躁狂发作，未特指",
    "F31.0 双相情感障碍，当前为轻躁狂发作",
    "F31.1 双相情感障碍，当前为不伴精神病性症状的躁狂发作",
    "F31.2 双相情感障碍，当前为伴精神病性症状的躁狂发作",
    "F31.3 双相情感障碍，当前为轻度或中度抑郁发作",
    "F31.4 双相情感障碍，当前为不伴精神病性症状的重度抑郁发作",
    "F31.5 双相情感障碍，当前为伴精神病性症状的重度抑郁发作",
    "F31.6 双相情感障碍，当前为混合发作",
    "F31.7 双相情感障碍，当前为缓解状态",
    "F31.8 其他双相情感障碍",
    "F31.9 双相情感障碍，未特指",
    "F32.0 轻度抑郁发作",
    "F32.1 中度抑郁发作",
    "F32.2 重度抑郁发作，不伴精神病性症状",
    "F32.3 重度抑郁发作，伴精神病性症状",
    "F32.8 其他抑郁发作",
    "F32.9 抑郁发作，未特指",
    "F33.0 复发性抑郁障碍，当前为轻度发作",
    "F33.1 复发性抑郁障碍，当前为中度发作",
    "F33.2 复发性抑郁障碍，当前为不伴精神病性症状的重度发作",
    "F33.3 复发性抑郁障碍，当前为伴精神病性症状的重度发作",
    "F33.4 复发性抑郁障碍，当前为缓解状态",
    "F33.8 其他复发性抑郁障碍",
    "F33.9 复发性抑郁障碍，未特指",
    "F34.0 环性心境",
    "F34.1 恶劣心境",
    "F34.8 其他持续性心境[情感]障碍",
    "F34.9 持续性心境[情感]障碍，未特指",
    "F38.0 其他单次发作的心境[情感]障碍",
    "F38.1 其他复发性心境[情感]障碍",
    "F38.8 其他特指的心境[情感]障碍",
    "F39 未特指的心境[情感]障碍",
    
    "F40.0 广场恐怖症",
    "F40.1 社交恐怖症",
    "F40.2 特定（孤立）恐怖症",
    "F40.8 其他恐怖性焦虑障碍",
    "F40.9 恐怖性焦虑障碍，未特指",
    "F41.0 惊恐障碍[间歇性发作性焦虑]",
    "F41.1 广泛性焦虑障碍",
    "F41.2 混合性焦虑和抑郁障碍",
    "F41.3 其他混合性焦虑障碍",
    "F41.8 其他特指的焦虑障碍",
    "F41.9 焦虑障碍，未特指",
    "F42.0 以强迫思维或穷思竭虑为主",
    "F42.1 以强迫动作[强迫仪式]为主",
    "F42.2 混合性强迫思维和动作",
    "F42.8 其他强迫性障碍",
    "F42.9 强迫性障碍，未特指",
    "F43.0 急性应激反应",
    "F43.1 创伤后应激障碍",
    "F43.2 适应障碍",
    "F43.8 其他对严重应激的反应",
    "F43.9 对严重应激的反应，未特指",
    "F44.0 分离性遗忘",
    "F44.1 分离性漫游",
    "F44.2 分离性木僵",
    "F44.3 出神与附体障碍",
    "F44.4 分离性运动障碍",
    "F44.5 分离性抽搐",
    "F44.6 分离性感觉麻木和感觉丧失",
    "F44.7 混合性分离[转换]性障碍",
    "F44.8 其他分离[转换]性障碍",
    "F44.9 分离[转换]性障碍，未特指",
    "F45.0 躯体化障碍",
    "F45.1 未分化性躯体形式障碍",
    "F45.2 疑病障碍",
    "F45.3 躯体形式植物神经功能紊乱",
    "F45.4 持续性躯体形式疼痛障碍",
    "F45.8 其他躯体形式障碍",
    "F45.9 躯体形式障碍，未特指",
    "F48.0 神经衰弱",
    "F48.1 人格解体-现实解体综合征",
    "F48.8 其他特指的神经症性障碍",
    "F48.9 神经症性障碍，未特指",
    
    "F50.0 神经性厌食",
    "F50.1 非典型神经性厌食",
    "F50.2 神经性贪食",
    "F50.3 非典型神经性贪食",
    "F50.4 与其他心理紊乱相关的暴食",
    "F50.5 与其他心理紊乱相关的呕吐",
    "F50.8 其他进食障碍",
    "F50.9 进食障碍，未特指",
    "F51.0 非器质性失眠症",
    "F51.1 非器质性嗜睡症",
    "F51.2 非器质性睡眠-觉醒节律障碍",
    "F51.3 睡行症[夜游症]",
    "F51.4 睡惊症[夜惊症]",
    "F51.5 梦魇",
    "F51.8 其他非器质性睡眠障碍",
    "F51.9 非器质性睡眠障碍，未特指",
    "F52.0 性欲缺乏或丧失",
    "F52.1 性厌恶和性乐缺乏",
    "F52.2 生殖器反应丧失",
    "F52.3 性高潮功能障碍",
    "F52.4 早泄",
    "F52.5 非器质性阴道痉挛",
    "F52.6 非器质性性交疼痛",
    "F52.7 性欲亢进",
    "F52.8 其他性功能障碍，非由器质性障碍或疾病引起",
    "F52.9 未特指的性功能障碍，非由器质性障碍或疾病引起",
    "F53.0 与产褥期有关的轻度精神及行为障碍，不可归类在他处者",
    "F53.1 与产褥期有关的重度精神及行为障碍，不可归类在他处者",
    "F53.8 其他与产褥期有关的精神及行为障碍，不可归类在他处者",
    "F53.9 与产褥期有关的精神及行为障碍，未特指",
    "F54 在它处分类的障碍及疾病伴有的心理及行为因素",
    "F55 非依赖性物质滥用",
    "F59 伴有生理紊乱和躯体因素的未特指的行为综合征",
    
    "F60.0 偏执型人格障碍",
    "F60.1 分裂样人格障碍",
    "F60.2 社交紊乱型人格障碍",
    "F60.3 情绪不稳型人格障碍",
    "F60.4 表演型人格障碍",
    "F60.5 强迫型人格障碍",
    "F60.6 焦虑（回避）型人格障碍",
    "F60.7 依赖型人格障碍",
    "F60.8 其他人格障碍",
    "F60.9 人格障碍，未特指",
    "F61 混合型和其他人格障碍",
    "F62.0 持久的人格改变，灾难性经历后",
    "F62.1 持久的人格改变，精神科疾病后",
    "F62.8 其他持久的人格改变",
    "F62.9 持久的人格改变，未特指",
    "F63.0 病理性赌博",
    "F63.1 病理性纵火[纵火癖]",
    "F63.2 病理性偷窃[偷窃癖]",
    "F63.3 拔毛癖",
    "F63.8 其他习惯和冲动障碍",
    "F63.9 习惯和冲动障碍，未特指",
    "F64.0 易性症",
    "F64.1 双重异装症",
    "F64.2 童年性身份障碍",
    "F64.8 其他性身份障碍",
    "F64.9 性身份障碍，未特指",
    "F65.0 恋物症",
    "F65.1 恋物性异装症",
    "F65.2 露阴症",
    "F65.3 窥阴症",
    "F65.4 恋童症",
    "F65.5 施虐受虐症",
    "F65.6 多种性偏好障碍",
    "F65.8 其他性偏好障碍",
    "F65.9 性偏好障碍，未特指",
    "F66.0 性成熟障碍",
    "F66.1 自我不和谐的性取向",
    "F66.2 性关系障碍",
    "F66.8 其他与性发育和性取向有关的心理及行为障碍",
    "F66.9 与性发育和性取向有关的心理及行为障碍，未特指",
    "F68.0 出于心理原因夸大躯体症状",
    "F68.1 有意制造或伪装症状或残疾[人为障碍]",
    "F68.8 其他特指的人格和行为障碍",
    "F69 未特指的成人人格和行为障碍",
    
]

ICD_LIST_wo_code = [re.sub(r'[a-zA-Z0-9\s.]', '', text) for text in ICD_LIST]


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: The solution string generated by the model
        ground_truth: The ground truth solution string. eg. "双相情感障碍"
        method: The scoring method to use. Options: 'strict', 'format'
        format_score: The score to give if the solution is correctly formatted
        score: The score to give if the solution is correct
        
    Returns:
        The score for the solution
    """
    target = ground_truth['diagnosis']
    target_code = target.split(' ')[0]
    patient_info = ground_truth['patient_info']
    solution_str = "<think>" + solution_str


    do_print = random.randint(1, 8) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Patient_info: {patient_info[:100]}")
        print(f"Solution string: {solution_str}")

    # reward 评估内容包含：1. 输出格式是否符合规范。2. 最终的诊断结果是否正确。3. 
    
    # 判断输出格式是否符合规范，即 <think>...</think> <answer>...</answer>
    think_part = ""
    answer_part = ""
    output_format_score = 0
    output_diagnosis_score = 0
    output_readability_score = 0
    try:
        think_part = re.findall(r"<think>(.*?)</think>", solution_str, re.DOTALL)
        answer_part = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)

        if len(re.findall(r"<think>", solution_str, re.DOTALL)) > 1 or len(re.findall(r"</think>", solution_str, re.DOTALL)) > 1 or len(re.findall(r"<answer>", solution_str, re.DOTALL)) > 1 or len(re.findall(r"</answer>", solution_str, re.DOTALL)) > 1:
            print(f"Output format wrong")
            output_format_score += -1
        
        if len(think_part) == 1 and len(answer_part) == 1:
            print(f"Output format correct")
            output_format_score += 1
        # else:
        #     print(f"Output format wrong")
        #     output_format_score += -2
    except:
        print(f"Output format wrong")
        output_format_score += -1
        return -3
    
    final_think = "".join(think_part)
    final_answer = "".join(answer_part)

    if final_think == "" or final_answer == "":
        print(f"Output completely wrong")
        return -3

    对answer部分的输出长度过长施加惩罚
    if len(final_answer) > 40:
        output_format_score += -0.5

    # 对anwer部分后面又继续输出内容施加惩罚
    if len(solution_str.split('</answer>')) > 1 and len(solution_str.split('</answer>')[1]) > 20:
        print(f"Output still after </answer>")
        output_format_score += -1

    # 对think部分中出现的关键词进行奖励
    # 一般的诊断标准包括：症状，病程，严重程度，排除。所以如果think中出现了这些关键字，则可以为其施加奖励
    # 对于一些体现思维过程的关键词也施加一定的奖励，如首先，其次，然后，但是等。
    # reward_keywords1 = ['症状', '病程', '严重程度', '排除', '鉴别']
    # if any(kw in final_think for kw in reward_keywords1):
    #     output_format_score += len([kw for kw in reward_keywords1 if kw in final_think])*0.1
    
    
    # reward_keywords2 = ['首先', '其次', '然后', '但是', '鉴于']
    # if any(kw in final_think for kw in reward_keywords1):
    #     output_format_score += 0.1

    # 对think部分中出现的胡言乱语的现象进行惩罚
    # 观察到的具体现象有：输出一大段没有标点符号且没有逻辑的话，可以通过给两个标点符号之间的长度施加惩罚
    matches_between_punc = re.findall(r'[.,;!?。，、？！:：”“ ](.*?)(?=[.,;!?。，、？！:：”“ ])', final_think)
    for match in matches_between_punc:
        if len(match) > 80:
            print(f"Too long sentence: {match}")
            output_format_score += -0.5
            break
    
    # 尝试通过和参考答案计算相似度，对相似度过低的答案施加较大惩罚，来约束模型不要胡言乱语。只施加惩罚，不施加奖励
    reference = """首先，我们需要分析患者的病程、症状严重程度、病程持续时间和结构上的线索来确定最可能的精神疾病诊断。以下是我的逐步推理过程：
1. **病程和发作的持续时间**：
- 患者从2020年初开始，即13岁左右时出现症状。这表明在青少年时期，尤其是学习压力可能增加的阶段，出现长期的情绪困扰和行为变化。这种长期的发展可能提示精神障碍的长变。
2. **症状严重程度和功能损害**：
- 从诊断标准来看，患者表现出明显的抑郁症状，如兴趣缺失、学习状态差、睡眠问题（夜眠差），以及自伤行为。这些症状的严重程度已经影响到日常生活和学校情况，尤其是在年3月23日发生了用刀片自伤的情况，说明患者的自杀意念可能会加剧风险。
3. **情绪障碍的排除**：
- 患者没有表现出明显的躁狂或双相情感障碍的症状，比如情绪波动在阳性、激情和低落之间变化的模式。因此，排除双相情感障碍的可能性。
- 情绪障碍的评估倾向于短暂的或复发性的抑郁发作，因为症状持续了两年以上，且伴随有自伤行为，这通常与严重抑郁的特征相符。
4. **症状类型和严重程度**：
- 黑质上瘾（k黑上瘾）已经被排除，因为这种症状通常与虚构愉悦感或长期的过度使用物质相关联，而非抑郁或双相情感障碍的症状。此外，患者并未提及有长期的食欲或睡眠模式，除此以外的黑质上瘾特点，如嗜睡、过多睡眠等。
5. **鉴别诊断分析**：
- **抑郁发作**：该疾病的诊断需要考虑是否为轻度、中度或重度抑郁发作，患者的病程表明了严重的抑郁状态（至少持续了两年），因此“重度抑郁发作，不伴精神病性症状”（F32是一个较强的可能性。尤其是考虑到自伤行为通常与重度抑郁有关。
- **双相情感障碍**：虽然患者有长期的情绪困扰，但没有明显的躁狂发作的历史或者迹象，因此可以排除双相情感障碍。
- **其他心境障碍**：复发性抑郁障碍、环性心境、恶性心境等。由于症状持续时间较长，且没有出现缓解期，复发性抑郁症不适合，而恶性心境情绪波动小而长期，通常伴有病情郁的特点，这不符合患者的严重程度。环性心境则没有相关的严重抑郁或缓解期。
6. **鉴别诊断中的排除**：
- **分裂症（精神分裂症）**：患者的病程中伴随了两年多，且在情绪症状并存的情况下（如自伤行为），其不符合单纯型、偏执型或其他分裂症的特征。此外，自我伤害行为通常郁而非前者有关联。
- **反应性精神病**：由于没有明确的压力事件足以解释其长期症状的持续性，排除了由某个特定应激源引发的反应性精神病的可能性。
7. **诊断选择和患者评估**：
- 在这种情况下，“重度抑郁发作，不伴精神病性症状”（F32.2）是最好的选择。这是因为患者的症状严重程度、病程的长期性（超过两年）以及对社会功能的极大损害（自伤行为）合这一诊断标准。此外，精神检查时患者的情绪持续低落，兴趣减退，显著影响了日常生活的多个方面，这些都是重度抑郁发作的核心特征。
8. **排除条件**：
- 鉴别其他可能的精神障碍（如焦虑症、强迫症）时，应注意到患者表现出的情绪障碍的严重性和持续时间，以及是否存在妄想或其他精神病性的症状。缺乏这些特征支持其他诊断，而是倾向于抑郁症。
9. **考虑病程稳定状态**：
- 诊断为“F32.2 重度抑郁发作，不伴精神病性症状”的原因还在于患者的当前症状没有显示出持续的精神病性元素，如怪异的感知或思维内容。然而，长期病程中的功能损害（如学态差）和自残行为表明了较高的严重程度。
10. **鉴别与排除分析**：
- 在排除其他可能的原因（如反应性状态、焦虑症等）后，重度抑郁发作最符合患者的情况。患者的病程和当前症状的严重程度没有其他更可能的解释。
综上所述，患者的病程、症状严重程度、长期影响以及缺乏其他精神障碍的特征支持了重度抑郁发作的诊断，而不符合其他重大考量。因此，诊断应为：F32.2 重度抑郁发作，不伴精神症状。"""

    move = """首先，分析患者的症状。患者表现出情绪低落、消极厌世、行动迟缓等抑郁症状。同时，有反复追究手术失误的负罪感，这可能暗示自责情绪。此外，伴随着多思多虑以及睡眠问题如早醒，这些也符合抑郁症的特征。接下来，看看排除其他可能的诊断。氟西汀和舍曲林是常用的抗抑郁药，对于患者的症状可能是有效的，说明她可能正在经历抑郁症而非焦虑症。另外，没有提及明显的精神分裂症阳性症状，所以排除精神分裂症的可能性。再考虑是否是双相情感障碍，但患者没有提到过躁狂发作，只有抑郁症状，所以排除双相障碍。颅脑疾病、药物滥用或其他物质使用也可能导致这些症状，但患者已接受相关诊断，没有这些情况。总结一下，她的主要问题是情绪低落、兴趣丧失、睡眠问题，这些都是抑郁症的核心症状。此外，伴随的躯体不适和自卑感也符合抑郁症的表现，所以胃肠道症状和幻觉并不是影响诊断的关键，但提示可能存在功能异常或其他因素。 因此，根据ICD-10分类，抑郁症被编码为F32.9或者更详细的编码。在这里，考虑到持续的症状和可能的详细病史记录，最合适的诊断可能为“重度抑郁发作，未特指”（F32.9）。但是，为了更准确，应该考虑患者的抑郁严重程度和伴随的症状来确定具体的亚型或余留编码。"""

    

#     if len(final_think) > 1000:
#         instruction = f"""请判断下面这段文字中是否包含异常的乱码或严重无法理解的语句（轻微的逻辑不通可以忽略）。如果包含则输出0，如果不包含则输出1，不要输出任何其他内容。待判断文字：
# {final_think}"""
        
#         response = client.chat(model='qwen2.5:3b', messages=[
#             {
#                 'role': 'user',
#                 'content': instruction
#             },
#         ],
        
#         # options = {"temperature":0.1}
#         )
#         readability = response['message']['content']
#         if '0' in readability:
#             print(f"Too poor readability:{readability}")
#             output_readability_score += -1
#         elif '1' in readability:
#             print(f"Good readability:{readability}")
#             output_readability_score += 1
    
#     if difflib.SequenceMatcher(None, reference, final_think).ratio() > 0.5:
#         print(f"Good readability:{readability}")
#         output_readability_score += 0.5

    


    # 判断最终诊断结果是否正确
    # 由于模型输出的诊断结果和标准答案术语可能存在差异，但含义一致，所以需要给这种情况施加一定的奖励
    # 首先根据ICD编码进行判断
    answer_code = final_answer.split(' ')[0]
    # code_list = ['F20', 'F21', 'F22', 'F23', 'F24', 'F28', 'F30', 'F31', 'F32', 'F33', 'F34', 'F38']
    # filtered_list = [code for code in code_list if code != target_code.split('.')[0]]

    if target == final_answer:
        print(f"Diagnosis completely correct")
        output_diagnosis_score += 2
    elif target_code in final_answer:# and all(item not in final_answer for item in filtered_list):
        print(f"Diagnosis ICD-code correct")
        output_diagnosis_score += 2
    elif target_code.split('.')[0] in final_answer:# and all(item not in final_answer for item in filtered_list):
        print(f"Diagnosis partly correct: FXX")
        output_diagnosis_score += 2
    elif target_code.split('.')[0][:2] in final_answer:# and all(item not in final_answer for item in filtered_list):
        print(f"Diagnosis partly correct: FX")
        output_diagnosis_score += -2
    # elif difflib.SequenceMatcher(None, target, final_answer).ratio() > 0.5:
    #     print(f"Semantic diagnosis correct")
    #     output_diagnosis_score += 0.1
    else:
        print(f"Diagnosis completely wrong")
        output_diagnosis_score += -2


    
    final_score = output_diagnosis_score + output_format_score + output_readability_score
    print("=="*50)
    print(f"Format Score: {output_format_score}, Readability Score: {output_readability_score}, Diagnosis Score: {output_diagnosis_score}, Final Score: {final_score}")
    print("=="*50)
    return final_score
    
# score = compute_score("我认为诊断结果应该是偏执型人格障碍", {'diagnosis': "双相情感障碍，目前为缓解状态"}, method='strict', format_score=0.1, score=1.)

# print(score)