import sys
sys.path.append('/home/sjtu/wrx/code/LLaMA-Factory-0729/src/llamafactory/graphrag')

import os
import logging
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from nano_graphrag._op import chunking_by_seperators

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# !!! qwen2-7B maybe produce unparsable results and cause the extraction of graph to fail.
WORKING_DIR = "/root/autodl-tmp/nano-graphrag-main/knowledge_drugs"
# MODEL = "qwen2:ctx32k"
MODEL = "qwen2.5:ctx32k-8k"

EMBED_MODEL = SentenceTransformer("/home/sjtu/wrx/code/LLaMA-Factory-0729/src/llamafactory/graphrag/my_embedding_model/custom_embedding_model_1")


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    # -----------------------------------------------------
    return result


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

patient_info = """
患者男性，26岁。主因“间断情绪不稳、冲动、行为怪异2年，复犯1周。”入院。
病例特点：
  1.发病基础：患者性别男，年龄26岁。病前性格:内向、话少。
  2.发病诱因：无明显诱因。
  3.起病形式及病程：慢性起病，间断病程2年。
  4.临床表现：（患者父亲不与患者同住，病史可靠欠详）患者自2020年春节起出现片段的怪异行为。6月后有删除家人联系方式，给家人打很多钱的行为，原因不明。同年8月有行为怪异、作揖下跪、自伤撞石头、紧张害怕、语乱。后冲动、喊叫、恐惧，从机场天台跳下，曾至赣州当地精神科医院住院，诊断“急性而短暂的精神病性障碍”，语乱、行为怪异基本消失，但仍显态度挑剔、蛮横、指责家人、夜不眠。于2020年9月21日至2020年11月6日住院，明确诊断“双相情感障碍，目前为不伴有精神病性症状的躁狂发作”，予以MECT治疗7次，予以丙戊酸钠缓释片日高量1250mg，喹硫平日高量800mg治疗，病情好转出院。出院后按时按量服药，未见异常行为，情绪稳定，工作能力正常，与同事交流可，人际关系正常，情绪稳定，未见异常行为。此后患者认为自己没有病，逐渐减药或者不规律服药。2022年7月6日父亲与患者通话期间，患者语气差显狂妄，对家人不理睬，告诉父亲自己又要病了，称自己一整晚未睡觉，14日家人赶至厦门，患者抱着母亲哭泣，对父亲发脾气，让父亲离自己远一些，不想看到父亲，进食、睡眠不规律，家人在厦门期间患者基本未服药，17日患者冲动拿灭火器喷向父亲，家人将其送入厦门市精神卫生中心，住院5天，考虑诊断“未分化型精神分裂症”，予帕利哌酮缓释片6mg/日、MECT治疗后出院，出院后患者病情改善，坚持每日服药，情绪稳定，未见冲动异常行为，7月23日父亲带其来北京进一步就诊，今日以“双相情感障碍”第二次非自愿收入我科。
  5.既往史：既往患乙型肝炎，目前已无阳性症状
  6.家族史：患者母亲10余年前曾因言语夸大、行为冲动住当地医院1周，诊断不详，经治疗至今虽未服药，亦未复犯。
  7.查体及辅助检查：查体未见异常。


初步诊断：双相情感障碍

鉴别诊断：精神分裂症


"""

patient_info_1 = """
患者欧枝琛 ，男，26岁。主因“间断情绪不稳、冲动、行为怪异2年，复犯1周。”入院。
病例特点：
  1.发病基础：患者性别男，年龄26岁。病前性格:内向、话少。
  2.发病诱因：无明显诱因。
  3.起病形式及病程：慢性起病，间断病程2年。
  4.临床表现：（患者父亲不与患者同住，病史可靠欠详）患者自2020年春节起出现片段的怪异行为。6月后有删除家人联系方式，给家人打很多钱的行为，原因不明。同年8月有行为怪异、作揖下跪、自伤撞石头、紧张害怕、语乱。后冲动、喊叫、恐惧，从机场天台跳下，曾至赣州当地精神科医院住院，诊断“急性而短暂的精神病性障碍”，语乱、行为怪异基本消失，但仍显态度挑剔、蛮横、指责家人、夜不眠。于2020年9月21日至2020年11月6日住院，明确诊断“双相情感障碍，目前为不伴有精神病性症状的躁狂发作”，予以MECT治疗7次，予以丙戊酸钠缓释片日高量1250mg，喹硫平日高量800mg治疗，病情好转出院。出院后按时按量服药，未见异常行为，情绪稳定，工作能力正常，与同事交流可，人际关系正常，情绪稳定，未见异常行为。此后患者认为自己没有病，逐渐减药或者不规律服药。2022年7月6日父亲与患者通话期间，患者语气差显狂妄，对家人不理睬，告诉父亲自己又要病了，称自己一整晚未睡觉，14日家人赶至厦门，患者抱着母亲哭泣，对父亲发脾气，让父亲离自己远一些，不想看到父亲，进食、睡眠不规律，家人在厦门期间患者基本未服药，17日患者冲动拿灭火器喷向父亲，家人将其送入厦门市精神卫生中心，住院5天，考虑诊断“未分化型精神分裂症”，予帕利哌酮缓释片6mg/日、MECT治疗后出院，出院后患者病情改善，坚持每日服药，情绪稳定，未见冲动异常行为，7月23日父亲带其来北京进一步就诊，今日以“双相情感障碍”第二次非自愿收入我科。
  5.既往史：既往患乙型肝炎，目前已无阳性症状
  6.家族史：患者母亲10余年前曾因言语夸大、行为冲动住当地医院1周，诊断不详，经治疗至今虽未服药，亦未复犯。
  7.查体及辅助检查：查体未见异常。
拟诊讨论：
诊断及诊断依据：
  1.病程标准：慢性起病，间断病程。
  2.症状学标准：患者意识清晰，定向力完整，接触显被动，对问话能答，对答尚切题，语量语速适中。承认既往存在幻觉妄想，目前未引出明确精神性性症状，承认近期夜眠差，睡眠需求减少，脑子想到过去的事情，想到父亲对母亲不好，情绪激动，存在冲动行为，有求治意愿，对将来有一定的规划，高级意向部分存在，自知力部分存在。
  3.严重程度标准：严重影响社会功能。
  4.排除标准：
排除脑器质性及精神活性物质所致精神和行为障碍。
  5.初步诊断：双相情感障碍,乙型病毒性肝炎表面抗原携带者
鉴别诊断：
1.精神分裂症：患者病史中存在语乱、行为怪异表现，需要考虑该病，患者病史中情绪问题占主导地位，反复追问下无明确妄想症状，故考虑该病可能性不大。
诊疗计划：
  1.完善辅助检查，急查血常规、血急化、尿常规；
  2.精神科一级护理；严防冲动；
  3.延续院外帕利哌酮缓释片，合并丙戊酸钠缓释片及碳酸锂稳定情绪，劳拉西泮抗焦虑、助眠，盐酸苯海索片对抗锥体外系反应；
  4.向家属交代各注意事项，签署相关知情同意书；
  5.采用自杀风险、攻击风险评估量表判断患者是否存在自杀观念及攻击倾向及其严重程度；采用PANSS观察患者是否存在精神病性症状及评定阳性与阴性症状的程度变化；采用锥体外系反应量表及药物副反应量表细致评定治疗过程中的药物副反应；采用躁狂量表、汉密尔顿抑郁量表、汉密尔顿焦虑量表评定患者是否存在相关症状；
  6.请上级医师查房，明确诊断，指导治疗。

"""

# """
# 拟诊讨论：
# 诊断及诊断依据：
#   1.病程标准：慢性起病，间断病程。
#   2.症状学标准：患者意识清晰，定向力完整，接触显被动，对问话能答，对答尚切题，语量语速适中。承认既往存在幻觉妄想，目前未引出明确精神性性症状，承认近期夜眠差，睡眠需求减少，脑子想到过去的事情，想到父亲对母亲不好，情绪激动，存在冲动行为，有求治意愿，对将来有一定的规划，高级意向部分存在，自知力部分存在。
#   3.严重程度标准：严重影响社会功能。
#   4.排除标准：
# 排除脑器质性及精神活性物质所致精神和行为障碍。
# 鉴别诊断：
# 1.精神分裂症：患者病史中存在语乱、行为怪异表现，需要考虑该病，患者病史中情绪问题占主导地位，反复追问下无明确妄想症状，故考虑该病可能性不大。
# 诊疗计划：
#   1.完善辅助检查，急查血常规、血急化、尿常规；
#   2.精神科一级护理；严防冲动；
#   3.延续院外帕利哌酮缓释片，合并丙戊酸钠缓释片及碳酸锂稳定情绪，劳拉西泮抗焦虑、助眠，盐酸苯海索片对抗锥体外系反应；
#   4.向家属交代各注意事项，签署相关知情同意书；
#   5.采用自杀风险、攻击风险评估量表判断患者是否存在自杀观念及攻击倾向及其严重程度；采用PANSS观察患者是否存在精神病性症状及评定阳性与阴性症状的程度变化；采用锥体外系反应量表及药物副反应量表细致评定治疗过程中的药物副反应；采用躁狂量表、汉密尔顿抑郁量表、汉密尔顿焦虑量表评定患者是否存在相关症状；
#   6.请上级医师查房，明确诊断，指导治疗。
# """

instruction = "请根据以下患者病历信息，对患者的初步诊断结果和鉴别诊断结果进行详细解释，并提供所参考的理论依据。"
my_query = f"{instruction}。\n患者病历信息：{patient_info}"

my_query_1 = f"""
### 任务说明：
您是一名精神医学人工智能助手，专注于解析和补充患者病历中涉及医生决策的内容，包括但不限于病例特点总结、诊断及鉴别诊断逻辑、治疗方案选择、检查项目选择依据等。您的目标是对病历中涉及医生思维过程的部分进行详细补充，结合临床指南及医学知识，说明医生每项决策背后的理论依据和逻辑推理。

### 输入内容：患者病历信息：{patient_info_1}

### 任务目标：
请对病历中所有涉及医生决策的部分，逐一补充其背后的理论依据和推理过程，具体包括以下内容：
1. **病例特点总结**：  
   - 结合患者病历，解释医生如何总结患者的主要临床特点，并如何根据病史、症状及其他信息筛选关键特征。  
2. **诊断逻辑**：  
   - 对医生给出的诊断结果，补充其依据的标准和推理过程。需结合病史、临床表现、辅助检查结果及相关指南，详细说明医生如何得出最终诊断。  
3. **鉴别诊断**：  
   - 对医生记录的鉴别诊断过程进行扩展，补充医生可能考虑的其他疾病及排除这些疾病的具体依据。  
4. **检查项目选择依据**：  
   - 说明医生为何选择特定的辅助检查或评估工具，以及这些检查在诊断和治疗中的作用。  
5. **治疗方案选择**：  
   - 解析医生选择特定药物或治疗手段的理论依据，包括药物的作用机制、适应症、替代方案及潜在风险。  
6. **其他决策的补充说明**：  
   - 对病历中未明确记录的其他决策，结合背景信息和参考知识进行合理推测和补充。

### 输出要求：
- 输出格式不限，可以以段落形式或条目形式呈现。
- 内容需逻辑清晰、推理严密、表达自然。
- 所有补充内容必须基于输入病历和参考知识，避免虚构或与输入矛盾。


"""



def query(kb, question):

    dir_kb = f'/home/sjtu/wrx/code/LLaMA-Factory-0729/src/llamafactory/graphrag/{kb}'

    rag = GraphRAG(
        working_dir=dir_kb,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=local_embedding,
        chunk_func=chunking_by_seperators,
        chunk_token_size= 4096,
        chunk_overlap_token_size= 0,
        enable_naive_rag=True,
    )
    # 使用利培酮治疗双向情感障碍的推荐起始剂量和理想剂量
    # 利培酮口服溶液的老年用药治疗精神分裂症的建议起始剂量
    
    # while True:
    #     print(rag.query(
    #         input("Enter your query: "), 
    #         param=QueryParam(mode="local", only_need_context=True)
    #     ))
    answer = rag.query(
        question, 
        param=QueryParam(mode="naive")
    )
    print(
        answer
    )
    
    reference = rag.query(
        question, 
        param=QueryParam(mode="naive", only_need_context=True)
    )
    print(
        reference
    )

    return answer, reference

    
    # print(
    #     rag.query(
    #         "奥氮平的不良反应", 
    #         param=QueryParam(mode="naive")
    #     )
    # )
    
    # print(
    #     rag.query(
    #         "奥氮平的不良反应", 
    #         param=QueryParam(mode="naive", only_need_context=True)
    #     )
    # )
    
    # print(
    #     rag.query(
    #         "丙戊酸钠的用法用量", 
    #         param=QueryParam(mode="naive")
    #     )
    # )
    
    # print(
    #     rag.query(
    #         "丙戊酸钠的用法用量", 
    #         param=QueryParam(mode="naive", only_need_context=True)
    #     )
    # )
    
    
    # print(
    #     rag.query(
    #         "氯硝西泮注射液的用法用量", 
    #         param=QueryParam(mode="naive")
    #     )
    # )    
    
    # print(
    #     rag.query(
    #         "卡马西平的用法用量", 
    #         param=QueryParam(mode="naive")
    #     )
    # )
    
    # print(
    #     rag.query(
    #         "下一步诊疗计划：患者氟西汀浓度不足，将氟西汀加至30mg，观察患者用药情况，注意药物不良反应。", 
    #         param=QueryParam(mode="naive")
    #     )
    # )
    
    # print(
    #     rag.query(
    #         """
    #         不良反应：无
    #         体格检查：未见明显异常
    #         精神检查：患者表现意识清晰，定向力完整，接触合作，问话能切题回答。称今天的心情很稳定，不好也不坏。心烦的情绪较昨天有所缓解。患者在病房内多独处，未见冲动伤人行为，情感反应协调，自知力部分。
    #         辅助检查：2023-01-11 奥氮平血药浓度,氟西汀＋去甲氟西汀血药浓度(样本:血清):氟西汀+去甲氟西汀45.51ng/ml↓,患者氟西汀浓度不足，将氟西汀加至30mg，观察患者用药情况，注意药物不良反应。2023-01-12，心电图：窦性心律过缓窦性心律不齐大致正常心电图
    #         下一步诊疗计划：患者氟西汀浓度不足，将氟西汀加至30mg，观察患者用药情况，注意药物不良反应。
    #         """, 
    #         param=QueryParam(mode="naive")
    #     )
    # )

def split_text_into_patches(text, max_length=20000):
    patches = []
    for i in range(0, len(text), max_length):
        patches.append(text[i:i+max_length])
    return patches

def insert(path_kb, dir_kb):
    from time import time
    
    remove_if_exist(f"{dir_kb}/vdb_entities.json")
    remove_if_exist(f"{dir_kb}/vdb_chunks.json")
    remove_if_exist(f"{dir_kb}/kv_store_full_docs.json")
    remove_if_exist(f"{dir_kb}/kv_store_text_chunks.json")
    remove_if_exist(f"{dir_kb}/kv_store_community_reports.json")
    remove_if_exist(f"{dir_kb}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=dir_kb,
        enable_llm_cache=True,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=local_embedding,
        chunk_func=chunking_by_seperators,
        chunk_token_size= 4096,
        chunk_overlap_token_size= 0,
        enable_naive_rag=True,
    )
    start = time()
    # rag.insert(FAKE_TEXT)
    # print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])

    kb_list = os.listdir(path_kb)
    for kb in kb_list:
        print(kb)
        FAKE_TEXT = ""
        with open(os.path.join(path_kb, kb), encoding="utf-8-sig") as f:
            FAKE_TEXT = f.read()
        f.close()
        
        # patches = split_text_into_patches(FAKE_TEXT)
        
        split_by_drugs = FAKE_TEXT.split('\n\n')
        
        cnt = 0
        for patch in split_by_drugs:
            print(patch.split('\n')[0])
            patch = patch.replace('\n', '###')
            rag.insert(patch)
            print("indexing time:", time() - start)
            # if cnt > 20:
            #     break
            # cnt += 1
        # rag.insert(FAKE_TEXT)
    
    # with open(r"/root/autodl-tmp/nano-graphrag-main/kb/沈渔邨第十三章.txt", encoding="utf-8-sig") as f:
    #     FAKE_TEXT = f.read()
        
    #     # patch_len = len(FAKE_TEXT) // 10
    #     # for i in range(10):
    #     #     rag.insert(FAKE_TEXT[i*patch_len:(i+1)*patch_len])
    # rag.insert(FAKE_TEXT)
    # print("indexing time:", time() - start)

    


if __name__ == "__main__":
    path_knowledge = '/root/autodl-tmp/nano-graphrag-main/knowledge_drugs_psychiatry'
    dir_kb = '../kb_drugs_psychiatry_sentence_naive'
    os.makedirs(dir_kb, exist_ok=True)
    
    # insert(path_knowledge, dir_kb)
    query(dir_kb)
    
    # path_knowledge = '/root/autodl-tmp/nano-graphrag-main/knowledge_guidelines'
    # dir_kb = '/root/autodl-tmp/nano-graphrag-main/kb_guidelines'
    # insert(path_knowledge, dir_kb)
    
    # path_knowledge = '/root/autodl-tmp/nano-graphrag-main/knowledge_books'
    # dir_kb = '/root/autodl-tmp/nano-graphrag-main/kb_books'
    # insert(path_knowledge, dir_kb)
    
    
    
