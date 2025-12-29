from openai_concurrent import OpenAIConcurrentClient
from pathlib import Path
from typing import List, Dict, Literal, Optional
import base64
from string import ascii_uppercase
import re
from itertools import product
import logging
from config import *


# 配置日志
handler = logging.StreamHandler()
formatter = logging.Formatter("[Agents] %(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger("agents")
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_id(image_path: Path):
    videoid_string = image_path.parent.parent.stem
    videoid = int(videoid_string[3:])
    frameid = int(image_path.stem)
    return videoid, frameid


def get_triplet_name_list(pred_ivt: List[bool]) -> List[str]:
    # 取得triplet名称
    triplet_names = [item for item, flag in zip(triplet_mapping, pred_ivt) if flag]

    return triplet_names


def format_grounding_coord(coord: List[int]) -> str:
    return f"[x_min={coord[0]}, y_min={coord[1]}, x_max={coord[2]}, y_max={coord[3]}]"


def count_chinese_chars(text):
    """统计字符串中汉字的个数"""
    pattern = re.compile(r"[\u4e00-\u9fff]")
    return sum(1 for _ in pattern.finditer(text))


def clean_markdown_detailed(text):
    """分步去除markdown格式"""
    # 去除粗斜体格式
    text = re.sub(r"\*{3}([^*]+)\*{3}", r"\1", text)
    # 去除加粗格式
    text = re.sub(r"\*{2}([^*]+)\*{2}", r"\1", text)
    # 去除斜体格式
    text = re.sub(r"(?<!\*)\*([^\*]+)\*(?!\*)", r"\1", text)
    # 去除标题标记
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    return text


def clean_special_char(text):
    """其他的特殊替换"""
    # “中文 (对应英文)”形式的替换为其中的英文
    text = re.sub(r"[\u4e00-\u9fff]+ \(([a-z0-9 ]+)\)", r" \1", text)
    # 规范撇号
    text = re.sub(r"\u2019", "'", text)
    # 规范双引号
    text = re.sub(r"[\u201c\u201d]", '"', text)
    # 规范短横杠
    text = re.sub(r"[\u2013\u2014]", "-", text)
    # 移除think标签
    text = text.replace("</think>", "").replace("<think>", "")
    # 移除box标签
    text = text.replace("<|end_of_box|>", "").replace("<|begin_of_box|>", "")
    # 移除代码段符号
    text = text.replace("```", "")
    # 移除无意义空格
    text = re.sub(r"[\u2003]", "", text)

    return text


async def mcq_agent(
    task_name: Literal[
        "instrument grounding", "instrument identification", "verb identification", "target identification"
    ],
    question: str,
    options: List[str],
    api_client: OpenAIConcurrentClient,
    image_path: Path,
) -> Dict:
    """解决选择题的Agent

    若指定的任务名称不在规定选项范围中，默认视为“instrument identification”任务。另外，选项长度最长为26，超过
    26的只截断取前26个。

    Args:
        task_name (Literal[ &quot;instrument grounding&quot;, &quot;instrument identification&quot;, &quot;verb identification&quot;, &quot;target identification&quot; ]): 要解决的选择题的任务名称，可选器械定位、器械识别、操作识别、目标识别
        question (str): 问题文本，要求仅包含问题本身，不包含选项
        options (List[str]): 选项列表
        api_client (OpenAIConcurrentClient): 要使用的API客户端
        image_path (Path): 要附加的图片路径

    Returns:
        Dict: 包含raw_result和agent_pred两项，分别为RequestResult类型的原始请求记录与List[bool]类型的onehot的agent预测结果
    """
    model_mapping = {
        "instrument grounding": instrument_grounding_model,
        "instrument identification": instrument_identification_model,
        "verb identification": verb_identification_model,
        "target identification": target_identification_model,
    }
    system_prompt_mapping = {
        "instrument grounding": instrument_agent_system_prompt,
        "instrument identification": instrument_agent_system_prompt,
        "verb identification": verb_agent_system_prompt,
        "target identification": target_agent_system_prompt,
    }
    if task_name not in (
        "instrument grounding",
        "instrument identification",
        "verb identification",
        "target identification",
    ):
        task_name = "instrument identification"
    if len(options) > len(ascii_uppercase):
        options = options[: len(ascii_uppercase)]
    videoid, frameid = extract_id(image_path)
    # 构建选项行
    format_options = [f"{ascii_uppercase[i]}. {x}" for i, x in enumerate(options)]
    format_options = "\n".join(format_options)
    # 构建输出约束内容
    option_headers = ", ".join(ascii_uppercase[: len(options) - 1]) + f" or {ascii_uppercase[len(options)-1]}"
    format_constraint = mcq_output_format_constraint.format(letter_headers=option_headers)
    # 拼合请求item
    request_item = {
        "id": f"{videoid}_{frameid}_{task_name.replace(' ', '-')}",
        "request": dict(
            model=model_mapping[task_name],
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_mapping[task_name],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
                        },
                        {
                            "type": "text",
                            "text": f"{question}\n{format_options}\nImportant: {format_constraint}",
                        },
                    ],
                },
            ],
            extra_body={"thinking": {"type": "disabled"}},
        ),
    }
    result = await api_client.process_request(request_item)

    # 定位提取模型给出的选项
    pattern = f"\\b[{ascii_uppercase[:len(options)]}]+\\b"
    if result.response:
        selected_options = re.findall(pattern, result.response["choices"][0]["message"]["content"])
        if selected_options:
            selected_options = selected_options[0]
        else:
            selected_options = ""
    else:
        selected_options = []
    # 构建one-hot形式的输出，存在多选的情况
    agent_pred = [False] * len(options)
    for opt in selected_options:
        agent_pred[ord(opt) - ord("A")] = True

    if len(selected_options) > 0:
        logger.info(f"{task_name} Agent调用完成，选项解析正常")
    else:
        # verb和target识别中，不选择时默认视为null
        if task_name in ("verb identification", "target identification"):
            agent_pred[-1] = True
            logger.warning(
                f"{task_name} Agent调用完成，未解析到选项，使用默认null - VID{str(videoid).zfill(2)}_{str(frameid).zfill(6)}"
            )
        else:
            logger.warning(
                f"{task_name} Agent调用完成，选项解析为空 - VID{str(videoid).zfill(2)}_{str(frameid).zfill(6)}"
            )

    return {"raw_result": result, "agent_pred": agent_pred}


async def triplet_agent(pred_i: List[bool], pred_v: List[bool], pred_t: List[bool]):
    # 构建components id -> triplet id的映射
    valid_combo = [(2, 2, 5), (3, 5, 9), (3, 2, 1), (5, 2, 4), (1, 3, 0), (1, 0, 1), (1, 1, 0), (5, 1, 8), (3, 5, 2), (3, 5, 11), (0, 1, 8), (4, 4, 3), (1, 3, 2), (1, 3, 11), (0, 0, 0), (1, 1, 2), (5, 1, 10), (2, 2, 0), (0, 1, 1), (0, 1, 10), (4, 4, 5), (2, 5, 11), (1, 3, 4), (0, 0, 2), (1, 1, 4), (0, 0, 11), (2, 2, 2), (2, 2, 11), (0, 1, 12), (5, 2, 1), (5, 2, 10), (3, 2, 10), (0, 0, 4), (0, 2, 1), (0, 0, 13), (0, 2, 10), (3, 5, 8), (2, 1, 8), (1, 2, 1), (1, 3, 8), (1, 2, 10), (3, 3, 10), (2, 3, 8), (3, 5, 1), (3, 5, 10), (4, 4, 2), (1, 3, 1), (1, 2, 3), (0, 0, 8), (5, 7, 4), (2, 3, 1), (5, 1, 0), (2, 3, 10), (3, 5, 3), (0, 1, 0), (4, 4, 4), (1, 3, 3), (0, 0, 1), (0, 0, 10), (2, 3, 3), (5, 2, 0), (3, 2, 0), (0, 0, 3), (0, 2, 0), (0, 0, 12), (5, 7, 8), (2, 3, 5), (1, 1, 8), (2, 5, 5), (5, 2, 2), (1, 2, 0), (0, 8, 0), (1, 2, 9), (1, 3, 10), (1, 1, 10), (2, 1, 0), (4, 4, 1), (1, 2, 2), (2, 3, 0), (1, 0, 13), (2, 2, 1), (3, 5, 5), (0, 1, 2), (2, 2, 10), (0, 1, 11), (1, 3, 5), (2, 3, 2), (5, 6, 6), (2, 2, 3), (0, 1, 4), (1, 3, 7), (5, 7, 7), (2, 3, 4), (1, 0, 8)]
    valid_combo = {v: i for i, v in enumerate(valid_combo)}
    # 转换为标签id列表
    lbs_i = [i for i, v in enumerate(pred_i) if v]
    lbs_v = [i for i, v in enumerate(pred_v) if v]
    lbs_t = [i for i, v in enumerate(pred_t) if v]
    # 排列组合生成所有三元组，并保留可能的三元组
    triplets = list(product(lbs_i, lbs_v, lbs_t))
    triplets = [valid_combo[t] for t in triplets if t in valid_combo]
    # 如果v/t有任一元为null，则添加94号三元组标签
    if any([pred_v[-1], pred_t[-1]]):
        triplets.append(95)
    # 转换为onehot
    triplets_onehot = [False] * 95
    for i in triplets:
        triplets_onehot[i if i < len(triplets_onehot) else -1] = True
    # TODO: 模拟triplet判断时的耗时操作，记得删除
    # await asyncio.sleep(3.0)

    return {"raw_result": None, "agent_pred": triplets_onehot}


async def phase_agent(
    question: str,
    options: List[str],
    api_client: OpenAIConcurrentClient,
    image_path: Path,
    pred_ivt: Optional[List[bool]] = None,
) -> Dict:
    if len(options) > len(ascii_uppercase):
        options = options[: len(ascii_uppercase)]
    videoid, frameid = extract_id(image_path)
    # 构建选项行
    format_options = [f"{ascii_uppercase[i]}. {x}" for i, x in enumerate(options)]
    format_options = "\n".join(format_options)
    # 构建输出约束内容
    option_headers = ", ".join(ascii_uppercase[: len(options) - 1]) + f" or {ascii_uppercase[len(options)-1]}"
    format_constraint = phase_output_format_constraint.format(letter_headers=option_headers)

    triplet_names = get_triplet_name_list(pred_ivt) if pred_ivt else None
    # 构建三元组提示
    if triplet_names is None or len(triplet_names) == 0:
        triplet_type_hint = ""
    else:
        triplet_type_hint = "Triplet Type: {" + ", ".join([f"({name})" for name in triplet_names]) + "}"
    # 拼合请求item
    request_item = {
        "id": f"{videoid}_{frameid}_phase-classification",
        "request": dict(
            model=phase_classification_model,
            messages=[
                {
                    "role": "system",
                    "content": phase_agent_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
                        },
                        {
                            "type": "text",
                            "text": f"{question}. {triplet_type_hint} \n{format_options}\nImportant: {format_constraint}",
                        },
                    ],
                },
            ],
            extra_body={"thinking": {"type": "disabled"}},
        ),
    }
    result = await api_client.process_request(request_item)

    # 定位提取模型给出的选项
    pattern = f"\\b[{ascii_uppercase[:len(options)]}]+\\b"
    if result.response:
        selected_options = re.findall(pattern, result.response["choices"][0]["message"]["content"])
        if selected_options:
            selected_options = selected_options[0]
        else:
            selected_options = ""
    else:
        selected_options = []
    # 构建one-hot形式的输出
    agent_pred = [False] * len(options)
    for opt in selected_options:
        agent_pred[ord(opt) - ord("A")] = True

    if len(selected_options) == 1:
        logger.info(f"Phase Agent调用完成，选项解析正常")
    elif len(selected_options) > 1:
        logger.warning(
            f"Phase Agent调用完成，选项解析存在多个 - VID{str(videoid).zfill(2)}_{str(frameid).zfill(6)}"
        )
    else:
        logger.warning(
            f"Phase Agent调用完成，选项解析为空 - VID{str(videoid).zfill(2)}_{str(frameid).zfill(6)}"
        )

    return {"raw_result": result, "agent_pred": agent_pred}


async def vqa_agent(
    pred_i: List[bool],
    pred_ivt: List[bool],
    pred_phase: List[bool],
    question: str,
    image_path: Path,
    api_client: OpenAIConcurrentClient,
    pred_grounding: Optional[List[List[int]]] = None,
) -> Dict:
    """生成open description的Agent

    Args:
        pred_i (List[bool]): 前序步骤的instrument identification结果
        pred_ivt (List[bool]): 前序步骤的triplet recognition结果
        pred_phase (List[bool]): 前序步骤的phase classification结果
        question (str): 输入给大模型的问题，要求仅包含问题本身
        image_path (Path): 要附加的图片路径
        api_client (OpenAIConcurrentClient): 要使用的API客户端
        pred_grounding (Optional[List[List[int]]], optional): 要附加的器械定位信息

    Returns:
        Dict: 大模型返回的原始内容以及初步清洗后的描述结果
    """
    videoid, frameid = extract_id(image_path)
    context_info = []
    if any(pred_phase):
        phase_lb = phase_descriptions[pred_phase.index(True)]
        context_info.append(f"This image shows the {phase_lb}.")
    if any(pred_ivt):
        triplet_lbs = get_triplet_name_list(pred_ivt)
        triplet_lbs = [lb.replace(",", " ") for lb in triplet_lbs]
        triplet_lbs = ", ".join(triplet_lbs)
        context_info.append(f"In this scene, {triplet_lbs}.")
    elif any(pred_i):
        instru_lbs = [name for name, flag in zip(instrument_names, pred_i) if flag]
        if len(instru_lbs) > 1:
            instru_lbs = ", ".join(instru_lbs[:-1]) + " and " + instru_lbs[-1]
        else:
            instru_lbs = instru_lbs[0]
        context_info.append(f"In this scene, {instru_lbs} detected.")

    if pred_grounding and (any(pred_ivt) or any(pred_i)):
        # 构建器械位置提示信息
        grounding_info = [format_grounding_coord(coord) for coord in pred_grounding]
        if len(grounding_info) > 1:
            grounding_info = ", ".join(grounding_info[:-1]) + " and " + grounding_info[-1]
        else:
            grounding_info = grounding_info[0]
        context_info.append(f'Instrument{"s" if len(pred_grounding) > 1 else ""} grounding at')

    final_prompt = (
        " ".join(context_info)
        + f"Based on the visual content and this clinical context, please answer the following question: {question}\n{open_description_format_constraint}"
    )

    request_item = {
        "id": f"{videoid}_{frameid}_open-description",
        "request": dict(
            model=open_description_model,
            messages=[
                {
                    "role": "system",
                    "content": vqa_agent_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
                        },
                        {
                            "type": "text",
                            "text": final_prompt,
                        },
                    ],
                },
            ],
            extra_body={"thinking": {"type": "disabled"}},
        ),
    }
    result = await api_client.process_request(request_item)
    if result.response:
        content = result.response["choices"][0]["message"]["content"]
    else:
        content = ""
    content = clean_markdown_detailed(clean_special_char(content))
    num_chinese_chars = count_chinese_chars(content)
    if num_chinese_chars > 0:
        logger.warning(
            f"VQA Agent调用完成，回答中包含{num_chinese_chars}个中文字符 - VID{str(videoid).zfill(2)}_{str(frameid).zfill(6)}"
        )
    else:
        logger.info(f"VQA Agent调用完成，回答解析正常")

    return {"raw_result": result, "agent_pred": content}


async def mrg_agent(
    pred_i: List[bool],
    pred_ivt: List[bool],
    pred_phase: List[bool],
    open_desc: Optional[str],
    image_path: Path,
    api_client: OpenAIConcurrentClient,
    api_client_refine: Optional[OpenAIConcurrentClient]=None,
    attach_img: bool=False,
    origin_mrg: Optional[str]=None
):
    if api_client_refine is None:
        api_client_refine = api_client
    # 可选是否使用具体的图片
    videoid, frameid = extract_id(image_path)
    context_info = ["Input:"]
    if any(pred_phase):
        phase_lb = phase_descriptions[pred_phase.index(True)]
        context_info.append(f"{len(context_info)}. Surgical Phase: {phase_lb}")
    if any(pred_ivt):
        triplet_lbs = get_triplet_name_list(pred_ivt)
        triplet_lbs = [f"\t- {lb.replace(',', '-')}" for lb in triplet_lbs]
        triplet_lbs = "\n".join(triplet_lbs)
        context_info.append(f"{len(context_info)}. Key validated actions (triplets):\n{triplet_lbs}")
    if any(pred_i):
        instru_lbs = [name for name, flag in zip(instrument_names, pred_i) if flag]
        instru_lbs = [f"\t- {lb}" for lb in instru_lbs]
        instru_lbs = "\n".join(instru_lbs)
        context_info.append(f"{len(context_info)}. Visible instruments:\n{instru_lbs}")
    if open_desc:
        context_info.append(f"{len(context_info)}. Visual description::\n\t{open_desc}")

    final_prompt = "\n".join(context_info) + f"\n\nTasks:\n{mrg_task_prompt}\n" + mrg_format_constraint

    request_item = {
        "id": f"{videoid}_{frameid}_mrg",
        "request": dict(
            model=mrg_model,
            messages=[
                {
                    "role": "system",
                    "content": mrg_agent_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": final_prompt,
                        },
                    ],
                },
            ],
            extra_body={"thinking": {"type": "disabled"}},
        ),
    }
    if attach_img:
        request_item["request"]["messages"][1]['content'].append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"}}
        )
    if not origin_mrg:
        result = await api_client.process_request(request_item)
        if result.response:
            content = result.response["choices"][0]["message"]["content"]
        else:
            content = ""
        content = clean_markdown_detailed(clean_special_char(content)).strip()
        num_chinese_chars = count_chinese_chars(content)
        if num_chinese_chars > 0:
            logger.warning(
                f"MRG Agent原始回答完成，回答中包含{num_chinese_chars}个中文字符 - VID{str(videoid).zfill(2)}_{str(frameid).zfill(6)}"
            )
        else:
            logger.info(f"MRG Agent原始回答完成，回答解析正常")
    else:
        result = None
        logger.info(f'原始 mrg 已存在，直接进行refine')
        content = origin_mrg
        content = clean_markdown_detailed(clean_special_char(content)).strip()
        num_chinese_chars = count_chinese_chars(content)
    

    extract_pattern = r"(?:Observed|Observations?):\s*(.*?)\s*Clinical Significance:\s*(.*?)\s*Next Steps:\s*(.*)"
    refine_request_item = {
        "id": f"{videoid}_{frameid}_mrg-refine",
        "request": dict(
            model=mrg_refine_model,
            messages=[
                {
                    "role": "system",
                    "content": mrg_refine_agent_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": MRG_REFINE_TEMPLATE.format(context=content),
                        },
                    ],
                },
            ],
            extra_body={"thinking": {"type": "disabled"}},
        ),
    }
    refine_result = await api_client_refine.process_request(refine_request_item)
    if refine_result.response:
        refined_content = refine_result.response["choices"][0]["message"]["content"]
    else:
        refined_content = ""
    refined_content = clean_markdown_detailed(clean_special_char(refined_content)).strip()
    match = re.search(extract_pattern, refined_content, re.IGNORECASE | re.DOTALL)
    if match is not None:
        observation = match.group(1).strip()
        clinical_significance = match.group(2).strip()
        next_steps = match.group(3).strip()
    else:
        observation = clinical_significance = next_steps = None

    if observation is None:
        logger.warning(
            f"MRG Agent refine失败，未能解析三个分点报告 - VID{str(videoid).zfill(2)}_{str(frameid).zfill(6)}"
        )
    else:
        logger.info(f"MRG Agen refine完成，回答解析正常")

    return {
        "raw_result": (result, refine_result),
        "agent_pred": {
            "detail": content,
            "observation": observation,
            "clinical_significance": clinical_significance,
            "next_steps": next_steps,
        },
    }
