import random
from pathlib import Path
from openai_concurrent import OpenAIConcurrentClient
import base64
from typing import Literal
import os
import json
from tqdm import tqdm
import asyncio
from pymongo import MongoClient
from pymongo.collection import Collection


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

######### 内容配置 #########
random.seed(666)

OUTPUT_FORMAT_CONSTRAINTS = """Important: Please provide your answer in a clear, text-based format, but do not use any markdown tables."""

BASE_URL = "https://api.siliconflow.cn/v1"

OPEN_DESCRIPTION_TEMPLATE = """{context}Based on the visual content and this clinical context, please answer the following question: {question}

Please provide a detailed and clinically accurate assessment based on what you can observe in the image.
{extra_content}"""

MRG_TEMPLATE = """{context}Based on the visual content and this clinical context, please generate a detailed medical report addressing the following: {question}

Please provide a comprehensive medical report that includes:
1. Observation of visible findings, surgical progress, and instrument usage.
2. Clinical significance of this moment/phase.
3. Relevant clinical considerations or next procedural steps.
{extra_content}"""

SYSTEM_PROMPT = """You are an AI assistant specializing in surgical video analysis.
You can also imagine yourself as a lecturer on surgery who explains surgeons' thought processes and other surgical rationales to new junior surgeons who ask you questions.
Your task is to provide detailed, medically accurate, and contextually relevant responses for surgical video frame analysis questions."""

open_description_questions = [
    "Can you describe the findings visible in this endoscopy image?",
    "What can you observe in this endoscopic image?",
    "Can you provide a detailed description of what's visible in this endoscopy image?",
    "Please describe the characteristics of the tissue shown in this endoscopic image.",
    "What are the key features you can identify in this endoscopy image?",
    "Can you analyze the visual features of this endoscopy image?",
    "What abnormalities or normal findings can you identify in this image?",
    "Please provide an analysis of the findings in this endoscopy image.",
    "Can you describe the characteristics of any lesions or tissue in this endoscopy image?",
    "What diagnostic features are visible in this endoscopic visualization?",
]

mrg_questions = [
    "Can you generate a detailed medical report for this endoscopy image?",
    "Please create a comprehensive endoscopy report based on this image.",
    "Generate a professional endoscopy examination report for this image.",
    "Please prepare a detailed endoscopy report with your assessment of this image.",
    "Create a medical report documenting the findings in this endoscopy image.",
    "Generate a clinical endoscopy report based on the visual findings in this image.",
    "Please provide a structured medical report for this endoscopic examination.",
    "Create a comprehensive diagnostic report for this endoscopy image.",
    "Generate a detailed clinical assessment report for this endoscopic image.",
    "Please write a professional medical report based on this endoscopy examination.",
]

phase_descriptions = [
    "the preparation phase",
    "Calot's triangle dissection phase",
    "cystic duct clipping and cutting phase",
    "gallbladder dissection phase",
    "gallbladder packaging or detachment phase",
    "the cleaning and coagulation phase",
    "gallbladder retraction phase",
]

instrument_names = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]


async def main():
    """关键变量说明

    - storage_path: CholecTrack20数据集目录
    - log_filepath: 完整请求调用日志文件的保存位置（日志可能会比较大）
    - batch_size: 单批次处理的任务数量
    - video_ids: 要处理视频的id
    - task_name: 用于benchmark记录的任务名称，可选值有 'open description', 'mrg'
    - jsonl_record_savepath: benchmark记录的保存路径，即要输出的最终结果jsonl的保存路径
    - selected_frame_json_filepath: 帧列表json文件路径
    """
    type_mapping_videoid = {
        "Testing": [1, 6, 7, 12, 25, 39, 92, 111],
        "Validation": [30, 110],
        "Training": [2, 4, 11, 13, 17, 23, 31, 37, 96, 103],
    }
    videoid_mapping_type = {
        video_id: category for category, ids in type_mapping_videoid.items() for video_id in ids
    }

    storage_path = Path("/media/georgehu/backupDisk/dataset/CholecTrack20")
    log_filepath = Path("log1.jsonl.gz")

    myclient = OpenAIConcurrentClient(
        os.getenv("SILICLOUD_API_KEY"),
        BASE_URL,
        log_archieve_filepath=log_filepath,
        max_requests_per_minute=1500,
        max_tokens_per_minute=35000,
        max_concurrent=15,
        max_retries=3,
    )

    # 单批次处理的任务数量
    batch_size = 30
    # 要处理视频的id
    video_ids = [30, 110, 1, 6, 7, 12, 25, 39, 92, 111]
    # 用于benchmark记录的任务名称，可选值 open description, mrg
    task_name: Literal["open description", "mrg"] = "open description"
    # 用于benchmark记录的任务类型
    task_type = "vqa"
    # benchmark记录的保存路径
    jsonl_record_savepath = Path("open_desc_tmp.jsonl")
    # 帧列表文件路径
    selected_frame_json_filepath = Path(
        "selected_frameid.json"
    )

    # 加载指定视频的帧列表
    print("加载帧列表......", end="")
    with open(selected_frame_json_filepath, "r", encoding="utf-8") as fi:
        selected_frame = json.load(fi)
    selected_frame = {int(k[3:]): v for k, v in selected_frame.items() if int(k[3:]) in video_ids}
    selected_frame = {
        25: {24126}
    }
    print(f"OK. {len(selected_frame)}个视频，共{sum([len(i) for i in selected_frame.values()])}帧")

    # 存放构造出的各批次请求内容
    request_batches = []

    for videoid, frameid_list in selected_frame.items():
        # 暂存当前视频的请求内容
        video_request = []
        # 计算当前视频所需的批次数量
        num_batch = len(frameid_list) // batch_size
        if batch_size * num_batch < len(frameid_list):
            num_batch += 1
        videoid_string = str(videoid).zfill(2)

        print(f"加载VID{videoid_string} {num_batch}个批次共{len(frameid_list)}帧的请求数据.........", end="")
        # 读取当前视频的标签文件
        with open(
            storage_path / videoid_mapping_type[videoid] / f"VID{videoid_string}/vid{videoid_string}.json",
            "r",
            encoding="utf-8",
        ) as fi:
            video_anno_json = json.load(fi)

        for frame_id in frameid_list:
            phaseids = {
                ann["phase"]
                for ann in video_anno_json["annotations"][str(frame_id)]
                if ann["phase"] >= 0 and ann["phase"] < len(phase_descriptions)
            }
            instrumentids = {
                ann["instrument"]
                for ann in video_anno_json["annotations"][str(frame_id)]
                if ann["instrument"] >= 0 and ann["instrument"] < len(instrument_names)
            }
            # 构建当前帧中场景的相关上下文文本信息
            phase_desc = ", ".join(phase_descriptions[phaseid] for phaseid in phaseids)
            context_text = f"This surgical frame represents {phase_desc} of a cholecystectomy."
            if len(instrumentids) > 0:
                instru_desc = ", ".join(instrument_names[instruid] for instruid in instrumentids)
                context_text += f" Visible surgical instruments include: {instru_desc}"
            else:
                context_text += " No specific surgical instruments are visibly active in this frame."

            img_path = (
                storage_path
                / videoid_mapping_type[videoid]
                / f"VID{videoid_string}/Frames/{str(frame_id).zfill(6)}.png"
            )
            video_request.append(
                {
                    "id": f"{videoid_string}_{frame_id}",
                    "request": dict(
                        model="zai-org/GLM-4.5V",
                        messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{encode_image(img_path)}"
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": (
                                            OPEN_DESCRIPTION_TEMPLATE.format(
                                                context=context_text + "\n",
                                                question=random.choice(open_description_questions),
                                                extra_content=OUTPUT_FORMAT_CONSTRAINTS,
                                            )
                                            if task_name == "open description"
                                            else MRG_TEMPLATE.format(
                                                context=context_text + "\n",
                                                question=random.choice(mrg_questions),
                                                extra_content=OUTPUT_FORMAT_CONSTRAINTS,
                                            )
                                        ),
                                    },
                                ],
                            },
                        ],
                        extra_body={"thinking": {"type": "disabled"}},
                    ),
                }
            )
        # 划分批次放入最终的请求列表
        request_batches += [
            video_request[i : i + batch_size] for i in range(0, len(video_request), batch_size)
        ]
        print("OK")

    def progress_callback(completed: int, total: int):
        print(f"当前批次进度: {completed}/{total} ({completed/total*100:.1f}%)")

    print("开始请求调用")
    # 存放请求失败任务的执行结果
    failed_request = []
    try:
        # 逐批次请求
        for batch in tqdm(request_batches):
            # 执行一个批次的请求
            results = await myclient.batch_requests(batch, progress_callback)
            jsonl = []
            for result in results:
                videoid_string, frame_id = result.request_id.split("_")
                if not result.success:
                    failed_request.append(result)
                    continue
                model_answer = result.response["choices"][0]["message"]["content"]
                model_answer = model_answer.replace("<|end_of_box|>", "").replace("<|begin_of_box|>", "").replace("<think>", "").replace("</think>", "")
                jsonl.append(
                    {
                        "image_path": f"CholecTrack20/{videoid_mapping_type[int(videoid_string)]}/VID{videoid_string.zfill(2)}/Frames/{str(frame_id).zfill(6)}.png",
                        "tasks": [
                            {
                                "name": task_name,
                                "type": task_type,
                                "question": result.request_data["messages"][1]["content"][1]["text"],
                                "model_answer": model_answer,
                            }
                        ],
                    }
                )
            # 每执行一个批次保存一下benchmark记录
            with open(jsonl_record_savepath, "a", encoding="utf-8") as fo:
                fo.write("\n".join([json.dumps(line) for line in jsonl]))
                if len(jsonl) > 0:
                    fo.write("\n")

            if len(failed_request) > 0:
                failed_request_ids = [res.request_id for res in failed_request]
                with open('failed_list', 'w', encoding='utf-8') as fo:
                    json.dump(failed_request_ids, fo)
                print(f"failed list: {failed_request_ids}")
    except Exception as e:
        print(e)
    finally:
        failed_request_ids = [res.request_id for res in failed_request]
        with open('failed_list', 'w', encoding='utf-8') as fo:
            json.dump(failed_request_ids, fo)
        # print(failed_request)
        print(f"final failed list: {failed_request_ids}")
        await myclient.close()


def merge_vqa_answer(
    cleaned_desc_filepath: Path,
    cleaned_mrg_filepath: Path,
    refined_mrg_filepath: Path,
    collection: Collection,
):
    def extract_id(image_path: Path):
        videoid_string = image_path.parent.parent.stem
        videoid = int(videoid_string[3:])
        frameid = int(image_path.stem)

        return videoid, frameid

    with open(cleaned_desc_filepath, 'r', encoding='utf-8') as fi:
        clean_desc = [json.loads(line) for line in fi.readlines()]
    with open(cleaned_mrg_filepath, 'r', encoding='utf-8') as fi:
        clean_mrg = [json.loads(line) for line in fi.readlines()]
    with open(refined_mrg_filepath, 'r', encoding='utf-8') as fi:
        refine_mrg = json.load(fi)

    # 追加MRG简述部分
    for mrg_record in clean_mrg:
        videoid, frameid = extract_id(Path(mrg_record['image_path']))
        videoid_string, frameid_string = str(videoid).zfill(2), str(frameid).zfill(6)
        detailed_ans = mrg_record['tasks'][0]['model_answer']
        observation = refine_mrg[f'{videoid_string}_{frameid_string}'][0]
        clinical_significance = refine_mrg[f'{videoid_string}_{frameid_string}'][1]
        next_steps = refine_mrg[f'{videoid_string}_{frameid_string}'][2]
        mrg_record['tasks'][0]['model_answer'] = {
            'detail': detailed_ans,
            'observation': observation,
            'clinical_significance': clinical_significance,
            'next_steps': next_steps
        }
    print(len(clean_mrg))

    # 合并open description任务
    print('开始合并open description任务......')
    counter = 0
    for desc_record in tqdm(clean_desc):
        # 先查询是否存在该帧的记录
        existing_doc = collection.find_one({'image_path': desc_record['image_path']})
        if not existing_doc:
            result = collection.insert_one(desc_record)
        else:
            # 检查目标帧是否已存在open description任务记录
            existing_tasks = existing_doc.get('tasks', [])
            # 若已存在open description记录则跳过，不覆盖
            if len(list(filter(lambda x: x['name'] == 'open description', existing_tasks))) > 0:
                continue
            # 不存在则增量更新tasks列表
            result = collection.update_one(
                {"image_path": desc_record["image_path"]},
                {"$push": {"tasks": {"$each": desc_record["tasks"]}}},
            )
        if result.modified_count == 0:
            print(f"{desc_record['image_path']}失败:")
            print(result)
        else:
            counter += result.modified_count
    print(f'完成open description任务合并，共修改了{counter}条记录')

    # 合并mrg任务
    print('开始合并MRG任务......')
    counter = 0
    for mrg_record in tqdm(clean_mrg):
        # 先查询是否存在该帧的记录
        existing_doc = collection.find_one({'image_path': mrg_record['image_path']})
        if not existing_doc:
            result = collection.insert_one(mrg_record)
        else:
            # 检查目标帧是否已存在MRG任务记录
            existing_tasks = existing_doc.get('tasks', [])
            # 若已存在mrg记录则跳过，不覆盖
            if len(list(filter(lambda x: x['name'] == 'mrg', existing_tasks))) > 0:
                continue
            # 不存在则增量更新tasks列表
            result = collection.update_one(
                {"image_path": mrg_record["image_path"]},
                {"$push": {"tasks": {"$each": mrg_record["tasks"]}}},
            )
        if result.modified_count == 0:
            print(f"{mrg_record['image_path']}失败:")
            print(result)
        else:
            counter += result.modified_count
    print(f'完成MRG任务合并，共修改了{counter}条记录')



if __name__ == "__main__":
    # asyncio.run(main())
    mongo_client = MongoClient("mongodb://localhost:27017")
    db = mongo_client["playground"]
    collection = db["cholectrack20_benchmark"]
    merge_vqa_answer(
        Path('/home/georgehu/Documents/Projects/surg-agent/output/cleaned_desc.jsonl'),
        Path('/home/georgehu/Documents/Projects/surg-agent/output/cleaned_mrg.jsonl'),
        Path('/home/georgehu/Documents/Projects/surg-agent/output/refined_mrg.json'),
        collection
    )
