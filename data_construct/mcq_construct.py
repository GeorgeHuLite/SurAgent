import pickle
import json
from typing import List, Tuple
from collections import defaultdict
import random
import copy
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection
from tqdm import tqdm


random.seed(666)

open_describe_question_template = [
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

mrg_question_template = [
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

instrument_grounding_question_template = [
    "Which bounding box corresponds to the {instrument} in this image?",
    "Select the bounding box that best localizes the {instrument}:",
    "Identify the bounding box coordinates for the {instrument}.",
    "Choose the correct [x_min, y_min, x_max, y_max] bounding box for the {instrument}.",
    "Which of the following bounding boxes encloses the {instrument}?",
    "Determine the correct bounding box for the {instrument} in this frame.",
    "Select the coordinates that accurately mark the {instrument}’s location.",
    "Which bounding box best identifies the {instrument}?",
    "Choose the bounding box that correctly indicates the {instrument} in the scene.",
    "Which of these coordinate sets corresponds to the {instrument}?",
    "Identify the region bounded by the correct coordinates for the {instrument}.",
    "Which bounding box correctly captures the {instrument} position?",
    "Select the bounding box that most precisely outlines the {instrument}.",
    "Choose the accurate coordinates that define the boundaries of the {instrument}.",
    "Which option correctly localizes the {instrument} in the image?",
]

phase_classification_question_template = [
    "What surgical phase is depicted in this endoscopic frame?",
    "Can you identify the ongoing surgical phase in this image?",
    "Which stage of the cholecystectomy procedure is represented here?",
    "Based on the visual cues, what is the current phase of the operation?",
    "Please select the correct surgical phase for this frame.",
    "According to the endoscopic view, which surgical phase is active?",
    "At what point in the cholecystectomy are we, judging by this frame?",
    "Based on the imagery, what phase of the operation does this frame illustrate?",
    "Which surgical step is currently unfolding, given this endoscopic perspective?",
    "Determine the specific surgical phase evident in this image.",
    "From the endoscopic scene, which phase of surgery is being performed?",
    "Which stage of the procedure can be recognized in this frame?",
    "What operative phase can you infer from this frame?",
    "Identify the ongoing step of the surgery given this endoscopic view.",
    "What stage of the surgery does the current image depict?",
]

instrument_identification_question_template = [
    "What surgical instruments are visible in this frame?",
    "Which surgical instruments best correspond to this endoscopic view?",
    "What surgical instruments can be observed in this image?",
    "Which surgical instruments are clearly identifiable in this image?",
    "What surgical instruments are present in this endoscopic image?",
    "What surgical instruments are clearly visible in the endoscopic view?",
    "What visible surgical instruments are featured in this image?",
    "What surgical instruments are featured in this endoscopic snapshot?",
    "What types of surgical instruments can be observed in this image?",
    "What surgical instruments are clearly displayed in this endoscopic frame?",
    "Can you identify all the surgical instruments shown in this endoscopic frame?",
    "Identify all the surgical instruments that can be seen in this image.",
    "What instruments are present and recognizable in this endoscopic scene?",
    "From this frame, what surgical instruments can be identified?",
    "Name all the instruments you can observe in this endoscopic capture.",
]

verb_identification_question_template = [
    "What surgical actions are being performed in this frame?",
    "Which surgical actions are prominent in this image?",
    "Can you identify the primary surgical actions taking place?",
    "What surgical procedures are being demonstrated?",
    "What surgical maneuvers is being executed in this endoscopic view?",
    "Identify the surgical actions occurring in this frame.",
    "Based on the visual cues, what actions are the surgeon currently undertaking?",
    "What are the correct terms for the surgical actions observed here?",
    "Select the appropriate verbs that describe current surgical activity.",
    "Which verbs best describe the surgical motion in this scene?",
    "What type of manipulation is the surgeon applying here?",
    "Identify the motions being carried out with the visible tool.",
    "From this endoscopic snapshot, what activity is occurring?",
    "What operative verbs accurately characterize this procedure?",
    "Identify the operative actions taking place in this view.",
]

target_identification_question_template = [
    "Which anatomical structure is being manipulated in this scene?",
    "What are the target organs or tissues that the instrument is interacting with?",
    "Can you name the structures that are currently exposed for dissection?",
    "What is the anatomical site where the surgical tool is applied?",
    "Which tissues are being operated in this frame?",
    "Which anatomical structures are the surgeon currently interacting with?",
    "What tissues or organs are being manipulated in this frame?",
    "Which anatomical regions are currently the focus of the surgical procedure?",
    "Which anatomical targets are visible in the operative field at present?",
    "Can you determine the organs or landmarks that the surgeon is operating on?",
    "What are the anatomical sites of interest in this frame?",
    "Identify the tissues that the surgical tools are in contact with.",
    "Which target organs or regions are approaching for intervention?",
    "Can you specify the anatomical components that are the current focus of manipulation?",
    "Which anatomical structures are being worked on in this step of surgery?",
]


def generate_random_bbox(width: int, height: int) -> List[int]:
    """随机生成一个bbox

    Args:
        width (int): 有效的宽度
        height (int): 有效的高度

    Returns:
        List[int]: 格式为[x_min, y_min, x_max, y_max]的bbox
    """
    x_min, y_min = random.randint(0, width - 2), random.randint(0, height - 2)
    x_max, y_max = random.randint(x_min + 1, width - 1), random.randint(y_min + 1, height - 1)
    return [x_min, y_min, x_max, y_max]


def generate_random_bbox_list(
    gt_bboxes: List[List[int]], width: int, height: int
) -> Tuple[List[List[int]], int]:
    """生成随机bbox选项列表

    最多支持三个bbox的处理，即gt_bboxes的长度不能超过3。
    若只有一个bbox，随机生成另外三个；若有两个bbox（a, b），随机再生成两个（c, d），组合为ab, bc, cd, ad；
    若有三个bbox（a, b, c），随机再生成一个（d），组合为abc, bcd, cda, dab。各bbox组合内部的顺序也都是随机的

    Args:
        gt_bboxes (List[List[int]]): 目标bbox的列表，要求格式为[x_min, y_min, x_max, y_max]
        width (int): 有效的宽度
        height (int): 有效的高度

    Returns:
        Tuple[List[List[int]], int]: 第一项为四个bbox选项，第二项为正确答案的下标
    """
    assert len(gt_bboxes) <= 3, "gt_bboxes长度不能超过3"
    # 根据情况生成额外的随机bbox
    extra_bboxes: List[List[int]] = []
    while len(extra_bboxes) < 4 - len(gt_bboxes):
        tmp = generate_random_bbox(width, height)
        # 保证bbox不重复
        if tmp in extra_bboxes or tmp in gt_bboxes:
            continue
        extra_bboxes.append(tmp)
    if len(gt_bboxes) == 1:
        option_a = [gt_bboxes[0].copy()]
        option_b = [extra_bboxes[0].copy()]
        option_c = [extra_bboxes[1].copy()]
        option_d = [extra_bboxes[2].copy()]
    elif len(gt_bboxes) == 2:
        option_a = [gt_bboxes[0].copy(), gt_bboxes[1].copy()]
        option_b = [gt_bboxes[1].copy(), extra_bboxes[0].copy()]
        option_c = copy.deepcopy(extra_bboxes)
        option_d = [gt_bboxes[0].copy(), extra_bboxes[1].copy()]
    else:
        option_a = [gt_bboxes[0].copy(), gt_bboxes[1].copy(), gt_bboxes[2].copy()]
        option_b = [gt_bboxes[1].copy(), gt_bboxes[2].copy(), extra_bboxes[0].copy()]
        option_c = [gt_bboxes[0].copy(), gt_bboxes[2].copy(), extra_bboxes[0].copy()]
        option_d = [gt_bboxes[0].copy(), gt_bboxes[1].copy(), extra_bboxes[0].copy()]

    final_res = [option_a, option_b, option_c, option_d]
    for i in final_res:
        random.shuffle(i)
    random.shuffle(final_res)

    return final_res, final_res.index(option_a)


def get_instrument_grounding_question(base_path: Path):
    """获取instrument grounding相关的所有问题

    Args:
        base_path (Path): 数据集所存放的目录
    """
    final_data = []
    cholectrack20_path = base_path / "CholecTrack20"
    anno_paths = list(cholectrack20_path.rglob("vid*.json"))
    # 逐个视频处理
    for vid_counter, anno_filepath in enumerate(anno_paths):
        print(f"{vid_counter}/{len(anno_paths)}")
        with open(anno_filepath, "r", encoding="utf-8") as fi:
            json_data = json.load(fi)
        width = json_data["video"]["width"]
        height = json_data["video"]["height"]
        # 器械标注id到名称的映射 int -> str
        tool_name_mapping = {i["id"]: i["name"] for i in json_data["categories"]["tools"]}
        # 逐个帧标注处理
        for frame_id, frame_ann in tqdm(json_data["annotations"].items()):
            # 暂存当前帧的任务，单就器械定位来说，同一帧的若有多种器械则分为多个定位任务
            tasks = []
            img_path = anno_filepath.parent / f"Frames/{frame_id.zfill(6)}.png"
            # 分类暂存各类工具的所有bbox GT
            tool_box_mapping = defaultdict(list)
            for triplet in frame_ann:
                x, y, w, h = triplet["tool_bbox"]
                x_min, y_min, x_max, y_max = (
                    int(x * width),
                    int(y * height),
                    int((x + w) * width),
                    int((y + h) * height),
                )
                tool_box_mapping[triplet["instrument"]].append([x_min, y_min, x_max, y_max])
            # 转换为task
            for tool_id, tool_bboxes in tool_box_mapping.items():
                # 同一帧中出现超过3次的器械类型，跳过不处理
                if len(tool_bboxes) > 3 or tool_id not in tool_name_mapping:
                    continue
                options, gt_index = generate_random_bbox_list(tool_bboxes, width, height)
                tasks.append(
                    {
                        "name": "instrument grounding",
                        "type": "mcq",
                        "question": random.choice(instrument_grounding_question_template).format(
                            instrument=tool_name_mapping[tool_id]
                        ),
                        "options": options,
                        "correct_answer": [gt_index],
                    }
                )
            # 解析相对路径并绑定当前的任务
            if len(tasks) > 0:
                final_data.append({"image_path": img_path.relative_to(base_path).as_posix(), "tasks": tasks})

    return final_data


def get_phase_recognition_question(base_path: Path):
    """获取阶段识别相关的所有问题

    Args:
        base_path (Path): 数据集所存放的目录
    """
    final_data = []
    cholectrack20_path = base_path / "CholecTrack20"
    anno_paths = list(cholectrack20_path.rglob("vid*.json"))
    phase_name_mapping = {
        0: "preparation",
        1: "calot triangle dissection",
        2: "clipping and cutting",
        3: "gallbladder dissection",
        4: "gallbladder packaging",
        5: "cleaning and coagulation",
        6: "gallbladder retraction",
    }
    # 逐个视频处理
    for vid_counter, anno_filepath in enumerate(anno_paths):
        print(f"{vid_counter}/{len(anno_paths)}")
        with open(anno_filepath, "r", encoding="utf-8") as fi:
            json_data = json.load(fi)

        # 逐个帧标注处理
        for frame_id, frame_ann in tqdm(json_data["annotations"].items()):
            img_path = anno_filepath.parent / f"Frames/{frame_id.zfill(6)}.png"

            # 检查同一帧是否标注了多个phase类型，如有则跳过
            phase_id = frame_ann[0]["phase"]
            for ann in frame_ann:
                if ann["phase"] != phase_id:
                    phase_id = -1
                    break
            if phase_id < 0:
                continue

            options = list(phase_name_mapping.values())
            final_data.append(
                {
                    "image_path": img_path.relative_to(base_path).as_posix(),
                    "tasks": [
                        {
                            "name": "phase classification",
                            "type": "mcq",
                            "question": random.choice(phase_classification_question_template),
                            "options": options,
                            "correct_answer": [options.index(phase_name_mapping[phase_id])],
                        }
                    ],
                }
            )

    return final_data


def get_ivt_component_identification_question(base_path: Path):
    final_data = []
    cholectrack20_path = base_path / "CholecTrack20"
    anno_paths = list(cholectrack20_path.rglob("vid*.json"))
    verb_name_mapping = {
        0: "grasp",
        1: "retract",
        2: "dissect",
        3: "coagulate",
        4: "clip",
        5: "cut",
        6: "aspirate",
        7: "irrigate",
        8: "pack",
        9: "null verb",
    }
    target_name_mapping = {
        0: "gallbladder",
        1: "cystic plate",
        2: "cystic duct",
        3: "cystic artery",
        4: "cystic pedicle",
        5: "blood vessel",
        6: "fluid",
        7: "abdominal wall cavity",
        8: "liver",
        9: "adhesion",
        10: "omentum",
        11: "peritoneum",
        12: "gut",
        13: "specimen bag",
        14: "null target",
    }
    phase_name_mapping = {
        0: "preparation",
        1: "carlot triangle dissection",
        2: "clipping and cutting",
        3: "gallbladder dissection",
        4: "gallbladder packaging",
        5: "cleaning and coagulation",
        6: "gallbladder extraction"
    }
    # 逐个视频处理
    for vid_counter, anno_filepath in enumerate(anno_paths):
        print(f"{vid_counter}/{len(anno_paths)}")
        with open(anno_filepath, "r", encoding="utf-8") as fi:
            json_data = json.load(fi)
        # 器械标注id到名称的映射 int -> str
        tool_name_mapping = {i["id"]: i["name"] for i in json_data["categories"]["tools"]}
        # 逐个帧标注处理
        for frame_id, frame_ann in tqdm(json_data["annotations"].items()):
            img_path = anno_filepath.parent / f"Frames/{frame_id.zfill(6)}.png"
            # 暂存当前帧包含的所有器械
            tool_ids, verb_ids, target_ids = set(), set(), set()
            for ann in frame_ann:
                tool_ids.add(ann["instrument"])
                verb_ids.add(ann["verb"])
                target_ids.add(ann["target"])
            correct_answer_i, correct_answer_v, correct_answer_t = (
                sorted(list(tool_ids)),
                sorted(list(verb_ids)),
                sorted(list(target_ids)),
            )
            final_data.append(
                {
                    "image_path": img_path.relative_to(base_path).as_posix(),
                    "tasks": [
                        {
                            "name": "instrument identification",
                            "type": "mcq",
                            "question": random.choice(instrument_identification_question_template),
                            "options": list(tool_name_mapping.values()),
                            "correct_answer": correct_answer_i,
                        },
                        {
                            "name": "verb identification",
                            "type": "mcq",
                            "question": random.choice(verb_identification_question_template),
                            "options": list(verb_name_mapping.values()),
                            "correct_answer": correct_answer_v,
                        },
                        {
                            "name": "target identification",
                            "type": "mcq",
                            "question": random.choice(target_identification_question_template),
                            "options": list(target_name_mapping.values()),
                            "correct_answer": correct_answer_t,
                        },
                    ],
                }
            )

    return final_data


def merge_frame_tasks(frame_reprs: List[dict], collection: Collection):
    def merge_one(frame_doc, collection: Collection):
        existing_doc = collection.find_one({"image_path": frame_doc["image_path"]})
        if existing_doc:
            existing_tasks = existing_doc.get("tasks", [])
            new_tasks = frame_doc["tasks"]
            existing_signatures = {json.dumps(task, sort_keys=True) for task in existing_tasks}
            tasks_to_add = []
            for task in new_tasks:
                if json.dumps(task, sort_keys=True) not in existing_signatures:
                    tasks_to_add.append(task)
            if tasks_to_add:
                result = collection.update_one(
                    {"image_path": frame_doc["image_path"]}, {"$push": {"tasks": {"$each": tasks_to_add}}}
                )
                return result
            else:
                return None
        else:
            result = collection.insert_one(frame_doc)
            return result

    print("merging data...")
    for frame_doc in tqdm(frame_reprs):
        merge_one(frame_doc, collection)


if __name__ == "__main__":
    storage_path = Path("/media/georgehu/backupDisk/dataset/")
    mongo_client = MongoClient("mongodb://localhost:27017")
    db = mongo_client["playground"]
    collection = db["cholectrack20_benchmark"]

    # print(collection.count_documents({}))
    # instru_ground_q = get_instrument_grounding_question(storage_path)
    # num_question = sum([len(i['tasks']) for i in instru_ground_q])
    # print(f'instrument grounding question: {num_question}')
    # collection.insert_many(instru_ground_q)
    # print(collection.count_documents({}))

    # print(collection.count_documents({}))
    # phase_classification_q = get_phase_recognition_question(storage_path)
    # num_question = sum([len(i["tasks"]) for i in phase_classification_q])
    # print(f"phase classification question: {num_question}")
    # merge_frame_tasks(phase_classification_q, collection)
    # print(collection.count_documents({}))

    # print(collection.count_documents({}))
    # ivt_identification_q = get_ivt_component_identification_question(storage_path)
    # num_question = sum([len(i["tasks"]) for i in ivt_identification_q])
    # print(f"ivt component identification question: {num_question}")
    # merge_frame_tasks(ivt_identification_q, collection)
    # print(collection.count_documents({}))
