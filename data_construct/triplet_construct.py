import json
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection
from tqdm import tqdm


def get_triplet_labels():
    type_mapping_videoid = {
        "Testing": [1, 6, 7, 12, 25, 39, 92, 111],
        "Validation": [30, 110],
        "Training": [2, 4, 11, 13, 17, 23, 31, 37, 96, 103],
    }
    videoid_mapping_type = {
        video_id: category for category, ids in type_mapping_videoid.items() for video_id in ids
    }

    selected_frame_json_filepath = Path("data_construct/selected_frameid.json")
    storage_path = Path("/media/georgehu/backupDisk/dataset/CholecTrack20")

    video_ids = [30, 110, 1, 6, 7, 12, 25, 39, 92, 111, 2, 4, 11, 13, 17, 23, 31, 37, 96, 103]
    with open(selected_frame_json_filepath, "r", encoding="utf-8") as fi:
        selected_frame = json.load(fi)
        selected_frame = {int(k[3:]): v for k, v in selected_frame.items() if int(k[3:]) in video_ids}


    jsonl = []

    for videoid, frameid_list in selected_frame.items():
        annotation_filepath = storage_path / videoid_mapping_type[videoid] / f'VID{str(videoid).zfill(2)}/vid{str(videoid).zfill(2)}.json'
        with open(annotation_filepath, 'r', encoding='utf-8') as fi:
            raw_anno = json.load(fi)
            annos = raw_anno['annotations']
            for frameid in frameid_list:
                frame_anno = annos[str(frameid)]
                triplet_label = list(set([i['triplet'] for i in frame_anno]))
                triplet_label.sort()
                jsonl.append(
                    {
                        "image_path": f"CholecTrack20/{videoid_mapping_type[videoid]}/VID{str(videoid).zfill(2)}/Frames/{str(frameid).zfill(6)}.png",
                        "tasks": [
                            {
                                "name": "triplet recognition",
                                "type": "triplet",
                                "correct_answer": triplet_label
                            }
                        ],
                    }
                )

    # with open('output/triplet_label.jsonl', 'w', encoding='utf-8') as fo:
    #     fo.write('\n'.join([json.dumps(line) for line in jsonl]))
    
    return jsonl


def merge_triplet_labels(collection: Collection, triplet_label_filepath: Path):
    with open(triplet_label_filepath, 'r', encoding='utf-8') as fi:
        triplet_labels = [json.loads(line) for line in fi.readlines()]
    
    print('开始合并triplet recognition任务......')
    counter = 0
    for triplet_record in tqdm(triplet_labels):
        # 先查询是否存在该帧的记录
        existing_doc = collection.find_one({'image_path': triplet_record['image_path']})
        if not existing_doc:
            result = collection.insert_one(triplet_record)
        else:
            # 检查目标帧是否已存在triplet recognition任务记录
            existing_tasks = existing_doc.get('tasks', [])
            # 若已存在triplet recognition记录则跳过，不覆盖
            if len(list(filter(lambda x: x['name'] == 'triplet recognition', existing_tasks))) > 0:
                continue
            # 不存在则增量更新tasks列表
            result = collection.update_one(
                {"image_path": triplet_record["image_path"]},
                {"$push": {"tasks": {"$each": triplet_record["tasks"]}}},
            )
        if result.modified_count == 0:
            print(f"{triplet_record['image_path']}失败:")
            print(result)
        else:
            counter += result.modified_count
    print(f'完成triplet recognition任务合并，共修改了{counter}条记录')


if __name__ == "__main__":
    get_triplet_labels()

    # mongo_client = MongoClient("mongodb://localhost:27017")
    # db = mongo_client["playground"]
    # collection = db["cholectrack20_benchmark"]
    # merge_triplet_labels(collection, Path('output/triplet_label.jsonl'))