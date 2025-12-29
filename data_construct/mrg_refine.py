from openai_concurrent import OpenAIConcurrentClient
from pathlib import Path
import json
import re
import asyncio
from tqdm import tqdm
import os


BASE_URL = "https://api.siliconflow.cn/v1"

MRG_REFINE_TEMPLATE = """User:
Please summarize the following medical report in three labeled lines:
Observation: [concise description of visible findings, progress, and instruments]
Clinical Significance: [explain clinical meaning or purpose of this phase]
Next Steps: [likely upcoming actions or considerations]
Avoid repetition and ensure medical precision. Please provide your answer in a clear, text-based format, but do not use any markdown syntax.

Medical Report:
{context}
"""

SYSTEM_PROMPT = """You are an AI assistant specializing in surgical video analysis.
You can also imagine yourself as a lecturer on surgery who explains surgeons' thought processes and other surgical rationales to new junior surgeons who ask you questions.
Your task is to provide detailed, medically accurate, and contextually relevant responses for surgical video frame analysis questions."""


def extract_id_from_filepath(image_path: Path):
    frameid = int(image_path.stem)
    videoid = int(image_path.parent.parent.stem[3:])

    return videoid, frameid


def count_chinese_chars(text):
    """统计字符串中汉字的个数"""
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return sum(1 for _ in pattern.finditer(text))


def clean_markdown_detailed(text):
    """分步去除markdown格式"""
    # 去除粗斜体格式
    text = re.sub(r'\*{3}([^*]+)\*{3}', r'\1', text)
    # 去除加粗格式
    text = re.sub(r'\*{2}([^*]+)\*{2}', r'\1', text)
    # 去除斜体格式
    text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'\1', text)
    # 去除标题标记
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    return text


def clean_special_char(text):
    """其他的特殊替换"""
    # “中文 (对应英文)”形式的替换为其中的英文
    text = re.sub(r'[\u4e00-\u9fff]+ \(([a-z0-9 ]+)\)',r' \1', text)
    # 规范撇号
    text = re.sub(r'\u2019', "'", text)
    # 规范双引号
    text = re.sub(r'[\u201c\u201d]', '"', text)
    # 规范短横杠
    text = re.sub(r'[\u2013\u2014]', '-', text)
    # 移除think标签
    text = text.replace("</think>", "")
    text = text.replace("<think>", "")
    # 移除代码段符号
    text = text.replace("```", "")
    # 移除无意义空格
    text = re.sub(r'\u2003', '', text)
    
    return text


async def main():
    mrg_jsonl_filepath = Path('/home/georgehu/Documents/Projects/surg-agent/output/cleaned_mrg.jsonl')
    log_filepath = Path("log1.jsonl.gz")
    refined_text_savepath = Path('/home/georgehu/Documents/Projects/surg-agent/refined_mrg_patch.json')
    extract_pattern = r'Observation:(.*?)Clinical Significance:(.*?)Next Steps:(.*)'
    batch_size = 10

    myclient = OpenAIConcurrentClient(
        os.getenv("SILICLOUD_API_KEY"),
        BASE_URL,
        log_archieve_filepath=log_filepath,
        max_requests_per_minute=1500,
        max_tokens_per_minute=35000,
        max_concurrent=20,
        max_retries=3,
    )

    with open(mrg_jsonl_filepath, 'r', encoding='utf-8') as fi:
        cleaned_mrg = [json.loads(line) for line in fi.readlines()]
    
    retry_frames_set = {
        # '110_37476', '6_59176', '4_25651', '2_28551', '23_30451', '7_30401', '110_8926', '92_50001', '37_33251', '103_10451', '30_23901', '11_676', '103_44901', '37_15276', '23_21051', '6_24676', '17_13676', '17_2876', '4_18151', '17_31726', '31_12851', '111_6251', '7_2451', '110_46576', '11_24001', '2_50651', '17_31801', '13_13051', '31_20126', '4_22976', '110_48101', '30_33226', '1_41526', '37_28401', '96_11126', '6_13201', '6_30651', '7_5201', '17_15826', '92_14451', '1_30251', '2_72401', '7_64701', '110_34276', '4_8801', '4_24051', '11_62026', '103_29301', '1_2251', '92_21326', '37_17826', '25_30901', '110_6451', '13_2176', '31_23501', '30_32726', '37_25601', '7_86001', '103_25901', '103_19326', '23_25951', '6_23451'
    }

    # 存放构造出的各批次请求内容
    request_batches = []

    print(f'开始构建{len(cleaned_mrg)}条请求内容........')
    for mrg_record in cleaned_mrg:
        videoid, frameid = extract_id_from_filepath(Path(mrg_record["image_path"]))
        if len(retry_frames_set) > 0 and f'{videoid}_{frameid}' not in retry_frames_set:
            continue
        model_answer = mrg_record['tasks'][0]['model_answer']
        request_batches.append(
            {
                "id": f"{videoid}_{frameid}",
                "request": dict(
                    model="Qwen/Qwen3-32B",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": MRG_REFINE_TEMPLATE.format(context=model_answer),
                                },
                            ],
                        },
                    ],
                    extra_body={"thinking": {"type": "disabled"}},
                ),
            }
        )
    # 批次分割
    request_batches = [request_batches[i : i + batch_size] for i in range(0, len(request_batches), batch_size)]
    print('OK')

    def progress_callback(completed: int, total: int):
        print(f"当前批次进度: {completed}/{total} ({completed/total*100:.1f}%)")

    print("开始请求调用")
    # 存放请求失败任务的执行结果
    failed_request = []
    output_dict = {}
    try:
        # 逐批次请求
        for batch in tqdm(request_batches):
            # 执行一个批次的请求
            results = await myclient.batch_requests(batch, progress_callback)
            for result in results:
                videoid_string, frameid_string = result.request_id.split("_")
                expand_request_id = f'{videoid_string.zfill(2)}_{frameid_string.zfill(6)}'

                if not result.success:
                    failed_request.append(result)
                    continue
                
                model_answer = result.response["choices"][0]["message"]["content"]
                model_answer = model_answer.replace("<|end_of_box|>", "").replace("<|begin_of_box|>", "").replace("<think>", "").replace("</think>", "")
                model_answer = clean_markdown_detailed(clean_special_char(model_answer))
                # 解析内容结构，匹配不上视为失败
                match = re.search(extract_pattern, model_answer, re.IGNORECASE | re.DOTALL)
                if match is None:
                    failed_request.append(result)
                    continue
                else:
                    observation = match.group(1).strip()
                    clinical_significance = match.group(2).strip()
                    next_steps = match.group(3).strip()
                output_dict[expand_request_id] = (observation, clinical_significance, next_steps)
                
            # 每执行一个批次保存一下benchmark记录
            with open(refined_text_savepath, "w", encoding="utf-8") as fo:
                json.dump(output_dict, fo)

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
        print(f"final failed list: {failed_request_ids}")
        await myclient.close()


if __name__ == "__main__":
    asyncio.run(main())
