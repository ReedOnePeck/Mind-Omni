import asyncio
import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
import numpy as np


# ============================================================
# API config
# ============================================================
API_BASE = "http://localhost:9000/v1/chat/completions"
MODEL_NAME = "Qwen3-VL-30B-A3B"
CONCURRENCY = 128
MAX_TOKENS = 128
TEMPERATURE = 0.0


# ============================================================
# Data config
# ============================================================
IMAGE_DIR = Path(
    "/data/home/luyizhuo/Datastation_lyz/Models/Mind_omni_rebuttal/LLM_as_judge/test_images_907_easyreason"
)
GROUNDTRUTH_PATH = Path(
    "/data/home/luyizhuo/Datastation_lyz/Models/Mind_omni_rebuttal/LLM_as_judge/easy_reasoning/easy_reasoning_groundtruth.json"
)
SUBJECT_ANSWER_PATHS = {
    "sub1": Path(
        "/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/results/train_stage1_2_32_32/reformated/sub1_step_1800_easy_reason_shuffle.jsonl"
    ),
    "sub2": Path(
        "/data/home/luyizhuo/Datastation_lyz/Models/Mind_omni_rebuttal/LLM_as_judge/easy_reasoning/sub2_step_1800_easy_reason_reformat.json"
    ),
    "sub5": Path(
        "/data/home/luyizhuo/Datastation_lyz/Models/Mind_omni_rebuttal/LLM_as_judge/easy_reasoning/sub5_step_1800_easy_reason_reformat.json"
    ),
    "sub7": Path(
        "/data/home/luyizhuo/Datastation_lyz/Models/Mind_omni_rebuttal/LLM_as_judge/easy_reasoning/sub7_step_1800_easy_reason_reformat.json"
    ),
}

# question extraction follows train_stage3/validate_stage3_VQA.py:268-342
QUESTION_JSON_DIR = Path(
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/easy_reasoning"
)
TEST_IMG_INDEX_PATH = Path(
    "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_fMRI_MNI_multi/test_data_sub1/test_img_index_start_from0.npy"
)

OUTPUT_DIR = Path(
    "/data/home/luyizhuo/Datastation_lyz/Models/Mind_omni_rebuttal/LLM_as_judge"
)


def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_json_dict(json_path: Path) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def extract_questions() -> List[str]:
    test_img_indices = np.load(TEST_IMG_INDEX_PATH)
    valid_img_nums = []

    for num in test_img_indices:
        json_path = QUESTION_JSON_DIR / f"{int(num):05d}.json"
        if json_path.exists():
            valid_img_nums.append(int(num))

    questions = []
    for num in valid_img_nums:
        json_path = QUESTION_JSON_DIR / f"{num:05d}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        question = data[0]["Question"].replace("<image>", "").replace("\n", "").strip()
        questions.append(question)

    return questions


def validate_alignment(
    questions: List[str],
    groundtruth: Dict[str, str],
    subject_answers: Dict[str, Dict[str, str]],
) -> List[str]:
    sample_ids = [str(i) for i in range(len(questions))]

    if len(questions) != 907:
        raise ValueError(f"Expected 907 questions, got {len(questions)}")

    for sample_id in sample_ids:
        image_path = IMAGE_DIR / f"{sample_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if sample_id not in groundtruth:
            raise KeyError(f"Missing groundtruth answer for sample {sample_id}")
        for subject_name, answers in subject_answers.items():
            if sample_id not in answers:
                raise KeyError(f"Missing answer for {subject_name}, sample {sample_id}")

    return sample_ids


def build_eval_texts(question: str, reference_answer: str, candidate_answer: str) -> Tuple[str, str]:
    text1 = (
        "你是一个宽松的视觉问答质检员。请结合图片、问题、标准答案，判断候选答案是否回答正确，由于输入信息的信噪比比较低，因此判卷的时候可以稍微宽容一些。"
        "判断标准以是否正确回答问题为主，不要求措辞和标准答案完全一致。"
        "如果候选答案与标准答案语义一致，或者虽然简短但正确回答了问题，判为正确。"
        "如果候选答案答非所问、事实错误、与图片明显不符，判为错误。\n\n"
        f"问题：{question}\n"
        f"标准答案：{reference_answer}"
    )
    text2 = (
        f"候选答案：{candidate_answer}\n\n"
        "请只输出一个JSON对象，不要输出任何额外解释。格式必须是："
        '{"verdict":"正确"或"错误","reason":"一句话原因"}'
    )
    return text1, text2


def parse_verdict(content: str) -> Tuple[str, str]:
    content = content.strip()

    try:
        parsed = json.loads(content)
        verdict = str(parsed.get("verdict", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
    except Exception:
        if "正确" in content and "错误" not in content:
            verdict, reason = "正确", content
        elif "错误" in content and "正确" not in content:
            verdict, reason = "错误", content
        else:
            verdict, reason = "解析失败", content

    if verdict not in {"正确", "错误"}:
        verdict = "解析失败"
    return verdict, reason


async def single_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    img_data_url: str,
    text1: str,
    text2: str,
    subject_name: str,
    sample_id: str,
) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img_data_url},
                    },
                    {
                        "type": "text",
                        "text": f"{text1}\n\n{text2}",
                    },
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    async with semaphore:
        try:
            async with session.post(API_BASE, json=payload) as resp:
                result = await resp.json()
                content = result["choices"][0]["message"]["content"]
                verdict, reason = parse_verdict(content)
                print(f"[{subject_name}][{sample_id}] {verdict}")
                return {
                    "subject": subject_name,
                    "sample_id": sample_id,
                    "status": "ok",
                    "raw_output": content,
                    "verdict": verdict,
                    "reason": reason,
                }
        except Exception as e:
            print(f"[{subject_name}][{sample_id}] 失败：{e}")
            return {
                "subject": subject_name,
                "sample_id": sample_id,
                "status": "error",
                "raw_output": None,
                "verdict": "解析失败",
                "reason": str(e),
            }


async def run_concurrent(tasks_payload: List[dict]) -> List[dict]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(connect=10, total=180)

    async with aiohttp.ClientSession(
        headers={"Authorization": "Bearer none"},
        timeout=timeout,
    ) as session:
        tasks = [
            single_request(
                session=session,
                semaphore=semaphore,
                img_data_url=item["img_data_url"],
                text1=item["text1"],
                text2=item["text2"],
                subject_name=item["subject"],
                sample_id=item["sample_id"],
            )
            for item in tasks_payload
        ]
        return await asyncio.gather(*tasks)


def prepare_requests(
    sample_ids: List[str],
    questions: List[str],
    groundtruth: Dict[str, str],
    subject_answers: Dict[str, Dict[str, str]],
) -> List[dict]:
    image_cache = {}
    request_items = []

    for sample_id in sample_ids:
        image_path = IMAGE_DIR / f"{sample_id}.png"
        image_cache[sample_id] = f"data:image/png;base64,{load_image_base64(image_path)}"

    for sample_id in sample_ids:
        question = questions[int(sample_id)]
        reference_answer = groundtruth[sample_id]
        for subject_name, answers in subject_answers.items():
            candidate_answer = answers[sample_id]
            text1, text2 = build_eval_texts(question, reference_answer, candidate_answer)
            request_items.append(
                {
                    "subject": subject_name,
                    "sample_id": sample_id,
                    "img_data_url": image_cache[sample_id],
                    "text1": text1,
                    "text2": text2,
                }
            )

    return request_items


def summarize_results(results: List[dict]) -> Dict[str, dict]:
    summary = {}
    for subject_name in SUBJECT_ANSWER_PATHS:
        subject_results = [r for r in results if r["subject"] == subject_name]
        valid_results = [r for r in subject_results if r["verdict"] in {"正确", "错误"}]
        correct_count = sum(1 for r in valid_results if r["verdict"] == "正确")
        total_count = len(subject_results)
        accuracy = correct_count / total_count if total_count else 0.0
        summary[subject_name] = {
            "total": total_count,
            "valid_judged": len(valid_results),
            "correct": correct_count,
            "accuracy": accuracy,
        }
    return summary


def save_outputs(results: List[dict], summary: Dict[str, dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_path = OUTPUT_DIR / "easy_reasoning_qc_detailed_results_kuansong.json"
    summary_path = OUTPUT_DIR / "easy_reasoning_qc_summary_kuansong.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {result_path}")
    print(f"汇总结果已保存到: {summary_path}")


def main():
    questions = extract_questions()
    groundtruth = load_json_dict(GROUNDTRUTH_PATH)
    subject_answers = {
        subject_name: load_json_dict(answer_path)
        for subject_name, answer_path in SUBJECT_ANSWER_PATHS.items()
    }

    sample_ids = validate_alignment(questions, groundtruth, subject_answers)
    request_items = prepare_requests(sample_ids, questions, groundtruth, subject_answers)

    print(f"将要发送 {len(request_items)} 个请求，最大并发 {CONCURRENCY}")
    results = asyncio.run(run_concurrent(request_items))
    summary = summarize_results(results)
    save_outputs(results, summary)

    print("\n===== 准确率统计 =====")
    for subject_name, metrics in summary.items():
        print(
            f"{subject_name}: "
            f"correct={metrics['correct']}, "
            f"total={metrics['total']}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"valid_judged={metrics['valid_judged']}"
        )


if __name__ == "__main__":
    main()
