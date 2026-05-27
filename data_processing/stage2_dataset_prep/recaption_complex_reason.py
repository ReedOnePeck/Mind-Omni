import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import numpy as np
import h5py
import json
import glob


model = Qwen2VLForConditionalGeneration.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/qwen2-vl-7B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",).to("cuda")#torch_dtype="auto"
processor = AutoProcessor.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/qwen2-vl-7B")
print("================================================================")


def save_qa_as_json(file_name, question, answer, output_dir):
    """
    将问答对按照指定格式保存为JSON文件。

    Args:
        file_name (str): 输出的JSON文件名 (不含扩展名, 例如 "02954")。
        question (str): 问题文本。
        answer (str): 答案文本。
        output_dir (str): 保存JSON文件的目标目录。
    """
    # 1. 确保目标目录存在，如果不存在则创建它
    os.makedirs(output_dir, exist_ok=True)

    # 2. 按照您指定的格式组织数据
    #    注意：这里根据您的示例，将 <image> 标记添加回了问题的末尾
    #    如果您不希望添加 <image>，可以改为 "Question": question
    output_data = [
        {
            "Question": question,
            "Answer": answer
        }
    ]

    # 3. 构建完整的文件路径
    #    例如: /data0/home/cddu/UniBrain/recaptioned_complex_reasoning/02954.json
    file_path = os.path.join(output_dir, f"{file_name}.json")

    # 4. 将数据写入JSON文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 使用 json.dump() 来写入文件
            # indent=2 会让JSON文件格式化，更易于阅读
            # ensure_ascii=False 确保中文字符或其他非ASCII字符能被正确写入
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"写入文件 {file_path} 时发生错误: {e}")



def custom_collate_fn(batch):
    # 分别收集conversation和split_line
    conversations = [item[0] for item in batch]
    json_name = [item[1] for item in batch]
    question = [item[2] for item in batch]
    return conversations, json_name, question


class conversation_Dataset(Dataset):
    def __init__(self, raw_complex_reason_path):
        """
        数据集的构造函数（初始化方法）。
        """
        self.path = raw_complex_reason_path
        # 初始化一个空列表，用于存储所有预处理过的数据
        # 每个元素将是一个包含 Q, A 和 id 的字典
        self.qa_pairs = []
        self._load_data()

    def _load_data(self):
        """
        一个内部辅助方法，用于扫描目录、读取JSON文件、提取Q&A和文件名ID。
        """
        print(f"正在从以下路径初始化数据集: {self.path}")
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"错误: 在'{self.path}'处未找到目录")

        json_files = glob.glob(os.path.join(self.path, '*.json'))

        if not json_files:
            print(f"警告: 在'{self.path}'目录中没有找到任何JSON文件")
            return

        print(f"找到了 {len(json_files)} 个JSON文件。正在加载和预处理...")

        for file_path in tqdm(json_files, desc="加载数据中"):
            try:
                # --- 修改点 1: 记录文件名中的数字ID ---
                # 从完整路径中获取文件名 (例如: "02954.json")
                base_name = os.path.basename(file_path)
                # 分离文件名和扩展名，并获取文件名部分 (例如: "02954")
                file_id = os.path.splitext(base_name)[0]

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        raw_question = item.get("Question", "未找到问题")
                        answer = item.get("Answer", "未找到答案")
                        question = raw_question.replace("<image>\n", "").strip()

                        # 将问答对和文件名ID一起存入列表
                        self.qa_pairs.append({"Q": question, "A": answer, "id": file_id})

            except json.JSONDecodeError:
                print(f"\n警告: 无法解析文件 {os.path.basename(file_path)} 的JSON内容。已跳过。")
            except Exception as e:
                print(f"\n处理文件 {os.path.basename(file_path)} 时发生未知错误: {e}。已跳过。")

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        """
        根据给定的索引 `idx`，获取数据集中对应的单个样本。

        Args:
            idx (int): 样本的索引。

        Returns:
            tuple: (conversation, file_id)
                   - conversation: 包含处理好的对话数据的列表
                   - file_id: 从文件名中提取的5位数字字符串 (例如: "02954")
        """
        # 1. 从预处理好的列表中，根据索引获取对应的问答对和ID
        item = self.qa_pairs[idx]
        Q = item['Q']
        A = item['A']
        file_id = item['id'] # 获取文件名ID

        prompt1 = 'You are a professional text summarizer. Your task is to rephrase and condense the given Answer of a Q&A pair to approximately 30 words. You must preserve the core meaning, key details, and original tone. Remove any redundant phrases or unnecessary elaborations.'
        prompt2 = f"Original reasoning Q&A: Question: {Q}. Answer: {A}"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt1},
                    {"type": "text", "text": prompt2},
                ],
            },
        ]

        # --- 修改点 2: 返回文件名ID而不是占位符 ---
        return conversation, file_id, Q



train_dataset = conversation_Dataset(raw_complex_reason_path = "/data0/home/cddu/UniBrain/complex_reasoning/")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)


for messages, file_ids, questions in tqdm(train_loader):
    with torch.no_grad():
        texts = [
            processor.apply_chat_template(msg, add_generation_prompt=True, add_vision_id=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=768)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    for idx, response in enumerate(output_texts):
        file_name = file_ids[idx]
        question = questions[idx]
        answer = response
        save_qa_as_json(file_name, question, answer, "/data0/home/cddu/UniBrain/recaptioned_complex_reasoning/")
        print(response)


    del texts,image_inputs, video_inputs,inputs,generated_ids,generated_ids_trimmed,output_texts
    torch.cuda.empty_cache()





