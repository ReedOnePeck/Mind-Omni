import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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


def extract_and_filter_data():
    """
    该函数执行两个主要任务：
    1. 从问题JSON文件中按顺序提取所有问题到一个列表。
    2. 从描述JSON文件中根据一个排除列表来过滤数据，并将结果存到另一个列表。
    最后返回这两个列表。
    """
    # --- 1. 定义文件路径和需要排除的索引 ---

    # 输入文件路径
    questions_path = '/data0/home/cddu/UniBrain/validate/OneLLM_reasoning/982_reason_Q_calculate.json'
    descriptions_path = '/data0/home/cddu/UniBrain/validate/OneLLM_reasoning/UMBRAVE_sub1.json'

    # 需要被排除（即过滤掉）的描述信息的索引列表
    exclude_indices_list = [
        0, 13, 28, 31, 33, 47, 62, 94, 99, 106, 122, 157, 164, 194, 258, 270, 271,
        280, 285, 340, 353, 408, 411, 422, 427, 444, 473, 489, 510, 534, 541, 542,
        552, 583, 588, 594, 614, 622, 629, 662, 671, 687, 727, 733, 748, 759, 764,
        803, 811, 840, 844, 851, 863, 874, 885, 936, 938, 942, 947
    ]

    # 初始化用于存储结果的列表
    questions_list = []
    descriptions_list_filtered = []

    # --- 2. 处理问题文件 (982_reason_Q_calculate.json) ---

    print(f"--- 正在处理问题文件: {questions_path} ---")
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            # 将整个JSON文件加载到一个Python字典中
            questions_data = json.load(f)

            # 为了确保严格按顺序提取，我们先对字典的键进行排序。
            # 必须使用 `key=int`，这样 "10" 才会排在 "9" 后面，而不是 "1" 后面。
            sorted_keys = sorted(questions_data.keys(), key=int)

            # 遍历排序后的键，按顺序将对应的值（问题）添加到列表中
            for key in sorted_keys:
                questions_list.append(questions_data[key])

            print(f"成功提取了 {len(questions_list)} 条问题。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {questions_path}。请检查路径。")
    except Exception as e:
        print(f"处理问题文件时发生错误: {e}")

    # --- 3. 处理并过滤描述文件 (OneLLM_sub1.json) ---

    print(f"\n--- 正在处理和过滤描述文件: {descriptions_path} ---")
    try:
        # 为了提高查找效率，将需要排除的索引列表转换为集合(set)
        # 判断一个元素是否在集合中，比在列表中快得多
        exclude_set = set(exclude_indices_list)

        with open(descriptions_path, 'r', encoding='utf-8') as f:
            # 加载描述数据的JSON文件
            descriptions_data = json.load(f)

            # 同样，按数字顺序对键进行排序以保证处理顺序
            sorted_keys = sorted(descriptions_data.keys(), key=int)

            # 遍历所有描述数据
            for key in sorted_keys:
                # 将字符串键转换回整数索引，以便在排除集合中检查
                current_index = int(key)

                # 如果当前索引 *不* 在需要排除的集合中
                if current_index not in exclude_set:
                    # 就将这个描述添加到我们的结果列表中
                    descriptions_list_filtered.append(descriptions_data[key])

            print(f"原始描述共 {len(descriptions_data)} 条。")
            print(f"需要排除 {len(exclude_set)} 条。")
            print(f"成功过滤并提取了 {len(descriptions_list_filtered)} 条描述。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {descriptions_path}。请检查路径。")
    except Exception as e:
        print(f"处理描述文件时发生错误: {e}")

    # --- 4. 返回最终结果 ---
    return questions_list, descriptions_list_filtered

questions_list, descriptions_list_filtered = extract_and_filter_data()

model = Qwen2VLForConditionalGeneration.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/qwen2-vl-7B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",).to("cuda")#torch_dtype="auto"
processor = AutoProcessor.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/qwen2-vl-7B")
print("================================================================")



def custom_collate_fn(batch):
    # 分别收集conversation和split_line
    conversations = [item[0] for item in batch]
    return conversations


class conversation_Dataset(Dataset):
    def __init__(self, Q_list, description_list):
        self.Q_list = Q_list
        self.description_list = description_list

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.Q_list)

    def __getitem__(self, idx):
        Q = self.Q_list[idx]
        description = self.description_list[idx]

        prompt1 = description
        prompt2 = Q


        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt1},
                    {"type": "text", "text": prompt2},
                ],
            },
        ]

        return conversation, np.array([111])



train_dataset = conversation_Dataset(questions_list, descriptions_list_filtered)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

As = []
for messages in tqdm(train_loader):
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
        answer = response
        As.append(answer)
        if idx == 0:
            print("rewrite_caption:", answer)


    del texts,image_inputs, video_inputs,inputs,generated_ids,generated_ids_trimmed,output_texts
    torch.cuda.empty_cache()


# 1. 定义输出文件的路径
output_path = "/data0/home/cddu/UniBrain/validate/OneLLM_reasoning/UMBRAVE_sub1_reason.json"
output_dir = os.path.dirname(output_path)  # 获取输出文件所在的目录

print(f"\n--- 推理循环完成，共生成 {len(As)} 条回复 ---")
print(f"准备将结果保存至: {output_path}")

# 2. 确保输出目录存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建目录: {output_dir}")

# 3. 将列表转换为带有从 "0" 开始的顺序数字键的字典
#    - enumerate(As) 会生成 (0, 第一个回复), (1, 第二个回复), ...
#    - str(i) 将数字索引转换为字符串键 "0", "1", ...
#    - 这个字典推导式高效地完成了格式转换，并严格保证了顺序
output_dict = {str(i): value for i, value in enumerate(As)}

# 4. 将最终的字典写入JSON文件
try:
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 json.dump 将字典序列化为JSON格式并写入文件
        # - 不使用 indent 参数，输出的JSON将是紧凑的、没有额外空格和换行的单行格式，与您的示例完全一致
        # - ensure_ascii=False 确保非ASCII字符（如中文）能被正确写入，而不是被转义
        json.dump(output_dict, f, ensure_ascii=False)

    print("\n文件保存成功！")
    print("文件内容格式预览:")
    # 打印前几个键值对以供验证
    for i in range(min(5, len(As))):
        print(f'  "{i}": "{output_dict[str(i)]}"')

except Exception as e:
    print(f"\n保存文件时发生严重错误: {e}")