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


Question_pool = ["What do you see happening in this image?",
                "What do you think is going on in this snapshot?",
                "Can you elaborate on the elements of the picture provided?",
                "Describe the following image.",
                "Write a detailed description of the given image.",
                 "Explain the visual content of the image in great detail.",
                "Analyze the image in a comprehensive and detailed manner.",
                "Can you describe the main features of this image for me?",
                 "Describe the following scene.",
                "What are the key elements in this picture?"
                 ]


def create_unified_caption_list(json_dir, txt_dir):
    """
    整合两个来源的文本数据，生成一个包含73000个描述的统一列表。

    该函数会顺次遍历0到72999的索引。对于每个索引：
    1. 优先从 json_dir 目录中查找对应的JSON文件并提取 "Answer"。
    2. 如果JSON文件不存在，则从 txt_dir 目录中读取对应的TXT文件内容作为备选。
    3. 同时，记录下每个处理的索引（五位数字格式）。

    Args:
        json_dir (str): 存有JSON文件的目录路径 (例如 /data0/home/cddu/UniBrain/detail)。
        txt_dir (str): 存有TXT文件的目录路径 (例如 /.../COCO_captions_recapted_Qw2VL)。

    Returns:
        tuple: (captions, indices)
               - captions (list): 包含73000个字符串的列表，内容为 "Answer" 或 txt 文件内容。
               - indices (list): 包含73000个对应的五位数字字符串的列表 (从 "00000" 到 "72999")。
    """
    # 1. 初始化两个空列表，用于存储最终结果
    captions_list = []
    indices_list = []
    total_count = 73000

    print(f"开始处理 {36500} 个文件...")
    print(f"JSON 目录 (优先): {json_dir}")
    print(f"TXT 目录 (备选): {txt_dir}")

    # 2. 顺次遍历 0 到 72999
    #    使用 tqdm 来显示一个清晰的进度条
    for i in tqdm(range(0, 36500), desc="正在整合描述"):
        # 将当前数字格式化为五位数的字符串 (例如: 5 -> "00005")
        file_id = f"{i:05d}"

        # 构建JSON文件的完整路径
        json_file_path = os.path.join(json_dir, f"{file_id}.json")

        # 默认的文本内容为空字符串，以防两个文件都找不到
        text_content = ""

        # 3. 检查JSON文件是否存在
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 提取 "Answer" 内容，并移除首尾可能存在的空白字符
                    text_content = data[0].get("Answer", "").strip()
            except Exception as e:
                # 如果JSON文件损坏或格式不正确，则打印错误并尝试使用TXT文件
                print(f"\n读取JSON文件 {json_file_path} 时出错: {e}。将尝试使用备选TXT文件。")
                # 构建TXT文件的完整路径
                txt_file_path = os.path.join(txt_dir, f"{file_id}.txt")
                if os.path.exists(txt_file_path):
                    with open(txt_file_path, 'r', encoding='utf-8') as f_txt:
                        text_content = f_txt.read().strip()
        else:
            # 如果JSON文件不存在，则读取对应的TXT文件
            txt_file_path = os.path.join(txt_dir, f"{file_id}.txt")
            if os.path.exists(txt_file_path):
                try:
                    with open(txt_file_path, 'r', encoding='utf-8') as f_txt:
                        text_content = f_txt.read().strip()
                except Exception as e:
                    print(f"\n读取TXT文件 {txt_file_path} 时出错: {e}。")
            else:
                # 连TXT文件也找不到时，打印一个警告
                print(f"\n警告: {json_file_path} 和 {txt_file_path} 均未找到。")

        # 4. 将获取到的文本内容和五位数字ID分别存入列表
        captions_list.append(text_content)
        indices_list.append(file_id)

    print("处理完成！")
    return captions_list, indices_list

def custom_collate_fn(batch):
    # 分别收集conversation和split_line
    conversations = [item[0] for item in batch]
    json_name = [item[1] for item in batch]
    raw_caption = [item[2] for item in batch]
    return conversations, json_name, raw_caption



class conversation_Dataset(Dataset):
    def __init__(self, captions_list, indices_list):
        self.captions_list = captions_list
        self.indices_list = indices_list

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.indices_list)

    def __getitem__(self, idx):
        raw_caption = self.captions_list[idx]
        file_id = self.indices_list[idx]

        prompt1 = 'Your task is to act as an expert editor. Rephrase the following sentences to be clear, concise, and around 30 words, while strictly preserving the original meaning. \
                   Follow the approach demonstrated in the three examples provided below.'
        prompt2 = " **Example 1:**  \
                    **Original:** At a bustling outdoor market, a man in a white shirt and green cap sorts through a large pile of vibrant orange carrots, surrounded by bags of potatoes and other fresh produce. \
                    **Rephrased:** The image depicts a bustling open-air market filled with people shopping for vegetables. Large piles of carrots and potatoes are prominently displayed throughout the market, drawing the shoppers' attention. At least sixteen people can be seen browsing, interacting, and shopping in the market area, some very close to the vegetable piles. \
                    **Example 2:**  \
                    **Original:** The image showcases a well-lit display case filled with an array of cakes. The cakes vary in size and design, with some adorned with colorful frosting and others featuring intricate decorations. The case is made of glass, allowing a clear view of the cakes inside. The lighting highlights the vibrant colors and textures of the cakes, creating an inviting and appetizing display.  \
                    **Rephrased:** The image depicts a beautifully illuminated display case in a room, exhibiting a variety of cakes on several shelves. There are several cakes in different shapes, sizes, and styles arranged neatly within the case. These delectable treats are the central focus, as the case lighting effectively draws attention to the cakes amidst the surrounding darkness. \
                    **Example 3:**  \
                    **Original:** Three friends sit at a wooden table in a cozy cafe, posing for a portrait. The woman wears a black and white dress, the man in a gray shirt, and the third person in a dark top. The table is set with clear glasses, a pitcher of beer, and a bowl of pink flowers. A palm tree mural and a blue lotus symbol adorn the yellow walls. \
                    **Rephrased:** The image depicts a scene that three friends are sitting in a cafe around a wooden dining table. There are multiple glasses full of drinks placed on the table, as well as cups located in various positions. The cafe features a cozy atmosphere, with chairs surrounding the table and potted plants placed in the background, adding to the ambiance of the space. \
                    "

        prompt3 = f"**Now, rephrase this text: {raw_caption}"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt1},
                    {"type": "text", "text": prompt2},
                    {"type": "text", "text": prompt3},
                ],
            },
        ]

        return conversation, file_id, raw_caption




captions_list, indices_list = create_unified_caption_list(json_dir='/data0/home/cddu/UniBrain/detail/', txt_dir='/nfs/diskstation/DataStation/public_dataset/NSD_complete/COCO_captions_recapted_Qw2VL/')


train_dataset = conversation_Dataset(captions_list, indices_list)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)


for messages, file_ids, raw_captions in tqdm(train_loader):
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
        raw_caption = raw_captions[idx]
        answer = response
        question = random.choice(Question_pool)
        save_qa_as_json(file_name, question, answer, "/data0/home/cddu/UniBrain/recaptioned_detail/")
        if idx == 0:
            print("raw_caption:",raw_caption)
            print("rewrite_caption:", answer)


    del texts,image_inputs, video_inputs,inputs,generated_ids,generated_ids_trimmed,output_texts
    torch.cuda.empty_cache()




