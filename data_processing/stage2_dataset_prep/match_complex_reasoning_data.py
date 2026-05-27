import os
import json
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def process_nsd_to_llava():
    """
    将NSD数据集中的刺激图像与LLaVA数据集中的对话数据进行匹配，
    并根据匹配结果生成相应的JSON文件或记录未匹配项。
    """
    # 1. 定义文件和文件夹路径
    stim_order_f = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_IDs/nsd_expdesign.mat"
    coco_id_template = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_IDs/cocoId_sub{sub_id}.npy"
    llava_json_path = "/data/home/luyizhuo/Datastation_lyz/Datasets/LLaVA-Instruct-150k/complex_reasoning_77k.json"
    output_dir = "/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/raw_data/complex_reasoning/"
    unmatched_npy_path = os.path.join(output_dir, "unmatched_nsd_indices.npy")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已确认: {output_dir}")

    # 2. 加载并预处理LLaVA对话数据为字典以便快速查找
    print("正在加载并预处理LLaVA JSON数据...")
    llava_data_map = {}
    with open(llava_json_path, 'r', encoding='utf-8') as f:
        llava_data = json.load(f)
        for item in llava_data:
            # 将COCO ID字符串转换为整数作为键
            coco_id = int(item['id'])
            # 提取对话内容
            conversation = item['conversations']
            question = ""
            answer = ""
            if len(conversation) >= 2:
                # 确保对话数据格式正确
                if conversation[0]['from'] == 'human' and conversation[1]['from'] == 'gpt':
                    question = conversation[0]['value']
                    answer = conversation[1]['value']

            if question and answer:
                llava_data_map[coco_id] = {"Question": question, "Answer": answer}
    print(f"LLaVA数据加载完毕，共 {len(llava_data_map)} 条有效对话。")

    # 3. 加载NSD刺激顺序数据
    print("正在加载NSD刺激顺序数据...")
    stim_order = loadmat(stim_order_f)['subjectim']  # shape: (8, 10000)
    print("NSD数据加载完毕。")

    # 4. 初始化用于记录的列表和集合
    unmatched_list = []
    processed_nsd_indices = set()

    # 5. 遍历每个被试的数据进行匹配
    total_subjects = stim_order.shape[0]
    total_trials = stim_order.shape[1]

    for sub_id in range(1, total_subjects + 1):
        print(f"\n--- 正在处理被试 {sub_id}/{total_subjects} ---")

        # 加载当前被试的COCO ID数据
        coco_ids_path = coco_id_template.format(sub_id=sub_id)
        if not os.path.exists(coco_ids_path):
            print(f"警告: 找不到文件 {coco_ids_path}，跳过被试 {sub_id}")
            continue

        subject_coco_ids = np.load(coco_ids_path)  # shape: (10000,)
        subject_nsd_indices = stim_order[sub_id - 1, :]  # 获取当前被试的NSD标号行

        # 使用tqdm显示处理进度
        for i in tqdm(range(total_trials), desc=f"被试 {sub_id} 进度"):
            nsd_index = subject_nsd_indices[i]

            # 检查此NSD图像是否已被处理过，避免重复工作
            if nsd_index in processed_nsd_indices:
                continue

            # 如果未处理，则标记为已处理
            processed_nsd_indices.add(nsd_index)

            coco_id = subject_coco_ids[i]

            # 在LLaVA数据字典中查找匹配项
            if coco_id in llava_data_map:
                # 匹配成功
                conversation_data = llava_data_map[coco_id]
                output_data = [conversation_data]

                # NSD标号从1开始，保存时需要减1
                file_index = nsd_index - 1
                output_filename = f"{file_index:05d}.json"
                output_filepath = os.path.join(output_dir, output_filename)

                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    json.dump(output_data, f_out, indent=2, ensure_ascii=False)
            else:
                # 匹配失败，记录下来
                unmatched_list.append(nsd_index - 1)

    # 6. 处理完毕，保存并打印未匹配的列表
    print("\n--- 所有被试处理完毕 ---")

    # 保存未匹配的NSD标号-1列表
    np.save(unmatched_npy_path, np.array(unmatched_list))
    print(f"总共找到 {len(unmatched_list)} 个未匹配的图像。")
    print(f"未匹配的图像索引（NSD标号-1）已保存至: {unmatched_npy_path}")

    # 打印未匹配的列表内容
    print("\n未匹配的NSD索引 (NSD标号-1) 列表:")
    print(unmatched_list)

    # 统计成功匹配的数量
    total_unique_stim = len(processed_nsd_indices)
    matched_count = total_unique_stim - len(unmatched_list)
    print(f"\n在 {total_unique_stim} 张独立刺激图像中，成功匹配 {matched_count} 张。")


if __name__ == '__main__':
    process_nsd_to_llava()




"""
1. 我有NSD数据集中8名被试观看的刺激图像的NSD序号（从1到73000）
上述数据存在stim_order_f = “/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_IDs/nsd_expdesign.mat”
你可以通过stim_order = loadmat(stim_order_f)['subjectim']获取，stim_order是一个形状为（8，10000）的ndarrray,其中8行中的每一行都代表一个被试，依次从1到8，每一列都是每个被试的刺激图像的NSD序号（从1到73000），其中所有被试每一行的前1000个数据都是一模一样的，你在后续处理的时候注意

2.我还有每个被试的这10000张刺激图像对应的COCO_ID，分别存在/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/COCO_IDs/cocoId_sub{sub_id}.npy中，sub_id取值为1-8。
每一个npy文件加载出来都是形状为（10000，）的ndarray，里面存的每一个数都是当前刺激的COCO_ID，并且和stim_order中对应被试所在的行的数据是一一对应的。

3.我在/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/NSD_imgs中存了所有被试观看的刺激图像，总计73000张，标号从0到72999.你需要注意的是，我保存图像的时候的标号从0开始，而NSD序号从1开始

4./data/home/luyizhuo/Datastation_lyz/Datasets/LLaVA-Instruct-150k/complex_reasoning_77k.json
该数据打开的前几组数据长这样：
[
  {
    "id": "000000215677",
    "image": "000000215677.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat skill set might someone need to perform such a frisbee trick?"
      },
      {
        "from": "gpt",
        "value": "To perform the frisbee trick shown in the image, where the man is passing a frisbee between or underneath his legs, a person would need a combination of skills. These skills include good hand-eye coordination, agility, balance, flexibility, and dexterity. Additionally, the ability to throw and catch the frisbee accurately while maintaining control of bodily movements would also be essential. To perfect the trick, practicing these skills and building up muscle memory through repetition would be beneficial."
      }
    ]
  },
  {
    "id": "000000296754",
    "image": "000000296754.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat precautions is the woman taking while walking in this weather?"
      },
      {
        "from": "gpt",
        "value": "The woman is taking precautions while walking in the rainy weather by using an open umbrella to shield herself from the rain. The umbrella helps to keep her dry and protected from getting soaked by the rain, ensuring her comfort and safety while walking down the wet street. Holding the umbrella also indirectly indicates that the woman might be more attentive to her surroundings and cautious of potential hazards caused by the wet conditions, such as slippery surfaces, puddles, or splashing from passing vehicles."
      }
    ]
  },
]

每条数据中的 "id"指的就是COCO_ID

5.我需要你用python编写一段程序，执行以下任务：
（1）读取stim_order_f中每一个被试的NSD刺激图像标号，对于每一个被试，在cocoId_sub{sub_id}.npy的对应位置找到该NSD图像对应的COCO_ID是多少
（2）拿着这个COCO_ID，在complex_reasoning_77k.json数据的id中找到匹配的对话数据（你在匹配的时候需要注意，要去掉"id"前面多余的0），若匹配成功，则提取出该对话中的"human"问题和"gpt"回复，存入以下示例格式
[
      {
        "Question": "<image>\nWhat precautions is the woman taking while walking in this weather?",
 "Answer": "The woman is taking precautions while walking in the rainy weather by using an open umbrella to shield herself from the rain. The umbrella helps to keep her dry and protected from getting soaked by the rain, ensuring her comfort and safety while walking down the wet street. Holding the umbrella also indirectly indicates that the woman might be more attentive to her surroundings and cautious of potential hazards caused by the wet conditions, such as slippery surfaces, puddles, or splashing from passing vehicles."
      },
       ]
并且把该条对话保存在/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/raw_data/complex_reasoning/  文件夹下，文件名为：{NSD标号-1}.json

（3)若匹配失败，则记录下当前的{NSD标号-1}，并把它存入一个list，等程序执行完毕后，保存在/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/raw_data/complex_reasoning/  文件夹下，存为npy文件格式，并且print出来内容

（4）提示：你需要特别仔细地处理序号配对的问题，比如每个被试的NSD标号数据（形状为10000）和COCOID标号数据（形状为10000）是一一对应的，再比如我在实际保存数据的时候要用{NSD标号-1}。亦或者每个被试的NSD标号数据的前1000个数据是完全相同的，清楚这一点你可以节省一些匹配次数

（5）为了确保你理解了全流程，请你先理解并复述这个代码写作思路，然后再开始写代码

（5）


"""