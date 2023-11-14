import librosa
import os
from tqdm import tqdm
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
import commons
import torch
from transformers import AutoTokenizer
from text.japanese_bert import get_bert_feature
from infer import latest_version
from mel_processing import spectrogram_torch
import h5py
import json
from multiprocessing import Pool
from functools import partial
from config import config

"""
将所有数据预处理步骤全部集合在这 (方便一路单步调试到底(bushi))
数据集打包为hdf5(一个h5里包含一条语音训练用的的所有数据),
训练则直接加载，仅进行张量计算
训练配置等不再牵扯上filelist,clean filelist,bert,spec等散装操作，直接指定一个文件夹即可加载数据训练

注意: 
1. 训练部分还未适配，仅修改了一份单卡用来调试用。(理论替换掉data loader即可)
2. 此行为会导致所需储存空间为原来的1.2倍左右
3. 此预处理暂时只对日语支持

filelist格式:
{wav_path}|{speaker_name}|{language}|{text} 其中wav_path为相对audio_root的路径
示例:

+-- audio_root
  +-- folder1
    |-- file1.ogg

folder1/file1.ogg|{speaker_name}|{language}|{text}
"""

train_file_list = config.data_pack_config.train_filelist  # 训练集文件
train_audio_root = config.data_pack_config.train_audios  # 训练集音频目
train_output_root = config.data_pack_config.train_datas  # 预处理训练集数据输出目录

eval_file_list = config.data_pack_config.eval_filelist  # 验证集文件
eval_audio_root = config.data_pack_config.eval_audios  # 验证集音频目录
eval_output_root = config.data_pack_config.eval_datas  # 预处理验证集数据输出目录

config_in = config.data_pack_config.config_in  # config.json的模版文件
config_out = config.data_pack_config.config_out  # config.json预处理输出

add_blank = config.data_pack_config.add_blank
num_processes = config.data_pack_config.num_process


def filter_data_and_build_config(file_list_path: str, audio_root: str):
    # 数据检查1：音频文本一对一处理，跳过音频缺失，如音频映射多条文本则取第一条，跳过后面
    data_list = open(file_list_path, "r", encoding="utf-8").readlines()
    filtered_list = []
    added_audio = []
    # 角色名-id 字典
    speaker_id_map = {}
    current_sid = 0

    for line in data_list:
        voice = line.split("|")[0]
        speaker = line.split("|")[1]
        # 筛选可用数据同时构建角色-id字典
        if os.path.exists(os.path.join(audio_root, voice)) and (voice not in added_audio):
            filtered_list.append(line)
            added_audio.append(voice)
            if speaker not in speaker_id_map.keys():
                speaker_id_map[speaker] = current_sid
                current_sid += 1
    print(f"Skiped {len(data_list) - len(filtered_list)} items.")
    json_config = json.load(open(config_in, encoding="utf-8"))
    json_config["data"]["spk2id"] = speaker_id_map
    json_config["data"]["train_root"] = train_output_root
    json_config["data"]["eval_root"] = eval_output_root
    json_config["version"] = latest_version

    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)
    return filtered_list, speaker_id_map


def pack_item(line, speaker_id_map, audio_root, output_root):
    audio_path, speaker, language, text = tuple(line.split("|"))
    # 加载音频并处理为张量，计算频谱
    audio_res, sr = librosa.load(os.path.join(audio_root, audio_path), sr=44100, res_type="soxr_vhq")
    # audio,spec Tensor
    # clamp 处理音频超过范围的值（暂不确定是否有影响）
    audio_norm = torch.FloatTensor(audio_res).unsqueeze(0).clamp(-1, 1)
    spec = spectrogram_torch(audio_norm, 2048, 44100, 512, 2048, center=False, ).squeeze(0)

    # 预处理文本，获取文本bert结果
    norm_text, phones, tones, word2ph = clean_text(text, language)
    phone_ids, tone_ids, language_ids = cleaned_text_to_sequence(phones, tones, language)
    language_id = torch.LongTensor([language_ids[0]])
    if add_blank:
        phone_ids = commons.intersperse(phone_ids, 0)
        tone_ids = commons.intersperse(tone_ids, 0)
        language_ids = commons.intersperse(language_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    # phone,tone,language Tensor
    phone = torch.LongTensor(phone_ids)
    tone = torch.LongTensor(tone_ids)
    language_tensor = torch.LongTensor(language_ids)
    # bert Tensor
    bert = get_bert_feature(norm_text, word2ph)
    # sid Tensor
    sid = torch.LongTensor([int(speaker_id_map[speaker])])

    if (int(phone.shape[0]) == int(tone.shape[0]) == int(language_tensor.shape[0]) == int(bert.shape[1])):
        h5_path = os.path.join(output_root, audio_path + ".h5")
        h5_folder = os.path.split(h5_path)[0]
        if not os.path.exists(h5_folder):
            os.makedirs(h5_folder)
        with h5py.File(h5_path, mode="w") as h5:
            h5["phones"] = phone
            h5["spec"] = spec
            h5["wav"] = audio_norm
            h5["sid"] = sid
            h5["tone"] = tone
            h5["language"] = language_tensor
            h5["bert"] = bert
            h5["language_id"] = language_id


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    # 多进程处理训练集
    filtered_list, speaker_id_map = filter_data_and_build_config(train_file_list, train_audio_root)
    partial_pack_item = partial(pack_item, speaker_id_map=speaker_id_map, audio_root=train_audio_root, output_root=train_output_root, )
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(partial_pack_item, filtered_list), total=len(filtered_list), ):
            pass
    # 处理验证集 注：speaker_id_map需使用训练集的，即与配置文件一致
    eval_filtered_list, _ = filter_data_and_build_config(eval_file_list, eval_audio_root)
    for line in eval_filtered_list:
        pack_item(line, speaker_id_map, eval_audio_root, eval_output_root)
