import librosa
import os
from tqdm import tqdm
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
import commons
import torch
from transformers import AutoTokenizer
from text.japanese_bert import get_bert_feature
from mel_processing import spectrogram_torch
import h5py
import json

"""
将所有数据预处理步骤全部集合在这 (方便一路单步调试到底(bushi))
数据集打包为hdf5(一个h5里包含一条语音训练用的的所有数据),
训练则直接加载，仅进行张量计算
训练配置等不再牵扯上filelist,clean filelist,bert,spec等散装操作，直接指定一个文件夹即可加载数据训练

注意: 
1. 训练部分还未适配，仅修改了一份单卡用来调试用。(理论替换掉data loader即可)
2. 此行为会导致所需储存空间为原来的1.2倍左右
"""
# TODO: 完善配置写入，以及训练那边读取。考虑一次处理多文件夹，考虑除bert外处理使用多进程（个人感觉不太必要）
tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")

audio_folder = 'test_data/audio'
file_list = 'test_data/debug.csv'
output_folder = 'test_data/process'
config_path = 'configs/config.json'

add_blank = True

# 数据检查1：音频文本一对一处理，跳过音频缺失，如音频映射多条文本则取第一条，跳过后面
# 匹配规则：由file list中音频文件名匹配audio folder中文件(此处音频文件名使用未处理之前的音频)
data_list = open(file_list,'r',encoding='utf-8').readlines()
filtered_list = []

audio_files = list(os.listdir(audio_folder))
added_audio = []
# 角色名-id 字典
speaker_id_map = {}
current_sid = 0

for line in data_list:
    voice = line.split('|')[0]
    speaker = line.split('|')[1]
    # 筛选可用数据同时构建角色-id字典
    if (voice in audio_files) and  (voice not in added_audio):
        filtered_list.append(line)
        added_audio.append(voice)
        if speaker not in speaker_id_map.keys():
            speaker_id_map[speaker] = current_sid
            current_sid += 1
print(f'Skiped {len(data_list)-len(filtered_list)} items.')
# 保存角色表到Config json
json_config = json.load(open(config_path, encoding="utf-8"))
json_config["data"]["spk2id"] = speaker_id_map
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(json_config, f, indent=2, ensure_ascii=False)


for line in tqdm(filtered_list):
    audio_path,speaker,language,text = tuple(line.split('|'))
    # 加载音频并处理为张量，计算频谱
    audio_res, sr = librosa.load(os.path.join(audio_folder,audio_path),sr=44100,res_type='soxr_vhq')
    # audio,spec Tensor
    # clamp 处理音频超过范围的值（暂不确定是否有影响）
    audio_norm = torch.FloatTensor(audio_res).unsqueeze(0).clamp(-1,1)
    spec = spectrogram_torch(audio_norm,2048,44100,512,2048,center=False,).squeeze(0)

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

    if int(phone.shape[0]) == int(tone.shape[0]) == int(language_tensor.shape[0]) == int(bert.shape[1]):
        with h5py.File(os.path.join(output_folder,audio_path+'.h5'), mode='w') as h5:
            h5['phones'] = phone
            h5['spec'] = spec
            h5['wav'] = audio_norm
            h5['sid'] = sid
            h5['tone'] = tone
            h5['language'] = language_tensor
            h5['bert'] = bert
            h5['language_id'] = language_id