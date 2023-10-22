# VITS2 Chinese 🎶🌟💕
## 只需上传一段语音素材，程序自动将语音切片、标注、预处理，一键训练
## 环境配置
1. 运行
```
git clone https://github.com/KevinWang676/VITS2-Chinese.git
cd VITS2-Chinese
pip install -r requirements.txt
```
2. 运行
```
cd monotonic_align
python setup.py build_ext --inplace
```
3. 上传语音文件：请上传一段**中文**、**单说话人**的语音文件，建议为长度大于10分钟的`.wav`文件
## 语音处理
4. 语音切片：在filename处填写上传的语音文件名
```
python split.py --filename {filename}.wav
```
5. 语音标注：标注完成后，可以在`filelists/short_character_anno.list`文件中对标注内容微调
```
python short_audio_transcribe.py --languages "C" --whisper_size large
```
6. 语音预处理
```
python preprocess.py
```
## 训练及推理
7. 开始训练
```
python train.py -c ./configs/config.json -m OUTPUT_MODEL
```
8. 推理

参考[inference.ipynb](https://github.com/KevinWang676/VITS2-Chinese/blob/main/inference.ipynb)
