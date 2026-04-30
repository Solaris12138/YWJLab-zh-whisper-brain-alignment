### 使用 Montreal Forced Aligner 进行文本音频对齐

Montreal Forced Aligner（简称 MFA）是一种文本与音频的对齐工具，如果在你的研究中需要获取每个字/音素准确的时间信息，那么 MFA 是一个很好的工具，MFA 的误差大概在 20 ms 左右，因此哪怕对于电生理数据的对齐也足够了。

为了实现对齐，你首先需要准备以下内容：
1. 完整的刺激录音（要求是 `.wav` 格式，采样率至少大于 16000 Hz）
2. 一个 `.txt` 文件，里面包含了录音内容的转写。

如果你未获取到录音内容的转写（例如对话任务），可以使用 faster-whisper 进行转录。在使用 MFA 之前，请确保你的 `Python` 环境中有 MFA 工具链，代码如下：

```bash
pip install montreal-forced-aligner -i https://pypi.tuna.tsinghua.edu.cn/simple # MFA 的主要框架
pip install textgrid -i https://pypi.tuna.tsinghua.edu.cn/simple # 用于读取 MFA 处理后的结果
pip install spacy-pkuseg dragonmapper hanziconv -i https://pypi.tuna.tsinghua.edu.cn/simple # 中文分词用工具
```

在安装好相应工具链之后，在命令行内安装字典和语音模型，字典用于告诉 MFA 每个单词应该如何被发音，而语音模型用于识别给定的音频片段最像哪个音素。安装字典和语音模型的代码如下：

```bash
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_china_mfa
```

函数 `mfa_tools.py` 中封装好了三个函数分别为 `mfa_align_word`、`mfa_align_char` 和 `mfa_align_ipa`，默认情况下只需要指定 `wav_path` 和 `txt_path` 两个参数即可，MFA 可能会要求两个文件的名字相同。如果你需要进行其他语言的文本转写，请在 [MFA 官方文档](https://mfa-models.readthedocs.io/en/latest/dictionary/index.html) 内查询相应的字典文件和语音模型。