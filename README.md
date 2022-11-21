# novel_generator
create novel with text generation model

# code
- data_preprocessing.py
- gpt2_finetune.py : data_processing 후 gpt2 모델 학습
- gpt2_generate.py : 학습 시킨 모델을 경로에 맞게 저장 후 text 생성
- gpt2_pretrained.py : 아직 finetune하지 않은 pretrained gpt2 모델
- tagging.py : NER tagging 모델
