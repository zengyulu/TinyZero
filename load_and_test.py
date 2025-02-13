from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 指定本地模型路径（替换为你的实际路径）
model_path = "/home/zengyu/Workspace/2_models/countdown_trained"
#model_path = "/home/zengyu/Workspace/2_models/Qwen2/Qwen2.5-3B"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",  # 自动分配设备（GPU/CPU）
    torch_dtype='auto' #torch.float16  # 自动处理数据类型
)

# 验证设备分配
#print(f"Model devices: {model.hf_device_map}")

# 设置模型为评估模式
model.eval()

# 推理函数
def generate_text(prompt, max_length=1024):
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成参数配置
    generate_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_length,
        "do_sample": True,  # 启用随机采样
        "temperature": 0.8,  # 控制随机性（0.1-1.0）
        "top_p": 0.9,        # 核采样参数
        "pad_token_id": tokenizer.eos_token_id  # 设置结束符为填充符
    }
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 使用示例
if __name__ == "__main__":
    prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers [73, 76, 6, 70], create an equation that equals 38. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>"
    print("输入提示:", prompt)
    print("生成结果:", generate_text(prompt))
