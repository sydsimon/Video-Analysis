# 定义起始和结束的score数字
start_score = -5
end_score = 5

# 定义要写入的元素
element = 'pbblt'

# 生成文本
result = ''
for i in range(start_score, end_score + 1):
    result += f"\"score_{i}\",\"{element}\","

print(result)