import json

data = []
# with open('./dataset/novel_chap30.csv', 'r') as f:
#     lines = f.readlines()
with open('./dataset/100novel_chap20.csv', 'r') as f:
    lines = f.readlines()

id = 0
para = ''
for i in lines:
    if i != '\n' and len(i)>20:
        i = i[:-1]+' '
        para += i
        if len(para) > 1000:
            temp = {'id':id,'text':para}
            data.append(temp)
            para = ''
            id += 1
print(data)
print("num of sequence of sentences:",len(data)) # /100novel_chap20.json : 9861    /dataset/10novel_chap3.json : 229
 
# with open('./dataset/novel_chap30.json', 'w') as w:
#     json.dump(data, w, indent=4)
with open('./dataset/100novel_chap20.json', 'w') as w:
    json.dump(data, w, indent=4)
