import json
import os
import stat

with open('data/train_10h.json', 'r') as f:
     data_train = json.load(f)

unique_characters = []
for _, y in data_train.items():
    sentence = y["word"].replace(" ", "")
    characters = list(set(sentence))
    for c in characters:
        if c not in unique_characters:
            unique_characters.append(c)

unique_characters.sort()

unique_characters.insert(0, "<blank>")
unique_characters.insert(1, "<space>")
unique_characters.insert(2, "<unk>")

print(unique_characters) 

with open("data/vocab.txt",'w') as fp:
    fp.write('\n'.join(unique_characters))

st = os.stat("data/vocab.txt")
os.chmod("data/vocab.txt", st.st_mode | stat.S_IEXEC)

