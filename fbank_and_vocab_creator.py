import json
import os
import torch
import torchaudio
import stat


# Create the fbank feature
os.makedirs("/rds/user/yl879/hpc-work/MLMI2/TIMIT/data/fbank/train", exist_ok=True)
os.makedirs("/rds/user/yl879/hpc-work/MLMI2/TIMIT/data/fbank/dev", exist_ok=True)
os.makedirs("/rds/user/yl879/hpc-work/MLMI2/TIMIT/data/fbank/test", exist_ok=True)

with open('train.json', 'r') as f:
     data_train = json.load(f)
     
dict_train = {}
for x, y in data_train.items():
    waveform, sample_rate = torchaudio.load(y["wav"])
    fbank_tensor = torchaudio.compliance.kaldi.fbank(waveform)
    path = "/rds/user/yl879/hpc-work/MLMI2/TIMIT/data/fbank/train"+"/"+y["spk_id"]
    os.makedirs(path, exist_ok=True)
    torch.save(fbank_tensor, path+"/"+x[(x.find("_")+1):-4])
    st = os.stat(path+"/"+x[(x.find("_")+1):-4])
    os.chmod(path+"/"+x[(x.find("_")+1):-4], st.st_mode | stat.S_IEXEC)
    dict_train[x[:-4]] = {"fbank": path+"/"+x[(x.find("_")+1):-4],
                    "duration": y["duration"], "spk_id": y["spk_id"], "phn": y["phn"]}
    
with open('dev.json', 'r') as f:
     data_dev = json.load(f)
     
dict_dev = {}
for x, y in data_dev.items():
    waveform, sample_rate = torchaudio.load(y["wav"])
    fbank_tensor = torchaudio.compliance.kaldi.fbank(waveform)
    path = "/rds/user/yl879/hpc-work/MLMI2/TIMIT/data/fbank/dev"+"/"+y["spk_id"]
    os.makedirs(path, exist_ok=True)
    torch.save(fbank_tensor, path+"/"+x[(x.find("_")+1):-4])
    st = os.stat(path+"/"+x[(x.find("_")+1):-4])
    os.chmod(path+"/"+x[(x.find("_")+1):-4], st.st_mode | stat.S_IEXEC)
    dict_dev[x[:-4]] = {"fbank": path+"/"+x[(x.find("_")+1):-4],
                    "duration": y["duration"], "spk_id": y["spk_id"], "phn": y["phn"]}
    

with open('test.json', 'r') as f:
     data_test = json.load(f)
     
dict_test = {}
for x, y in data_test.items():
    waveform, sample_rate = torchaudio.load(y["wav"])
    fbank_tensor = torchaudio.compliance.kaldi.fbank(waveform)
    path = "/rds/user/yl879/hpc-work/MLMI2/TIMIT/data/fbank/test"+"/"+y["spk_id"]
    os.makedirs(path, exist_ok=True)
    torch.save(fbank_tensor, path+"/"+x[(x.find("_")+1):-4])
    st = os.stat(path+"/"+x[(x.find("_")+1):-4])
    os.chmod(path+"/"+x[(x.find("_")+1):-4], st.st_mode | stat.S_IEXEC)
    dict_test[x[:-4]] = {"fbank": path+"/"+x[(x.find("_")+1):-4],
                    "duration": y["duration"], "spk_id": y["spk_id"], "phn": y["phn"]}
    

with open("train_fbank.json", "w") as fp:
    json.dump(dict_train,fp, indent = 4)

st = os.stat("train_fbank.json")
os.chmod("train_fbank.json", st.st_mode | stat.S_IEXEC)
    
with open("dev_fbank.json", "w") as fp:
    json.dump(dict_dev,fp, indent = 4)

st = os.stat("dev_fbank.json")
os.chmod("dev_fbank.json", st.st_mode | stat.S_IEXEC)
    
with open("test_fbank.json", "w") as fp:
    json.dump(dict_test,fp, indent = 4)

st = os.stat("test_fbank.json")
os.chmod("test_fbank.json", st.st_mode | stat.S_IEXEC)

# Create the 39 phones
vocab_unique = []
with open("phone_map") as fp:
    for text in fp:
        start_pos=(text.strip()).find(":")+2
        phone =  text.strip()[start_pos:]
        if phone not in vocab_unique and len(phone)!=0:
            vocab_unique.append(phone)

vocab_unique.sort()

vocab_unique.insert(0,"_")
        
print(vocab_unique) 

with open("vocab.txt",'w') as fp:
    fp.write('\n'.join(vocab_unique))

st = os.stat("vocab.txt")
os.chmod("vocab.txt", st.st_mode | stat.S_IEXEC)

