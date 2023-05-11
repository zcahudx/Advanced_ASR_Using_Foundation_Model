from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from decoder import decode
from utils import concat_inputs

from dataloader import get_dataloader, get_dataloader_wav


def train(model, args):
    torch.manual_seed(args.seed)
    if args.use_fbank:
        train_loader = get_dataloader(args.train_json, args.vocab, args.batch_size, True)
        val_loader = get_dataloader(args.val_json, args.vocab, args.batch_size, False)
    else:
        train_loader = get_dataloader_wav(args.train_json, args.vocab, args.batch_size, True)
        val_loader = get_dataloader_wav(args.val_json, args.vocab, args.batch_size, False)

    criterion = CTCLoss(zero_infinity=True)
    # optimiser = SGD(model.parameters(), lr=args.lr)
    optimiser = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.0, 
                                                  total_iters=int(args.num_epochs*len(train_loader)*0.5), last_epoch=- 1)
    lr_track = []
    def train_one_epoch(epoch):
        running_loss = 0.0
        last_loss = 0.0
        
        for idx, data in enumerate(train_loader):
            step = int(epoch*len(train_loader)+idx)
            inputs, in_lens, trans, durations = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            if args.use_fbank:
                inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            else:
                inputs = inputs.transpose(0, 1)
                in_lens = None

            # targets = [
            #     torch.tensor(
            #         list(map(lambda x: args.vocab[x], target.split())), dtype=torch.long
            #     )
            #     for target in trans
            # ]
            targets = trans
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long
            )
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)

            optimiser.zero_grad()
            outputs = log_softmax(model(inputs), dim=-1)
            if in_lens is None:
                in_lens = torch.tensor(
                    [
                        outputs.shape[1] * duration / max(durations)
                        for duration in durations
                    ],
                    dtype=torch.long,
                )
                outputs = outputs.transpose(0, 1)
            loss = criterion(outputs, targets, in_lens, out_lens)
            loss.backward()
            
            # warm up
            if step <= (int(args.num_epochs*len(train_loader)*0.1)-1):
                 for param_group in optimiser.param_groups:
                     param_group['lr'] = ((args.lr)/(int(args.num_epochs*len(train_loader)*0.1)-1))*step
            
            lr_track.append(optimiser.param_groups[0]["lr"])  
                   
            optimiser.step()
            
            # scheduler
            if step >=(int(args.num_epochs*len(train_loader)*0.5)-1):
                scheduler.step()

            running_loss += loss.item()
            if idx % args.report_interval + 1 == args.report_interval:
                last_loss = running_loss / args.report_interval
                print("  batch {} loss: {}".format(idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + idx + 1
                running_loss = 0.0
        return last_loss
    
    # Freeze model parameters except the output layer for the first 3 epoches
    for name ,param in model.named_parameters():
        if "model" in name:
            param.requires_grad = False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("checkpoints/{}".format(timestamp)).mkdir(parents=True, exist_ok=True)
    best_val_loss = 1e6
    best_val_wer =1e8
    for epoch in range(args.num_epochs):
        
        # unfreeze after the third epoch
        if epoch == 4:
            # str = ["encoder.layers.0.", "encoder.layers.1.", "encoder.layers.2.", "encoder.layers.3.",
            #        "encoder.layers.4.", "encoder.layers.5.", "encoder.layers.6.", "encoder.layers.7.",
            #        "encoder.layers.8."]
            for name ,param in model.named_parameters():
                if "feature" not in name: # and not any([x in name for x in str]):
                    param.requires_grad = True
        
        print("EPOCH {}:".format(epoch + 1))
        model.train(True)
        avg_train_loss = train_one_epoch(epoch)

        model.train(False)
        running_val_loss = 0.0
        for idx, data in enumerate(val_loader):
            inputs, in_lens, trans, durations = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            if args.use_fbank:
                inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            else:
                inputs = inputs.transpose(0, 1)
                in_lens = None
            # targets = [
            #     torch.tensor(
            #         list(map(lambda x: args.vocab[x], target.split())), dtype=torch.long
            #     )
            #     for target in trans
            # ]
            targets = trans
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long
            )
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)
            with torch.no_grad():
                outputs = log_softmax(model(inputs), dim=-1)
            if in_lens is None:
                in_lens = torch.tensor(
                    [
                        outputs.shape[1] * duration / max(durations)
                        for duration in durations
                    ],
                    dtype=torch.long,
                )
                outputs = outputs.transpose(0, 1)
            val_loss = criterion(outputs, targets, in_lens, out_lens)
            running_val_loss += val_loss
        avg_val_loss = running_val_loss / len(val_loader)
        val_decode = decode(model, args, args.val_json)
        print(
            "LOSS train {:.5f} valid {:.5f}, valid WER {:.2f}%".format(
                avg_train_loss, avg_val_loss, val_decode[4]
            )
        )

        if avg_val_loss < best_val_loss:
        # if val_decode[4] < best_val_wer:
            best_val_loss = avg_val_loss
            # best_val_wer = val_decode[4]
            model_path = "checkpoints/{}/model_{}".format(timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
    
    # plt.plot(lr_track)
    # plt.xlabel("steps")
    # plt.ylabel("lr")
    # plt.savefig("./lr_warmup.png")
    # plt.close()
    return model_path
