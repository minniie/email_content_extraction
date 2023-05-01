import os
import math
import random
import json

import torch
from transformers.integrations import TensorBoardCallback

from src.util.text import clean_decode, compute_f1
  

SAVE_PATH = "\mnt"


class MetricCallback(TensorBoardCallback):
    
    def on_evaluate(self, args, state, control, **kwargs):
        print(args.output_dir)
        
        # get model and dataset
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        eval_dataloader = kwargs["eval_dataloader"]
        inputs_text, preds_text, labels_text = [], [], []

        for batch in eval_dataloader:
            for input_ids, labels in zip(batch["input_ids"], batch["labels"]):
                # parse input_ids by masking out labels and padding 
                input_mask = torch.logical_and(input_ids != labels, input_ids != tokenizer.pad_token_id)
                input_ids = input_ids[input_mask].unsqueeze(0).to(model.device)
                
                # generate and decode prediction
                pred = model.generate(
                    input_ids=input_ids, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id
                ).squeeze().to("cpu")
                pred = pred[input_ids.size(-1):]
                inputs_text.append(tokenizer.batch_decode(input_ids))
                preds_text.append(clean_decode(pred, tokenizer))
                labels_text.append(clean_decode(labels, tokenizer)) 
        
        # compute ppl and f1
        loss = state.log_history[-1]["eval_loss"]
        ppl = math.exp(loss)
        f1 = compute_f1(preds_text, labels_text)

        # if training mode, report to tensorboard
        if args.do_train:
            self.tb_writer.add_scalar("eval/ppl", ppl, state.global_step)
            self.tb_writer.add_scalar("eval/f1", f1, state.global_step)
        
        # if evaluation mode, print metrics and save results
        if args.do_eval:
            print(
                f"... Metrics\n"
                f"> eval/ppl\n{ppl}\n"
                f"> eval/f1\n{f1}"
            )
            results = []
            for inp, pred, label in zip(inputs_text, preds_text, labels_text):
                results.append({
                    "input": inp,
                    "pred_output": pred,
                    "gold_output": label
                })
            with open(os.path.join(args.output_dir, "output.json"), "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)