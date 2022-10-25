import torch
import argparse
from datasets import load_dataset
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer


def generate_codet5(args):
    device = f"cuda:{args.gpuid}"
    dataset = load_dataset("neulab/conala")

    model_name = 'Salesforce/codet5-small'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    if args.model_pt is not None:
        model.load_state_dict(torch.load(args.model_pt, map_location=f'cuda:{args.gpuid}'))
    model.eval()
    count = 1
    bsz = 1
    with open(args.tgt_file, 'w') as fout:
        conala_data = dataset[args.split]
        source = []
        for data in conala_data:
            if data['rewritten_intent'] is not None:
                source.append(data['rewritten_intent'])
            else:
                source.append(data['intent'])
        slines = [source[0].strip().lower()]
        source = source[1:]
        for sline in source:
            if count % 100 == 0:
                print(count, flush=True)
            if count % bsz == 0:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, return_tensors="pt", padding='max_length', max_length=1024, truncation=True)
                    summaries = model.generate(
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
                        num_return_sequences=128, num_beam_groups=16, diversity_penalty=0.1, num_beams=128,
                        length_penalty=0.6,
                        max_length=140,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                dec = [dec[i] for i in range(len(dec)) if i % 8 == 0]
                for hypothesis in dec:
                    hypothesis = hypothesis.replace("\n", " ")
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []
            sline = sline.strip().lower()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1
        if slines != []:
            with torch.no_grad():
                dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                summaries = model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    num_return_sequences=128, num_beam_groups=16, diversity_penalty=0.1, num_beams=128,
                    length_penalty=0.6,
                    max_length=140,
                )
                dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            dec = [dec[i] for i in range(len(dec)) if i % 8 == 0]
            for hypothesis in dec:
                hypothesis = hypothesis.replace("\n", " ")
                fout.write(hypothesis + '\n')
                fout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--gpuid", type=int, default=0, help="gpu ids")
    parser.add_argument("--split", type=str, default=None, help="train or test")
    parser.add_argument("--tgt_file", type=str, help="target file")
    parser.add_argument("--model_pt", type=str, default=None, help="model checkpoint file path")
    args = parser.parse_args()
    generate_codet5(args)