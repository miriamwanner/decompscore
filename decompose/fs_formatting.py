
import argparse
import string
import json

def annotate_fs(annotated_newdict, decomp_type, original_factscore):
    out = []
    for fs, new in zip(original_factscore, annotated_newdict):
        assert fs["topic"] == new["topic"], "Topics do not match"
        annotations = []
        for sent_dict in new["decomposition"]:
            sentence = sent_dict["sentence"]
            af = sent_dict[decomp_type]
            af_dict = [{"text": f} for f in af]
            annotations.append({"text": sentence, "model-atomic-facts": af_dict})
        fs["annotations"] = annotations
        out.append(fs)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotated_dict',
                        type=str,
                        default=None)
    parser.add_argument('--factscore_dict',
                        type=str,
                        default=None)
    parser.add_argument('--decomp_type',
                        type=str,
                        default="factscore",
                        choices=["factscore", "conllu", "factscore2", "chen23", "wice", "decompose"])
    args = parser.parse_args()
    
    annotated_nd = []
    with open(args.annotated_dict) as f:
        for line in f:
            dp = json.loads(line)
            annotated_nd.append(dp)
    
    original_fs = []
    with open(args.factscore_dict) as g:
        for line in g:
            dp = json.loads(line)
            original_fs.append(dp)
    
    if args.decomp_type == "factscore":
        dt = "factscore prompt"
    elif args.decomp_type == "factscore2":
        dt = "factscore prompt"
    elif args.decomp_type == "conllu":
        dt = "conllu prompt"
    elif args.decomp_type == "chen23":
        dt = "factscore prompt"
    elif args.decomp_type == "wice":
        dt = "factscore prompt"
    elif args.decomp_type == "decompose":
        dt = "factscore prompt"
    
    new_fs_format = annotate_fs(annotated_nd, dt, original_fs)
    if args.decomp_type == "factscore":
        print("Writing to "+str(args.factscore_dict.replace(".jsonl", f"_formatted_fs.jsonl")))
        with open(args.factscore_dict.replace(".jsonl", f"_formatted_fs.jsonl"), 'w') as f:
            for line in new_fs_format:
                json_record = json.dumps(line)
                f.write(json_record+'\n')
    if args.decomp_type == "conllu":
        print("Writing to "+str(args.factscore_dict.replace(".jsonl", f"_formatted_fs_conllu.jsonl")))
        with open(args.factscore_dict.replace(".jsonl", f"_formatted_fs_conllu.jsonl"), 'w') as f:
            for line in new_fs_format:
                json_record = json.dumps(line)
                f.write(json_record+'\n')
    if args.decomp_type == "factscore2":
        print("Writing to "+str(args.factscore_dict.replace(".jsonl", f"_formatted_fs_2examples.jsonl")))
        with open(args.factscore_dict.replace(".jsonl", f"_formatted_fs_2examples.jsonl"), 'w') as f:
            for line in new_fs_format:
                json_record = json.dumps(line)
                f.write(json_record+'\n')
    if args.decomp_type == "chen23":
        print("Writing to "+str(args.factscore_dict.replace(".jsonl", f"_formatted_fs_chen23.jsonl")))
        with open(args.factscore_dict.replace(".jsonl", f"_formatted_fs_chen23.jsonl"), 'w') as f:
            for line in new_fs_format:
                json_record = json.dumps(line)
                f.write(json_record+'\n')
    if args.decomp_type == "wice":
        print("Writing to "+str(args.factscore_dict.replace(".jsonl", f"_formatted_fs_wice.jsonl")))
        with open(args.factscore_dict.replace(".jsonl", f"_formatted_fs_wice.jsonl"), 'w') as f:
            for line in new_fs_format:
                json_record = json.dumps(line)
                f.write(json_record+'\n')
    if args.decomp_type == "decompose":
        print("Writing to "+str(args.factscore_dict.replace(".jsonl", f"_formatted_decompose.jsonl")))
        with open(args.factscore_dict.replace(".jsonl", f"_formatted_decompose.jsonl"), 'w') as f:
            for line in new_fs_format:
                json_record = json.dumps(line)
                f.write(json_record+'\n')