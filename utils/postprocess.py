import os
from conllu import parse_incr
from collections import defaultdict
from transformers import BertTokenizer
import sys

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

if __name__ == "__main__":
    prediction_file = sys.argv[1]
    original_file = sys.argv[2]
    output_folder = sys.argv[3]
    language = prediction_file.replace(".txt", "").split("_")[-1]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    out_file = "{}/{}".format(output_folder, prediction_file.split("/")[-1].replace(".txt", ".cupt"))


    fields = ('id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc', 'PARSEME:MWE'.lower())
    data = [line for line in open(prediction_file).read().split("\n\n") if line != ""]
    f = open(out_file, "w", encoding="utf8")
    print(prediction_file)
    for i, tokenlist in enumerate(parse_incr(open(original_file), fields=fields)):
        # print(tokenlist)
        preds = [l.split()[1].replace("O", "*") for l in data[i].split("\n")]
        seen = defaultdict(int)
        try:
            assert len(preds) == len(tokenlist)
        except AssertionError:
            preds.extend(["*"] * (len(tokenlist) - len(preds)))
            print(len(preds), len(tokenlist))
            sent = " ".join([x["form"] for x in tokenlist])
            print(len(tokenizer.tokenize(sent)))

        for id, p in enumerate(preds):
            if p == "*":
                tokenlist[id]["parseme:mwe"] = "*"
            else:
                p, t = p.split("-")
                if t == "B": seen[p] += 1
                tokenlist[id]["parseme:mwe"] = str(seen[p]) + ":" + p if t == "B" else str(seen[p])

        f.writelines(tokenlist.serialize())
    print(len(list(parse_incr(open(original_file)))))
    print(len(data))
