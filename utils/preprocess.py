import os, sys
from collections import defaultdict
import numpy as np
from transformers import BertTokenizer
import logging



def process_label(label):
    if label == "*":
        return "*"
    if ";" in label:
        label = label.split(";")[0]
    label = label.split(":")
    return label


if __name__ == "__main__":

    data_path = sys.argv[1] #"sharedtask-data/1.2"
    out_dir = sys.argv[2]  #"processed_data"
    try:
        target_lang = sys.argv[3]
    except:
        target_lang = None

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    logging.basicConfig(filename=f'{out_dir}/sentence-sizes.log', filemode='w',  level=logging.INFO, format='%(message)s')

    langs = [lang for lang in os.listdir(data_path) if len(lang) == 2 and lang != "HE"]

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    for lang in sorted(langs):
        logging.info(f"*****{lang}******")
        if target_lang and target_lang not in lang: continue
        for ds in ["train", "dev"]:
            data = open(os.path.join(data_path, lang, "{}.cupt".format(ds))).read().split("\n\n")
            out_path = os.path.join(out_dir)
            if not os.path.exists(out_path): os.makedirs(out_path)
            f_out = open(os.path.join(out_path, "{}_{}.csv".format(ds, lang.lower())), "w", encoding="utf8")
            label_set = set()
            sen_len = []
            bert_sen_len = []
            for sent in data:
                label_dict = defaultdict()
                if sent == "": continue
                sentence = " ".join([line.split("\t")[1] for line in sent.split("\n") if not line.startswith("#")])
                sen_len.append(len(sent.split("\n")))
                bert_size = len(tokenizer.tokenize(sentence))
                #if bert_size > 250:
                #    print(bert_size)
                #    print(" ".join([line.split("\t")[1] for line in sent.split("\n") if not line.startswith("#")]))
                bert_sen_len.append(bert_size)
                for line in sent.split("\n"):
                    if line.startswith("#"): continue
                    word, label = line.split("\t")[1], process_label(line.split("\t")[10])
                    word = word.replace(" ", "")
                    if "EL" in lang:
                        word = word.lower()
                    if label[0] == "*":
                        f_out.write("{}\t{}\n".format(word, "O"))
                        continue
                    if len(label) > 1:
                        label_dict[label[0]] = label[1]
                        label = label[1] + "-B"
                    else:
                        try:
                            label = label_dict[label[0]] + "-I"
                        except:
                            label = "O"
                    f_out.write("{}\t{}\n".format(word, label))
                f_out.write("\n")

            f_out.close()
            logging.info("{} data: {} words".format(ds.upper(), len(data)))
            logging.info("Longest sent (words): {} , longest sent. (tokens): {} , # of sent > 250 tokens: {}, "
                         "shortest sent (tokens): {}, avg. sentence (tokens): {:.3f}".format(np.max(sen_len), np.max(bert_sen_len),
                                                                 len((np.where(np.array(bert_sen_len) > 250)[0])),
                                                                 np.min(bert_sen_len),
                                                                 np.average(bert_sen_len)))
        logging.info(f"***********")

        print(f"{lang} is processed")
