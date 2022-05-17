
def check_predict_file(predict_file):
    try:
        content = [len(line.split("\t")) == 2 for line in open(predict_file, "r").readlines() if line.strip() != ""]
        assert sum(content) == len(content)
    except:
        raise ValueError(f"Please make sure that {predict_file} consists of lines of 'token\tlabel'."
                         "You can use utils/preprocess.py to convert the .cupt files to the desired format.")


def write_predictions(predictions, content_to_be_predicted, output_file):
    with open(output_file, "w") as writer:
        with open(content_to_be_predicted, "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = line.split("\t")[0] + " " + predictions[example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split("\t")[0])
