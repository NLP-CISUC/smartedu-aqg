from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("voidful/bart-distractor-generation-both")
model = AutoModelForSeq2SeqLM.from_pretrained("voidful/bart-distractor-generation-both")
maxLength = 1024

def getDistractors(context, question, answer, nDistractors):
    QAinput = ' </s> ' + question + ' </s> ' + answer
    length = maxLength - len(QAinput)
    input = context[:length] + QAinput
    output = tokenizer.batch_decode(model.generate(tokenizer.encode(input,return_tensors='pt'),num_return_sequences=nDistractors,num_beams=nDistractors))
    
    distractors = []
    for item in output:
        distractor = item.replace("</s>", "").replace("<pad>", "")
        distractors.append(distractor)

    return distractors