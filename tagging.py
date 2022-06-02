from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

text = "The ship on which Alice boarded was wrecked in the storm, unable to overcome the heavy waves. \"Gwank! I'm sorry to disturb your day so early.\" Captain Jin commanded as he looked at the small group of people sitting in the back. \"Princess, this is none of your business. We are here to help you.\" Grandpa Hernandez greeted. \"We've been hired by a private company to Avalon Inc. We'll handle all the duties for the next few weeks. We'll continue to keep the Mor Co. honest and get the best out of this operation. It won't be long before your family and friends are all gone.\" Grandpa Hernandez said with a grin before handing the keys over to the private investigators working for the agency. They proceeded to hand over the keys to the owner of the Lim family business to the head of the investigation unit. After handing over the keys to the head of the investigation unit, Captain Jin commanded the girls to follow the Princess through all of the streets of Mor Co. Once inside Mor Co., all the investigators proceeded to the small bar that was being used by the company to pay their employees. "

def get_name(text):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER") # bert-large-NER # bert-base-NER
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    example = text
    ner_results = nlp(example)
    per = []
    # print(ner_results)
    for ner in ner_results:
        if ner['entity'][-3:] =='PER' and ner['score'] > 0.7:
            per.append(ner)
    return per

per = get_name(text)

print(text)
print(per)

w = text

for p in per:
    print(p['word'], end=" / ")
pe = input("\nChoose name : ")
name = input("input name : ")
print(pe, '->', name)

for x in range(len(per)):
    i = len(per)-1-x
    if per[i]['word'] == pe:
        s = per[i]['start']
        e = per[i]['end']
        w = w[:s] + name + w[e:]
print(w)