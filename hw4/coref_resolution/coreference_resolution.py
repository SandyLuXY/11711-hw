import spacy
import neuralcoref
import json

class CoreferenceResolution:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(self.nlp)
    
    def get_coreferenced_text(self,text):
        doc = self.nlp(text)
        return doc._.coref_resolved


def process_train_data(train_df):
    cf = CoreferenceResolution()
    text_dict = {}
    for (i, row) in train_df.iterrows():
        # print(i,type(i))
        coref_text = cf.get_coreferenced_text(row['text'])
        text_dict[i] = coref_text
        if i % 500 == 0:
            print(f"{i}/{len(train_df)}")
            
    with open('data/coreferenced_data.json','w') as f:
        f.write(json.dumps(text_dict, indent=4))

def run_coreference_resolution():
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    doc1 = nlp('My sister has a dog. She loves him.')
    # print(doc1._.coref_clusters)
    rst = doc1._.coref_resolved
    print(rst, type(rst))

    doc2 = nlp('Angela lives in Boston. She is quite happy in that city.')
    for ent in doc2.ents:
        print(ent._.coref_cluster)



if __name__ == '__main__':
    # process train data
    train_df = pd.read_csv("data/train.tsv", sep="\t", header=None, names=["text", "labels", "id"])
    process_train_data(train_df)
