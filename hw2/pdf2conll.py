from PyPDF2 import PdfReader
import os
import spacy
import argparse

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    tokenizer = nlp.tokenizer

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('pdf_pth', type=str)
    parser.add_argument('sav_pth', type=str)
    parser.add_argument('sen_sav_pth', type=str)

    args = parser.parse_args()
    
    pdfdata_pth = args.pdf_pth#'./new_pdf/'
    sav_pth = args.sav_pth#'./plaintxt/'
    sen_sav_pth = args.sen_sav_pth#'./sentxt/'
    os.makedirs(pdfdata_pth, exist_ok=True)
    os.makedirs(sav_pth, exist_ok=True)
    os.makedirs(sen_sav_pth, exist_ok=True)
    pdf_lst = os.listdir(pdfdata_pth)

    for name in pdf_lst:
        pdf_pth = pdfdata_pth+name
        reader = PdfReader(pdf_pth)
        txt = ''
        page = reader.pages
        for i in range(len(page)):
            txt += page[i].extract_text()
            
        tokenized_txt = ''
        sentence_txt = ''
        sentences = [i for i in nlp(txt).sents]
        for sen in sentences:
            sen = str(sen).replace('-\n','').replace('\n',' ')
            tokens = tokenizer(sen)
            for t in tokens:
                t = str(t)
                # if t=='\n': continue
                tokenized_txt += '{} O\n'.format(t)

            tokenized_txt += '\n'
            sentence_txt += sen+'\n'

        with open(sav_pth+name.replace('pdf','txt'), 'w') as f:
            f.write(tokenized_txt)

        with open(sen_sav_pth+name.replace('pdf','txt'), 'w') as f:
            f.write(sentence_txt)

        