from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import docx
from win32com import client as wc


def pdf2txt(pdf_file, txt_file):
    maxpages = 0
    password = ''
    pagenos = set()
    caching = True
    laparams = LAParams()
    rsrcmgr = PDFResourceManager(caching=caching)
    txt_file = open(txt_file, 'wt', encoding='utf-8', errors='ignore')

    converter = TextConverter(rsrcmgr, txt_file, laparams=laparams)

    with open(pdf_file, 'rb') as fp:
        process_pdf(rsrcmgr, converter, fp, pagenos, maxpages=maxpages, password=password,
                    caching=caching, check_extractable=True)
    converter.close()
    txt_file.close()


def docx2txt(docx_file, txt_file):
    docx_file = docx.Document(docx_file)
    with open(txt_file, 'wt') as fp:
        for para in docx_file.paragraphs:
            fp.write(para.text + '\n')


def doc2txt(doc_file, txt_file):
    word_app = wc.Dispatch('Word.Application')
    doc_file = word_app.Documents.Open(doc_file)
    temp_file = 'g:\\temp\\temp.docx'
    doc_file.SaveAs(temp_file, 12, False, "", True, "", False, False, False, False)
    doc_file.Close()
    word_app.Quit()
    docx2txt(temp_file, txt_file)



if __name__ == '__main__':
    # pdf_file = 'g:\\temp\\test.pdf'
    # txt_file = 'g:\\temp\\test.txt'
    # pdf2txt(pdf_file, txt_file)

    # doc_file1 = 'g:\\temp\\test1.docx'
    # txt_file1 = 'g:\\temp\\test1.txt'
    # docx2txt(doc_file1, txt_file1)

    doc_file2 = 'g:\\temp\\test2.doc'
    txt_file2 = 'g:\\temp\\test2.txt'
    doc2txt(doc_file2, txt_file2)