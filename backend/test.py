import fitz

pdf = fitz.open("data/schemes/PM_Kisan_Scheme.pdf")

for page in pdf:
    text = page.get_text()
    print(len(text))