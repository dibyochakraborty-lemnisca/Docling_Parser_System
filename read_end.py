import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

start = text.find(r'\section{Conclusion}')
end = text.find(r'\appendix')
print(text[start:end])
