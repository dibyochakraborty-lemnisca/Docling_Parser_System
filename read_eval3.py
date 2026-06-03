import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

start = text.find(r'\subsection{Phase 2: Empirical Evaluation')
end = text.find(r'\section{Discussion}')
print(text[start:end])
