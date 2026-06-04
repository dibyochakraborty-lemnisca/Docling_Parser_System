import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

start = text.find(r'\subsection{Phase 1: Architectural Audit}')
end = text.find(r'\subsection{Phase 2: Empirical Evaluation')
print(text[start:end])
