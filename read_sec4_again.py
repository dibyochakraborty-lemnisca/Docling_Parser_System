with open('paper/main.tex', 'r') as f:
    text = f.read()

start = text.find(r'\subsection{Phase 2: Process-Trace Scoring}')
end = text.find(r'\subsection{Phase 3: Planted-Fault Recognition}')
print(text[start:end])
