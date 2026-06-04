import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

start = text.find(r'\section{Evaluation methodology}')
end = text.find(r'\section{Discussion}')
content = text[start:end]

print("=== EVALUATION SECTION ===")
print(content[:1500])
print("...")
