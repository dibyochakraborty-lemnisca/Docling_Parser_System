import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

def print_section_length(start_marker, end_marker, name):
    start = text.find(start_marker)
    end = text.find(end_marker) if end_marker else text.find(r'\bibliographystyle{plainnat}')
    if start != -1 and end != -1:
        words = len(re.findall(r'\w+', text[start:end]))
        print(f"{name}: {words} words")

print_section_length(r'\section{Introduction}', r'\section{Related work}', '1. Intro')
print_section_length(r'\section{Related work}', r'\section{System architecture', '2. Related Work')
print_section_length(r'\section{System architecture', r'\section{Evaluation methodology}', '3. Architecture')
print_section_length(r'\section{Evaluation methodology}', r'\section{Discussion}', '4. Evaluation')
print_section_length(r'\section{Discussion}', r'\section{Conclusion}', '5. Discussion')
print_section_length(r'\section{Conclusion}', r'\bibliographystyle{plainnat}', '6. Conclusion')

