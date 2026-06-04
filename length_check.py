import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

# Estimate content length
content_start = text.find(r'\section{Introduction}')
content_end = text.find(r'\bibliographystyle{plainnat}')

content = text[content_start:content_end]
words = len(re.findall(r'\w+', content))
print(f"Content word count: {words}")

# A typical 8-page NeurIPS/CAISc paper (with figures) can hold about 3,500 - 4,500 words.
# Let's see how much we need to cut.
