import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Dataset footnote
content = re.sub(
    r'\\href\{https://drive\.google\.com/[^}]+\}\{IndPenSim-derived planted-fault bundle\}',
    r'IndPenSim-derived planted-fault bundle, provided in the supplementary material',
    content
)

# Traces
for i in range(1, 6):
    content = re.sub(
        r'\\href\{https://drive\.google\.com/[^}]+\}\{Trace ' + str(i) + r'\}',
        f'Trace {i} provided in supplementary material',
        content
    )

# Public repository references
content = content.replace('full trace inventory is in the public repository', 'full trace inventory is provided in the supplementary material')

content = content.replace('publicly available at \\url{https://github.com/Lemniscabio/fermdocs}', 'provided in the supplementary material')

content = content.replace('available in the public repository', 'provided in the supplementary material')

content = content.replace('in the repository and reproduced in supplementary material', 'provided in the supplementary material')

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)
