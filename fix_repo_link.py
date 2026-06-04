import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

# Replace the text reference to "public repository" with an explicit hyperlink to the Github repo
text = text.replace(
    'five representative cases, each highlighting a distinct architectural property; full trace inventory is in the public repository.',
    r'five representative cases, each highlighting a distinct architectural property; full trace inventory is available in the \href{https://github.com/Lemniscabio/fermdocs}{FermDocs GitHub repository}.'
)

# And explicitly link to the Github Repository in the Open Access checklist
text = text.replace(
    'The complete codebase is fully open source and publicly available, and the evaluation relies on public simulators like IndPenSim~\citep{goldrick2015}.',
    r'The complete codebase is fully open source and publicly available at \url{https://github.com/Lemniscabio/fermdocs}, and the evaluation relies on public simulators like IndPenSim~\citep{goldrick2015}.'
)


with open('paper/main.tex', 'w') as f:
    f.write(text)

print("Linked Github Repo securely.")
