import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

# I also need to link the Github repository in Section 4.2 as requested by the user
old_42_intro = r"""Five representative cases, each highlighting a distinct architectural property; full trace inventory is in the public repository."""
new_42_intro = r"""We detail five representative case studies, each selected to highlight a distinct architectural property; the full trace inventory is available in the \href{https://github.com/Lemniscabio/fermdocs}{FermDocs GitHub repository}."""

# The previous script might not have found the exact match due to formatting differences. Let's do a wider search
text = re.sub(
    r'five representative cases, each highlighting a distinct architectural property; full trace inventory is in the public repository\.',
    r'We detail five representative case studies, each selected to highlight a distinct architectural property; the full trace inventory is available in the \\href{https://github.com/Lemniscabio/fermdocs}{FermDocs GitHub repository}.',
    text
)

# And in Section 5 (Discussion), mention the repository
text = text.replace(
    'Importantly, FermDocs is fully open-source and publicly available.',
    'Importantly, FermDocs is fully open-source and publicly available at \\url{https://github.com/Lemniscabio/fermdocs}.'
)


with open('paper/main.tex', 'w') as f:
    f.write(text)

print("Linked Github Repo in Sec 4.2 and Sec 5.")
