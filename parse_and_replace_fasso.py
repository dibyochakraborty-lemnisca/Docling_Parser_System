import re

def process_file():
    with open('paper/main.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    # Special handling for first instance
    # Find first FermDocs outside of code/hyperlinks
    first_pass = True
    
    def repl(m):
        nonlocal first_pass
        text = m.group(0)
        if "github" in text.lower() or "fermdocs_" in text or "FermDocs_" in text or "src/" in text or "\\url" in text or "href" in text or "texttt" in text:
            return text
        
        if first_pass:
            first_pass = False
            return "FASSO (Fermentation Agentic Scientific Synthesis and Observation)"
        else:
            return "FASSO"

    # We want to match FermDocs exactly
    new_content = re.sub(r'FermDocs(?:_[a-zA-Z0-9_\.]+)?', repl, content)
    
    with open('paper/main.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)

process_file()
