import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

# 1. Update Author and Final flag based on user's provided tex
text = text.replace(r'\usepackage{caisc_2026}', r'\usepackage[final]{caisc_2026}')

old_author = r"""\author{
  Anonymous Author(s) \\
  Affiliation \\
  \texttt{email}
}"""
new_author = r"""\author{
  Dibyo Chakraborty \\
  Lemnisca \\
  \texttt{dibyo.chakraborty@lemnisca.bio}
}"""
text = text.replace(old_author, new_author)

# 2. Add Synap context to Section 3.5
old_35 = r"The defining element of our architecture, differentiating it from virtually all prior AI-for-Science multi-agent systems, is the memory layer (Invariant~2). Traditional RAG systems store text chunks and retrieve them via vector similarity. If they are wrong once, they will retrieve the wrong text again."

new_35 = r"""The defining element of our architecture, differentiating it from virtually all prior AI-for-Science multi-agent systems, is the memory layer (Invariant~2), which is powered by Synap, a managed memory product by Maximem AI. Traditional RAG systems store text chunks and retrieve them via vector similarity. If they are wrong once, they will retrieve the wrong text again. Synap provides state-of-the-art long-term memory retrieval for agentic systems, ensuring our domain-scoped lessons are reliably fetched. We report its performance on standard memory benchmarks in Table~\ref{tab:synap_benchmarks}, confirming its efficacy for complex scientific reasoning~\citep{maximem_eval}.

\begin{table}[htbp]
\centering
\small
\caption{Retrieval performance of the Synap memory backend (Maximem AI) used in FermDocs, evaluated on standard long-term memory benchmarks~\citep{maximem_eval}.}
\label{tab:synap_benchmarks}
\begin{tabular}{lc}
\toprule
\textbf{Benchmark} & \textbf{Accuracy (\%)} \\
\midrule
LoCoMo & 93.2 \\
LongMemEval & 92.0 \\
\bottomrule
\end{tabular}
\end{table}"""

if old_35 in text:
    text = text.replace(old_35, new_35)
else:
    print("Warning: Could not find old_35 text")

# Further Synap integration
text = text.replace(
    r"the memory backend accumulates an increasingly comprehensive set of priors.",
    r"the Synap memory backend accumulates an increasingly comprehensive set of priors."
)

text = text.replace(
    r"persisting validated transitions into a compounding memory, FermDocs occupies",
    r"persisting validated transitions into a compounding memory via Synap, FermDocs occupies"
)

# 3. Add Bibliography Entry
bib_entry = r"""\bibitem[Maximem AI(2026)]{maximem_eval}
Maximem AI.
\newblock Memory and Context Eval Harness.
\newblock \url{https://github.com/maximem-ai/memory_and_context_eval_harness}, 2026.

\bibitem[Pajak"""

text = text.replace(r"\bibitem[Pajak", bib_entry)

with open('paper/main.tex', 'w') as f:
    f.write(text)

print("Applied Synap integration and author details successfully.")
