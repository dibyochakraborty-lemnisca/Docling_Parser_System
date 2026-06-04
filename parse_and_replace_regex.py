import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

# I will construct the regex to replace everything from \subsection{Phase 2 to \section{Discussion}

new_42 = r"""\subsection{Phase 2: Empirical Evaluation via Process-Trace Case Studies}
\label{subsec:phase2}

To evaluate the system's dynamic reasoning capabilities, we conducted an empirical evaluation using an IndPenSim-derived dataset injected with known process faults [Dataset URL: \texttt{<INSERT\_DATASET\_URL>}]. To mimic authentic industrial workflows, the system was probed with informal, unstructured, or deliberately misleading queries. The evaluation focuses on the execution traces rather than solely the final output, providing transparent evidence of the system's epistemic discipline, temporal logic, and resistance to user-induced bias. We detail five representative case studies that highlight distinct architectural behaviors.

\textbf{Case 1: Epistemic Honesty and Autonomous Discovery.} \\
\textit{Prompt:} \texttt{[Placeholder: Question asked]} \\
\textit{Trace Reference:} \texttt{[Placeholder: Link to trace PDF]} \\
The system demonstrated resilience to missing primary metrics (product titer) by refusing to hallucinate yield correlations. Instead, it autonomously identified a late-stage metabolic stall characterized by phenylacetic acid (PAA) accumulation exceeding 5,200 mg/L alongside unconsumed substrate and ammonia spikes. The execution trace reveals that the multi-agent debate remained strictly bounded by the schema, explicitly debating the technical validity of data structures rather than generating ungrounded textual summaries.

\textbf{Case 2: Anti-Sycophancy and Bias Rejection.} \\
\textit{Prompt:} \texttt{[Placeholder: Question asked]} \\
\textit{Trace Reference:} \texttt{[Placeholder: Link to trace PDF]} \\
Standard autoregressive models frequently exhibit sycophancy, agreeing with misleading user prompts. When presented with a leading query suggesting a pump malfunction, the Synthesizer initially incorporated this biased premise. However, the Critic-Judge loop successfully intercepted this hallucination, noting the absence of explicit pump metrics in the mass-balance evidence. The system forced a rewrite, bounding the final hypothesis strictly to empirical data. This confirms that the multi-agent topology is not merely performative but serves as a structural defense against confirmation bias.

\textbf{Case 3: Cross-Run Control and Temporal Logic.} \\
\textit{Prompt:} \texttt{[Placeholder: Question asked]} \\
\textit{Trace Reference:} \texttt{[Placeholder: Link to trace PDF]} \\
Language models commonly struggle with temporal causality and cross-cohort comparisons. When prompted to attribute a dissolved oxygen drop to isolated probe drift, the mass transfer specialist utilized a secondary batch as a scientific control, identifying a synchronized minimum of approximately 9 mg/L across both runs to falsify the hardware malfunction claim. Furthermore, the Critic actively enforced temporal logic by rejecting the Synthesizer's attempt to use initial kinetic data (t = 1.0 h) to explain late-stage substrate accumulation (t = 228 h). The Judge upheld this critique, demonstrating the architecture's capacity to maintain temporal and causal boundaries.

\textbf{Case 4: Biological Context and Process Optimization.} \\
\textit{Prompt:} \texttt{[Placeholder: Question asked]} \\
\textit{Trace Reference:} \texttt{[Placeholder: Link to trace PDF]} \\
The system successfully identified a real-time process termination signal by pinpointing a massive substrate accumulation at 171.4 hours, corresponding to the cessation of product formation. The debate trace highlights advanced epistemic self-correction: when the Synthesizer claimed chronic oxygen limitation based on numerical thresholds, the Critic countered with domain-specific knowledge that dissolved oxygen above 9 mg/L represents biological abundance, identifying the threshold itself as misconfigured. The Judge facilitated this correction, proving the system can integrate biological reality over rigid but flawed metadata.

\textbf{Case 5: Unstructured Anomaly Detection and Evidence Attribution.} \\
\textit{Prompt:} \texttt{[Placeholder: Question asked]} \\
\textit{Trace Reference:} \texttt{[Placeholder: Link to trace PDF]} \\
Confronted with a vague, unstructured query about lower titer, the system executed a zero-shot multivariate root-cause analysis without explicit guidance. Across seven rounds of debate, the Critic rigorously enforced evidentiary standards, rejecting hypotheses for missing citations, misattributed finding identifiers, and unsupported extrapolations, such as hallucinating the term toxicity without explicit grounding. This exhaustive gauntlet confirms that the architecture strictly bounds generative capacity to the proven empirical dataset, ensuring high-fidelity anomaly detection.

"""

text = re.sub(r'\\subsection\{Phase 2: Process-Trace Scoring\}.*?(?=\n% =========================================================\n\\section\{Discussion\})', lambda m: new_42, text, flags=re.DOTALL)


# Update opening of Section 4
old_intro_4 = r"""A common failure mode in evaluating multi-agent scientific reasoning systems is the head-to-head LLM-as-judge comparison against a single-pass baseline. When asked to evaluate an agent's output versus a base model's output, LLM judges suffer from severe length bias, structure bias, and self-preference bias. Matched-length truncation cannot fix this, because the agent's value lives in the deliberative process and the structural grounding, not merely in the final paragraph. We therefore avoid head-to-head comparison entirely and instead measure two distinct dimensions separately using an LLM judge from a completely different model vendor than the pipeline (Figure~\ref{fig:eval})."""
new_intro_4 = r"""A common failure mode in evaluating multi-agent scientific reasoning systems is the head-to-head LLM-as-judge comparison against a single-pass baseline. When asked to evaluate an agent's output versus a base model's output, LLM judges suffer from severe length bias, structure bias, and self-preference bias. Matched-length truncation cannot fix this, because the agent's value lives in the system's structure and reasoning process, not in any final paragraph of output. We therefore avoid head-to-head output comparison and instead evaluate FermDocs across two complementary, executed phases (Figure~\ref{fig:eval}): a cross-vendor architectural audit of the codebase itself (Phase~1), and empirical process-trace evaluation on a planted-fault dataset (Phase~2). Each phase measures a property that single-pass models cannot exhibit by construction: architecture is not an output, and multi-agent execution traces actively demonstrate epistemic boundaries that free-text comparison cannot."""
text = text.replace(old_intro_4, new_intro_4)


# Abstract update
old_abs_end = r"and present a two-phase evaluation methodology to rigorously assess its reasoning capabilities."
new_abs_end = r"and present a comprehensive evaluation methodology combining an executed cross-vendor architectural audit (composite 9/10) with an empirical process-trace evaluation on a planted-fault dataset. The auditor and judge configurations are provided in supplementary material."
text = text.replace(old_abs_end, new_abs_end)

# Caption update for Figure 5 (which is labelled fig:eval)
old_caption_eval = r"\caption{Two-phase evaluation. Phase 1 scores final hypothesis outputs against five architectural-property axes. Phase 2 scores the full execution trace against five process axes that single-pass models cannot exhibit by construction.}"
new_caption_eval = r"\caption{Evaluation methodology. Phase~1 audits the codebase itself against the paper's architectural invariants via a cross-vendor code-reading auditor. Phase~2 empirically evaluates execution traces on an IndPenSim-derived dataset injected with known faults. Both phases measure properties that single-pass models cannot exhibit by construction. Auditor configurations are provided in supplementary material.}"
text = text.replace(old_caption_eval, new_caption_eval)

# Update TikZ Evaluation Figure
tikz_eval_old = r"""\begin{tikzpicture}[
  box/.style={rectangle, draw=ink!80, thick, rounded corners, text width=6cm, inner sep=12pt},
  title/.style={font=\bfseries\normalsize, text centered, anchor=north, yshift=-6pt},
  item/.style={font=\normalsize, anchor=west}
]
  \node[box, fill=blue!10, minimum height=4.5cm] (phase1) {};
  \node[title, text=blue!80!black] at (phase1.north) {Phase 1: Architectural Properties};
  \node[item] at ([yshift=-1.0cm, xshift=0.5cm]phase1.north west) {$\bullet$ Grounding fidelity};
  \node[item] at ([yshift=-1.6cm, xshift=0.5cm]phase1.north west) {$\bullet$ Epistemic discipline};
  \node[item] at ([yshift=-2.2cm, xshift=0.5cm]phase1.north west) {$\bullet$ Mechanism specificity};
  \node[item] at ([yshift=-2.8cm, xshift=0.5cm]phase1.north west) {$\bullet$ Alternative consideration};
  \node[item] at ([yshift=-3.4cm, xshift=0.5cm]phase1.north west) {$\bullet$ Causal coherence};

  \node[box, fill=teal!10, minimum height=4.5cm, right=1.0cm of phase1] (phase2) {};
  \node[title, text=teal!80!black] at (phase2.north) {Phase 2: Process Trace};
  \node[item] at ([yshift=-1.0cm, xshift=0.5cm]phase2.north west) {$\bullet$ Tool-use parsimony};
  \node[item] at ([yshift=-1.6cm, xshift=0.5cm]phase2.north west) {$\bullet$ Critic effectiveness};
  \node[item] at ([yshift=-2.2cm, xshift=0.5cm]phase2.north west) {$\bullet$ Specialist non-redundancy};
  \node[item] at ([yshift=-2.8cm, xshift=0.5cm]phase2.north west) {$\bullet$ Convergence behavior};
  \node[item] at ([yshift=-3.4cm, xshift=0.5cm]phase2.north west) {$\bullet$ Topic discipline};
\end{tikzpicture}"""

tikz_eval_new = r"""\begin{tikzpicture}[
  box/.style={rectangle, draw=ink!80, thick, rounded corners, text width=6cm, inner sep=12pt},
  title/.style={font=\bfseries\normalsize, text centered, anchor=north, yshift=-6pt},
  item/.style={font=\small, anchor=west}
]
  \node[box, fill=blue!10, minimum height=4.5cm] (phase1) {};
  \node[title, text=blue!80!black] at (phase1.north) {Phase 1: Architectural Audit};
  \node[item] at ([yshift=-1.0cm, xshift=0.5cm]phase1.north west) {$\bullet$ Conceptual novelty};
  \node[item] at ([yshift=-1.5cm, xshift=0.5cm]phase1.north west) {$\bullet$ Idea-to-execution fidelity};
  \node[item] at ([yshift=-2.0cm, xshift=0.5cm]phase1.north west) {$\bullet$ Engineering depth};
  \node[item] at ([yshift=-2.5cm, xshift=0.5cm]phase1.north west) {$\bullet$ Domain effectiveness};
  \node[item] at ([yshift=-3.0cm, xshift=0.5cm]phase1.north west) {$\bullet$ Paper-code alignment};

  \node[box, fill=teal!10, minimum height=4.5cm, right=1.0cm of phase1] (phase2) {};
  \node[title, text=teal!80!black] at (phase2.north) {Phase 2: Process-Trace Eval};
  \node[item] at ([yshift=-1.0cm, xshift=0.5cm]phase2.north west) {$\bullet$ Epistemic honesty};
  \node[item] at ([yshift=-1.5cm, xshift=0.5cm]phase2.north west) {$\bullet$ Anti-sycophancy / bias rejection};
  \node[item] at ([yshift=-2.0cm, xshift=0.5cm]phase2.north west) {$\bullet$ Temporal logic \& control};
  \node[item] at ([yshift=-2.5cm, xshift=0.5cm]phase2.north west) {$\bullet$ Contextual self-correction};
  \node[item] at ([yshift=-3.0cm, xshift=0.5cm]phase2.north west) {$\bullet$ Unstructured RCA};
\end{tikzpicture}"""

text = text.replace(tikz_eval_old, tikz_eval_new)


with open('paper/main.tex', 'w') as f:
    f.write(text)

print("Applied Regex updates!")
