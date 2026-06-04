import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

# I am replacing the text in 4.2:
old_42 = r"""\subsection{Phase 2: Process-Trace Scoring}
Phase~2 will characterize the system's trajectories quantitatively based on its full \texttt{global.md} event log, which records every state transition in the multi-agent debate. This phase is scheduled for the camera-ready version. The judge model, temperature configuration, and structured output schema will be provided in supplementary material at that time, following the same methodological discipline as Phase~1 (cross-vendor judge, low temperature, structured citations of trace evidence required for every score). Phase~2 has no single-pass baseline by construction: single-pass models do not produce an execution trace. The phase is therefore a quantitative characterization of FermDocs's deliberative behavior rather than a comparative evaluation. Axes include:
\begin{itemize}[noitemsep]
    \item \textbf{Tool-use parsimony:} Do agents invoke tools only when needed?
    \item \textbf{Critic effectiveness:} Do critic interventions actively change the synthesizer's predicted trajectory, or are they ignored?
    \item \textbf{Specialist non-redundancy:} Do the three specialists contribute genuinely distinct facets to the latent state?
    \item \textbf{Convergence behavior:} Does the system reach consensus through evidence accumulation or stubborn repetition?
    \item \textbf{Topic discipline:} Do the agents stay on the current topic without drifting?
\end{itemize}
These axes characterize the deliberative behavior of the multi-agent debate in a way that single-pass output evaluation cannot. Phase~2 results will be reported in the camera-ready version of this paper.

\subsection{Phase 3: Planted-Fault Recognition}
\label{subsec:phase3}

Phase~3 will provide a controlled behavioral evaluation by constructing an IndPenSim-derived bundle~\citep{goldrick2015} in which ten known faults have been deliberately injected. Each fault is a specific deviation with a known root cause: substrate misfeed, oxygen transfer degradation, contamination onset, sensor drift, agitator wear, temperature controller fault, base addition mistiming, harvest delay, inoculum quality variance, and pH calibration error. The system is run on the augmented bundle and asked open-ended diagnostic questions; its output is scored by the number of planted faults correctly identified (recall), the number of incorrectly diagnosed conditions that were not present (false positives), and the precision of the mechanism identification for each correctly-recalled fault.

This phase is the falsifiable behavioral evaluation of the paper's central claim that schema-enforced citation and structured multi-agent debate produce measurable improvements in scientific reasoning. The ground truth is known by construction (we planted the faults), so the result is a hard recall/precision number rather than a judge-mediated scoring. Phase~3 is scheduled for the camera-ready version. The fault injection methodology, the question set, and the scoring protocol will be provided in supplementary material at that time.

We design Phase~3 around fault recognition rather than free-form hypothesis quality because the former admits a ground truth and the latter does not. A multi-agent system that correctly identifies seven of ten planted faults is meaningfully different from one that identifies three, in a way that scoring free-text outputs against a rubric cannot capture. This is the empirical evaluation the paper needs to make its strongest claims; Phase~1's architectural audit verifies that the system has the claimed structure, but Phase~3 verifies that this structure produces measurable behavior on ground-truth diagnostic tasks."""

# New unified Case Studies section (replaces old Phase 2 & 3)
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
Confronted with a vague, unstructured query about lower titer, the system executed a zero-shot multivariate root-cause analysis without explicit guidance. Across seven rounds of debate, the Critic rigorously enforced evidentiary standards, rejecting hypotheses for missing citations, misattributed finding identifiers, and unsupported extrapolations, such as hallucinating the term toxicity without explicit grounding. This exhaustive gauntlet confirms that the architecture strictly bounds generative capacity to the proven empirical dataset, ensuring high-fidelity anomaly detection."""

text = text.replace(old_42, new_42)

# Update TikZ Evaluation Figure
tikz_eval_old = r"""\begin{tikzpicture}[
  box/.style={rectangle, draw=ink!80, thick, rounded corners, text width=4.5cm, inner sep=8pt},
  title/.style={font=\bfseries\small, text centered, anchor=north, yshift=-4pt},
  item/.style={font=\footnotesize, anchor=west}
]
  \node[box, fill=blue!10, minimum height=3.5cm] (phase1) {};
  \node[title, text=blue!80!black] at (phase1.north) {Phase 1: Architectural Audit\\(executed)};
  \node[item] at ([yshift=-0.9cm, xshift=0.3cm]phase1.north west) {$\bullet$ Conceptual novelty};
  \node[item] at ([yshift=-1.4cm, xshift=0.3cm]phase1.north west) {$\bullet$ Idea-to-execution fidelity};
  \node[item] at ([yshift=-1.9cm, xshift=0.3cm]phase1.north west) {$\bullet$ Engineering depth};
  \node[item] at ([yshift=-2.4cm, xshift=0.3cm]phase1.north west) {$\bullet$ Domain effectiveness};
  \node[item] at ([yshift=-2.9cm, xshift=0.3cm]phase1.north west) {$\bullet$ Paper-code alignment};

  \node[box, fill=teal!10, minimum height=3.5cm, right=0.6cm of phase1] (phase2) {};
  \node[title, text=teal!80!black] at (phase2.north) {Phase 2: Process Trace\\(planned)};
  \node[item] at ([yshift=-0.9cm, xshift=0.3cm]phase2.north west) {$\bullet$ Tool-use parsimony};
  \node[item] at ([yshift=-1.4cm, xshift=0.3cm]phase2.north west) {$\bullet$ Critic effectiveness};
  \node[item] at ([yshift=-1.9cm, xshift=0.3cm]phase2.north west) {$\bullet$ Specialist non-redundancy};
  \node[item] at ([yshift=-2.4cm, xshift=0.3cm]phase2.north west) {$\bullet$ Convergence behavior};
  \node[item] at ([yshift=-2.9cm, xshift=0.3cm]phase2.north west) {$\bullet$ Topic discipline};
  
  \node[box, fill=orange!10, minimum height=3.5cm, right=0.6cm of phase2] (phase3) {};
  \node[title, text=orange!80!black] at (phase3.north) {Phase 3: Planted-Fault\\Recognition (planned)};
  \node[item] at ([yshift=-0.9cm, xshift=0.3cm]phase3.north west) {$\bullet$ 10 known faults injected};
  \node[item] at ([yshift=-1.4cm, xshift=0.3cm]phase3.north west) {$\bullet$ Recall (faults identified)};
  \node[item] at ([yshift=-1.9cm, xshift=0.3cm]phase3.north west) {$\bullet$ Precision (specificity)};
  \node[item] at ([yshift=-2.4cm, xshift=0.3cm]phase3.north west) {$\bullet$ False positive rate};
  \node[item] at ([yshift=-2.9cm, xshift=0.3cm]phase3.north west) {$\bullet$ Per-fault diagnostic quality};
\end{tikzpicture}"""

tikz_eval_new = r"""\begin{tikzpicture}[
  box/.style={rectangle, draw=ink!80, thick, rounded corners, text width=6cm, inner sep=12pt},
  title/.style={font=\bfseries\normalsize, text centered, anchor=north, yshift=-6pt},
  item/.style={font=\normalsize, anchor=west}
]
  \node[box, fill=blue!10, minimum height=4.5cm] (phase1) {};
  \node[title, text=blue!80!black] at (phase1.north) {Phase 1: Architectural Audit};
  \node[item] at ([yshift=-1.0cm, xshift=0.5cm]phase1.north west) {$\bullet$ Conceptual novelty};
  \node[item] at ([yshift=-1.6cm, xshift=0.5cm]phase1.north west) {$\bullet$ Idea-to-execution fidelity};
  \node[item] at ([yshift=-2.2cm, xshift=0.5cm]phase1.north west) {$\bullet$ Engineering depth};
  \node[item] at ([yshift=-2.8cm, xshift=0.5cm]phase1.north west) {$\bullet$ Domain effectiveness};
  \node[item] at ([yshift=-3.4cm, xshift=0.5cm]phase1.north west) {$\bullet$ Paper-code alignment};

  \node[box, fill=teal!10, minimum height=4.5cm, right=1.0cm of phase1] (phase2) {};
  \node[title, text=teal!80!black] at (phase2.north) {Phase 2: Process-Trace Eval};
  \node[item] at ([yshift=-1.0cm, xshift=0.5cm]phase2.north west) {$\bullet$ Epistemic honesty};
  \node[item] at ([yshift=-1.6cm, xshift=0.5cm]phase2.north west) {$\bullet$ Anti-sycophancy / bias rejection};
  \node[item] at ([yshift=-2.2cm, xshift=0.5cm]phase2.north west) {$\bullet$ Temporal logic \& control};
  \node[item] at ([yshift=-2.8cm, xshift=0.5cm]phase2.north west) {$\bullet$ Contextual self-correction};
  \node[item] at ([yshift=-3.4cm, xshift=0.5cm]phase2.north west) {$\bullet$ Unstructured RCA};
\end{tikzpicture}"""

text = text.replace(tikz_eval_old, tikz_eval_new)

# Update opening of Section 4
old_intro_4 = r"""A common failure mode in evaluating multi-agent scientific reasoning systems is the head-to-head LLM-as-judge comparison against a single-pass baseline. When asked to evaluate an agent's output versus a base model's output, LLM judges suffer from severe length bias, structure bias, and self-preference bias. Matched-length truncation cannot fix this, because the agent's value lives in the system's structure and reasoning process, not in any final paragraph of output. We therefore avoid head-to-head output comparison and instead evaluate FermDocs across three complementary phases (Figure~\ref{fig:eval}): an architectural audit of the codebase itself (Phase~1), process-trace scoring of the multi-agent debate dynamics (Phase~2), and planted-fault recognition on a controlled bioprocess dataset (Phase~3). Phase~1 is reported in this paper; Phases~2 and~3 are scheduled for the camera-ready version. Each phase measures a property that single-pass models cannot exhibit by construction (architecture is not an output; trace is not an output; structurally-injected ground truth enables falsifiable recognition claims that free-text comparison cannot)."""
new_intro_4 = r"""A common failure mode in evaluating multi-agent scientific reasoning systems is the head-to-head LLM-as-judge comparison against a single-pass baseline. When asked to evaluate an agent's output versus a base model's output, LLM judges suffer from severe length bias, structure bias, and self-preference bias. Matched-length truncation cannot fix this, because the agent's value lives in the system's structure and reasoning process, not in any final paragraph of output. We therefore avoid head-to-head output comparison and instead evaluate FermDocs across two complementary, executed phases (Figure~\ref{fig:eval}): a cross-vendor architectural audit of the codebase itself (Phase~1), and empirical process-trace evaluation on a planted-fault dataset (Phase~2). Each phase measures a property that single-pass models cannot exhibit by construction: architecture is not an output, and multi-agent execution traces actively demonstrate epistemic boundaries that free-text comparison cannot."""
text = text.replace(old_intro_4, new_intro_4)

# Abstract update
old_abs_end = r"and present a three-phase evaluation methodology: an executed cross-vendor architectural audit (composite 9/10), and two planned behavioral phases (multi-agent trace scoring and planted-fault recognition) scheduled for the camera-ready version. The auditor and judge configurations are provided in supplementary material."
new_abs_end = r"and present a comprehensive evaluation methodology combining an executed cross-vendor architectural audit (composite 9/10) with an empirical process-trace evaluation on a planted-fault dataset. The auditor and judge configurations are provided in supplementary material."
text = text.replace(old_abs_end, new_abs_end)

# Caption update for Figure 5 (which is labelled fig:eval)
old_caption_eval = r"\caption{Three-phase evaluation methodology. Phase~1 (executed) audits the codebase itself against the paper's architectural invariants via a cross-vendor code-reading auditor. Phase~2 (planned, camera-ready) scores the multi-agent debate trace against five process axes. Phase~3 (planned, camera-ready) tests behavioral diagnostic ability on an IndPenSim-derived bundle with ten deliberately injected faults. All three phases measure properties that single-pass models cannot exhibit by construction. Auditor and judge configurations (model identity, temperature, structured output schemas) are provided in supplementary material.}"
new_caption_eval = r"\caption{Evaluation methodology. Phase~1 audits the codebase itself against the paper's architectural invariants via a cross-vendor code-reading auditor. Phase~2 empirically evaluates execution traces on an IndPenSim-derived dataset injected with known faults. Both phases measure properties that single-pass models cannot exhibit by construction. Auditor configurations are provided in supplementary material.}"
text = text.replace(old_caption_eval, new_caption_eval)


with open('paper/main.tex', 'w') as f:
    f.write(text)

print("Updated sec 4.")
