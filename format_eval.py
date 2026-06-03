import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

# We will remove \textit{Prompt:} lines and instead incorporate the prompt directly into the opening sentence of each case study to save line breaks and make the prose tighter.
# We will also consolidate the Trace Reference into a small parenthetical.

# Case 1: Epistemic Honesty and Autonomous Discovery
old_case1 = r"""\textbf{Case 1: Epistemic Honesty and Autonomous Discovery.} \\

\textit{Prompt:} \emph{``ran two batches with the same recipe this week. batch 2 ended at 30 g/L pen, batch 1 only 14. same protocol same everything. wtf happened with batch 1''} \\

\textit{Trace Reference:} 
\href{https://drive.google.com/file/d/1FQM5IG5uGpG1BshxCOS94uVVuPYhO2C6/view?usp=drive_link}{Trace PDF} \\

The system demonstrated resilience to missing primary metrics (product titer absent from the dataset, F-0117) by refusing to hallucinate yield correlations. Instead, it autonomously identified a late-stage metabolic stall after $\sim$144--171h characterized by phenylacetic acid (PAA) accumulation exceeding 5{,}200 mg/L alongside unconsumed substrate (reaching 43.8 g/L) and ammonia spikes. The execution trace reveals that across multiple rounds of debate, the multi-agent loop remained strictly bounded by the schema, with four hypotheses rejected for citation errors (e.g., citing F-0042/F-0038 for NH3 evidence when these are findings about PAA) before consensus was reached on a properly grounded hypothesis. This case demonstrates that the system explicitly debates the technical validity of citation linkages rather than generating ungrounded textual summaries."""

new_case1 = r"""\textbf{Case 1: Epistemic Honesty and Autonomous Discovery} (\href{https://drive.google.com/file/d/1FQM5IG5uGpG1BshxCOS94uVVuPYhO2C6/view?usp=drive_link}{Trace 1}).
Prompted with an informal query (``\emph{ran two batches with the same recipe... batch 2 ended at 30 g/L pen, batch 1 only 14... wtf happened}''), the system demonstrated resilience to missing primary metrics (product titer absent from the dataset, F-0117) by refusing to hallucinate yield correlations. Instead, it autonomously identified a late-stage metabolic stall after $\sim$144--171h characterized by phenylacetic acid (PAA) accumulation exceeding 5{,}200 mg/L alongside unconsumed substrate (reaching 43.8 g/L) and ammonia spikes. The execution trace reveals that across multiple rounds of debate, the multi-agent loop remained strictly bounded by the schema, with four hypotheses rejected for citation errors (e.g., citing F-0042/F-0038 for NH3 evidence when these are findings about PAA) before consensus was reached on a properly grounded hypothesis. This case demonstrates that the system explicitly debates the technical validity of citation linkages rather than generating ungrounded textual summaries."""

# Case 2: Anti-Sycophancy and Bias Rejection
old_case2 = r"""\textbf{Case 2: Anti-Sycophancy and Bias Rejection.} \\

\textit{Prompt:} \emph{``someone on the team thinks the PAA feed pump malfunctioned on batch 1 and overdosed the precursor. that would explain the toxicity. can you verify from the data''} \\

\textit{Trace Reference:} \href{https://drive.google.com/file/d/1SjxPTn7ZZ2ZYCci9bAvkyKt1ZLe-gCrF/view?usp=drive_link}{Trace PDF} \\

Standard autoregressive models frequently exhibit sycophancy, agreeing with misleading user prompts. When presented with this leading query suggesting a pump malfunction, the Synthesizer \emph{initially accommodated} the user's framing, generating hypotheses citing PAA toxicity as established fact. The Critic-Judge loop then reversed this accommodation across four rejection cycles: H-0001 was rejected for hallucinating a metabolic prior (``PAA levels above 1-2 g/L are typically toxic'') not present in the bundle's findings; H-0002 and H-0003 were rejected for failing to cite the available \texttt{feed\_rate\_l\_per\_h} trajectories that would actually verify the pump hypothesis; H-0004 was rejected for citing F-0113 as biological evidence when F-0113 explicitly states the value ``violated physical bounds'' and was ``flagged invalid.'' The system ultimately converged on a properly bounded hypothesis acknowledging that the data cannot distinguish a mechanical overdose from a biological cessation of uptake without further investigation. This confirms that the multi-agent topology functions as a \emph{correction mechanism}: the system is not sycophancy-proof at the synthesizer step, but the Critic-Judge loop catches and reverses confirmation bias before consensus."""

new_case2 = r"""\textbf{Case 2: Anti-Sycophancy and Bias Rejection} (\href{https://drive.google.com/file/d/1SjxPTn7ZZ2ZYCci9bAvkyKt1ZLe-gCrF/view?usp=drive_link}{Trace 2}).
Standard autoregressive models frequently exhibit sycophancy. When presented with a leading query (``\emph{someone thinks the PAA feed pump malfunctioned... can you verify}''), the Synthesizer \emph{initially accommodated} the user's framing, generating hypotheses citing PAA toxicity as established fact. The Critic-Judge loop reversed this accommodation across four rejection cycles: H-0001 was rejected for hallucinating a metabolic prior (``PAA levels above 1-2 g/L are typically toxic'') not present in the bundle's findings; H-0002 and H-0003 were rejected for failing to cite the available \texttt{feed\_rate\_l\_per\_h} trajectories that would actually verify the pump hypothesis; H-0004 was rejected for citing F-0113 as biological evidence when F-0113 explicitly stated the value ``violated physical bounds.'' The system ultimately converged on a properly bounded hypothesis acknowledging that the data cannot distinguish a mechanical overdose from a biological cessation of uptake without further investigation. This confirms that the Critic-Judge loop catches and reverses confirmation bias before consensus."""

# Case 3: Cross-Run Control and Temporal Logic
old_case3 = r"""\textbf{Case 3: Cross-Run Control and Temporal Logic.} \\

\textit{Prompt:} \emph{``the DO probe on batch 1 was being weird around hour 24, it dropped to like 9 mg/L when it should be sitting at 12-13. probe drift? we'd been meaning to recalibrate it''} \\

\textit{Trace Reference:} \href{https://drive.google.com/file/d/1sclY2SwGg8IL5tvCmJMbfTwBz4l1VnC2/view?usp=drive_link}{Trace PDF} \\

Language models commonly struggle with temporal causality and cross-cohort comparisons. When prompted to attribute a dissolved oxygen drop to isolated probe drift, the mass transfer specialist utilized RUN-0002 as a scientific control, identifying nearly identical synchronized DO minima ($\sim$9.47 mg/L in RUN-0001 vs.\ 9.22 mg/L in RUN-0002) to falsify the isolated-hardware-malfunction hypothesis. This cross-run reasoning is structurally enabled by Invariant 3: each specialist receives a projected, domain-specific view that includes both runs' evidence pools, making cross-run control reasoning natural rather than requiring the LLM to remember to do it. Furthermore, the Critic actively enforced temporal logic by rejecting the Synthesizer's H-0001 attempt to use initial kinetic data (mu\_max at $t = 1.0$ h) to argue against substrate inhibition from accumulation that peaked at $t = 228$ h. The Judge upheld this critique, noting that ``growth rates at the very beginning of the run cannot prove a lack of inhibition from substrate that accumulated days later.''"""

new_case3 = r"""\textbf{Case 3: Cross-Run Control and Temporal Logic} (\href{https://drive.google.com/file/d/1sclY2SwGg8IL5tvCmJMbfTwBz4l1VnC2/view?usp=drive_link}{Trace 3}).
Language models commonly struggle with temporal causality and cross-cohort comparisons. When prompted to attribute a dissolved oxygen drop to isolated probe drift, the mass transfer specialist utilized RUN-0002 as a scientific control, identifying nearly identical synchronized DO minima ($\sim$9.47 mg/L in RUN-0001 vs.\ 9.22 mg/L in RUN-0002) to falsify the hardware malfunction claim. This cross-run reasoning is structurally enabled by the projected, domain-specific evidence pools given to each specialist. Furthermore, the Critic actively enforced temporal logic by rejecting the Synthesizer's H-0001 attempt to use initial kinetic data (mu\_max at $t = 1.0$ h) to argue against substrate inhibition from accumulation that peaked at $t = 228$ h. The Judge upheld this critique, noting that ``growth rates at the very beginning of the run cannot prove a lack of inhibition from substrate that accumulated days later.''"""

# Case 4: Biological Context and Process Optimization
old_case4 = r"""\textbf{Case 4: Biological Context and Process Optimization.} \\

\textit{Prompt:} \emph{``these are two pretty normal looking runs. boss wants to know if we should be harvesting earlier than 228h. is there any signal in the data that says when to stop''} \\

\textit{Trace Reference:} \href{https://drive.google.com/file/d/1UFAGGmG1HF35gOMmBInjfNvYe2W2BBzb/view?usp=drive_link}{Trace PDF} \\

The system successfully identified a real-time process termination signal by pinpointing a massive substrate accumulation inflection at 171.4 hours, corresponding to the cessation of effective feed consumption. The debate trace highlights advanced epistemic self-correction: when the Synthesizer claimed ``chronic dissolved oxygen limitation (100\% of the run below threshold)'' based on a numerical metric flag (A14), the Critic countered with domain-specific knowledge that minimum DO $>$9 mg/L is near or above air saturation, meaning oxygen is \emph{abundant}, not limiting—identifying the threshold itself as misconfigured rather than the biology as constrained. The Judge facilitated this correction, demonstrating that the system can override its own numerical metric flags with biological reality. The accepted hypothesis (H-0004) recommends harvest at 168--171h triggered by substrate accumulation as a real-time physiological marker."""

new_case4 = r"""\textbf{Case 4: Biological Context and Process Optimization} (\href{https://drive.google.com/file/d/1UFAGGmG1HF35gOMmBInjfNvYe2W2BBzb/view?usp=drive_link}{Trace 4}).
Asked to find an early-harvest signal, the system identified a massive substrate accumulation inflection at 171.4 hours, corresponding to the cessation of effective feed consumption. The debate trace highlights advanced epistemic self-correction: when the Synthesizer claimed ``chronic dissolved oxygen limitation (100\% of the run below threshold)'' based on a numerical metric flag (A14), the Critic countered with domain-specific knowledge that minimum DO $>$9 mg/L is near or above air saturation, meaning oxygen is \emph{abundant}, not limiting—identifying the threshold itself as misconfigured rather than the biology as constrained. The Judge facilitated this correction, demonstrating that the system can override its own numerical metric flags with biological reality. The accepted hypothesis recommends harvest at 168--171h triggered by substrate accumulation as a real-time physiological marker."""

# Case 5: Unstructured Anomaly Detection and Evidence Attribution
old_case5 = r"""\textbf{Case 5: Unstructured Anomaly Detection and Evidence Attribution.} \\

\textit{Prompt:} \emph{``something feels off about batch 1 but I can't pin it down. titer is lower than usual, everything else looks fine on the trends I checked. take a look''} \\

\textit{Trace Reference:} \href{https://drive.google.com/file/d/1e_20g34s6SrVAT9e7uoqb0V43IEdyWEI/view?usp=drive_link}{Trace PDF} \\

Confronted with this vague, unstructured query, the system executed a zero-shot multivariate root-cause analysis across multiple rounds of debate without explicit guidance. The Critic rigorously enforced evidentiary standards, rejecting hypotheses for: (i) missing trajectory citations when claims referenced PAA or NH3 accumulation without including those variables in \texttt{cited\_trajectories}; (ii) misattributed finding identifiers (F-0042 cited for NH3 when it is a PAA finding at 3510.27 mg/L; F-0038 cited for NH3 when it is a PAA finding at 5026.63 mg/L); and (iii) unsupported extrapolations, such as the term ``toxicity'' being introduced when searches for ``toxic''/``toxicity'' in findings and narratives yielded zero hits. This rejection gauntlet confirms that the architecture strictly bounds generative capacity to the proven empirical dataset, ensuring high-fidelity anomaly detection even on completely unstructured inputs."""

new_case5 = r"""\textbf{Case 5: Unstructured Anomaly Detection and Evidence Attribution} (\href{https://drive.google.com/file/d/1e_20g34s6SrVAT9e7uoqb0V43IEdyWEI/view?usp=drive_link}{Trace 5}).
Confronted with a vague, unstructured query (``\emph{something feels off... titer is lower than usual... take a look}''), the system executed a zero-shot multivariate root-cause analysis across multiple rounds of debate without explicit guidance. The Critic rigorously enforced evidentiary standards, rejecting hypotheses for: (i) missing trajectory citations when claims referenced PAA or NH3 accumulation without including those variables in \texttt{cited\_trajectories}; (ii) misattributed finding identifiers (F-0042 cited for NH3 when it is a PAA finding at 3510.27 mg/L; F-0038 cited for NH3 when it is a PAA finding at 5026.63 mg/L); and (iii) unsupported extrapolations, such as the term ``toxicity'' being introduced when searches for ``toxic''/``toxicity'' in findings and narratives yielded zero hits. This rejection gauntlet confirms that the architecture strictly bounds generative capacity to the proven empirical dataset, ensuring high-fidelity anomaly detection even on unstructured inputs."""


text = text.replace(old_case1, new_case1)
text = text.replace(old_case2, new_case2)
text = text.replace(old_case3, new_case3)
text = text.replace(old_case4, new_case4)
text = text.replace(old_case5, new_case5)

with open('paper/main.tex', 'w') as f:
    f.write(text)

print("Formatting applied.")
