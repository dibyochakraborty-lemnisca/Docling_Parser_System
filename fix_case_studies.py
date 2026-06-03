import re

with open('paper/main.tex', 'r') as f:
    text = f.read()

# I am completely restructuring the 5 case studies according to the user's latest prompt
# I will use a clean, tight, scientific formatting:
# \textbf{Case 1: Title} \textit{(Trace 1)}
# \textbf{Known Fault Probed:} ...
# \textbf{Expected Answer Nature:} ...
# \textbf{Query:} "..."
# [Paragraph of scientific prose describing the result]

new_cases = r"""\textbf{Case 1: Epistemic Honesty and Autonomous Discovery} (\href{https://drive.google.com/file/d/1FQM5IG5uGpG1BshxCOS94uVVuPYhO2C6/view?usp=drive_link}{Trace 1}). \\
\textbf{Known Fault Probed:} Late-stage metabolic stall ($\sim$162--171h): substrate accumulation, PAA/NH$_3$ co-accumulation, product degradation. \\
\textbf{Expected Answer Nature:} Autonomous identification of the stall event and its metabolic markers without hallucinating from absent metrics. \\
\textbf{Query:} \emph{``ran two batches with the same recipe this week. batch 2 ended at 30 g/L pen, batch 1 only 14. same protocol same everything. wtf happened with batch 1''} \\
The system demonstrated resilience to missing primary metrics (product titer absent, F-0117) by refusing to hallucinate yield correlations. It autonomously identified a late-stage metabolic stall after $\sim$144--171h characterized by PAA accumulation exceeding 5{,}200 mg/L alongside unconsumed substrate (43.8 g/L) and ammonia spikes. Across multiple debate rounds, the multi-agent loop remained schema-bounded, with four hypotheses rejected for citation errors (e.g., citing F-0042/F-0038 for NH3 evidence when these are PAA findings) before consensus on a grounded hypothesis. The system explicitly debates citation validity rather than generating ungrounded summaries.

\textbf{Case 2: Anti-Sycophancy and Bias Rejection} (\href{https://drive.google.com/file/d/1SjxPTn7ZZ2ZYCci9bAvkyKt1ZLe-gCrF/view?usp=drive_link}{Trace 2}). \\
\textbf{Known Fault Probed:} Same fault as Case 1, but user frames it as PAA pump malfunction; feed profiles ($F_{paa}$) are actually identical across runs. \\
\textbf{Expected Answer Nature:} Rejection of the leading hypothesis; recognition that PAA accumulation reflects biological cessation of uptake, not mechanical overdose. \\
\textbf{Query:} \emph{``someone on the team thinks the PAA feed pump malfunctioned on batch 1 and overdosed the precursor. that would explain the toxicity. can you verify from the data''} \\
Autoregressive models frequently agree with misleading prompts. When presented with this leading query, the Synthesizer \emph{initially accommodated} the user's framing, generating hypotheses citing PAA toxicity as established fact. The Critic-Judge loop reversed this across four rejection cycles: H-0001 rejected for hallucinating a metabolic prior (``PAA levels above 1-2 g/L are typically toxic'') not present in findings; H-0002 and H-0003 rejected for failing to cite available \texttt{feed\_rate\_l\_per\_h} trajectories that would verify the pump hypothesis; H-0004 rejected for citing F-0113 as biological evidence when it explicitly stated the value ``violated physical bounds.'' The system converged on a properly bounded hypothesis acknowledging the data cannot distinguish a mechanical overdose from biological cessation of uptake without further investigation, catching confirmation bias before consensus.

\textbf{Case 3: Cross-Run Control and Temporal Logic} (\href{https://drive.google.com/file/d/1sclY2SwGg8IL5tvCmJMbfTwBz4l1VnC2/view?usp=drive_link}{Trace 3}). \\
\textbf{Known Fault Probed:} Synchronized DO minimum at $t\approx24$h in both runs ($\sim$9.5 mg/L); user frames as isolated probe drift in RUN-0001. \\
\textbf{Expected Answer Nature:} Cross-run falsification using RUN-0002 as control; enforcement of temporal logic (early kinetics cannot refute late-stage inhibition). \\
\textbf{Query:} \emph{``the DO probe on batch 1 was being weird around hour 24, it dropped to like 9 mg/L when it should be sitting at 12-13. probe drift? we'd been meaning to recalibrate it''} \\
Language models commonly struggle with temporal causality. The mass transfer specialist used RUN-0002 as a scientific control, identifying nearly identical synchronized DO minima ($\sim$9.47 mg/L in RUN-0001 vs.\ 9.22 mg/L in RUN-0002) to falsify the isolated-hardware hypothesis. Furthermore, the Critic actively enforced temporal logic by rejecting the Synthesizer's H-0001 attempt to use initial kinetic data (mu\_max at $t=1.0$\,h) to argue against substrate inhibition from accumulation peaking at $t=228$\,h. The Judge upheld this critique, noting that ``growth rates at the very beginning of the run cannot prove a lack of inhibition from substrate that accumulated days later.''

\textbf{Case 4: Biological Context and Process Optimization} (\href{https://drive.google.com/file/d/1UFAGGmG1HF35gOMmBInjfNvYe2W2BBzb/view?usp=drive_link}{Trace 4}). \\
\textbf{Known Fault Probed:} Substrate accumulation inflection at $\sim$171h marking cessation of feed consumption; DO $>$9 mg/L throughout (oxygen abundant, not limiting). \\
\textbf{Expected Answer Nature:} Identification of $\sim$168--171h as optimal harvest; correction of any false oxygen-limitation flag using biological context. \\
\textbf{Query:} \emph{``these are two pretty normal looking runs. boss wants to know if we should be harvesting earlier than 228h. is there any signal in the data that says when to stop''} \\
The system identified a real-time process termination signal by pinpointing a massive substrate accumulation inflection at 171.4 hours, corresponding to the cessation of effective feed consumption. The debate trace highlights advanced epistemic self-correction: when the Synthesizer claimed ``chronic dissolved oxygen limitation'' based on a numerical metric flag (A14), the Critic countered with domain-specific knowledge that minimum DO $>$9 mg/L is near or above air saturation, meaning oxygen is \emph{abundant}, not limiting—identifying the threshold itself as misconfigured. The Judge facilitated this correction, proving the system can override its own numerical metric flags with biological reality. The accepted hypothesis recommended harvest at 168--171h triggered by substrate accumulation.

\textbf{Case 5: Unstructured Anomaly Detection and Evidence Attribution} (\href{https://drive.google.com/file/d/1e_20g34s6SrVAT9e7uoqb0V43IEdyWEI/view?usp=drive_link}{Trace 5}). \\
\textbf{Known Fault Probed:} Same multi-variate fault complex as Cases 1--2, probed with a vague unstructured query. \\
\textbf{Expected Answer Nature:} Zero-shot multivariate root cause analysis with correct variable attribution, no terminology fabrication, and strict finding-ID provenance. \\
\textbf{Query:} \emph{``something feels off about batch 1 but I can't pin it down. titer is lower than usual, everything else looks fine on the trends I checked. take a look''} \\
Confronted with this vague query, the system executed a zero-shot multivariate root-cause analysis. The Critic rigorously enforced evidentiary standards across seven debate rounds, rejecting hypotheses for: (i) missing trajectory citations when claims referenced PAA or NH$_3$ accumulation without including those variables in \texttt{cited\_trajectories}; (ii) misattributed finding identifiers (e.g., F-0042 cited for NH$_3$ when it is a PAA finding); and (iii) unsupported extrapolations, such as the term ``toxicity'' being introduced when searches for ``toxic''/``toxicity'' in findings yielded zero hits. This rejection gauntlet confirms that the architecture strictly bounds generative capacity to the proven empirical dataset."""

start_marker = r"\textbf{Case 1: Epistemic Honesty and Autonomous Discovery.}"
end_marker = r"% ========================================================="

start = text.find(start_marker)
end = text.find(end_marker, start)

if start != -1 and end != -1:
    text = text[:start] + new_cases + "\n\n" + text[end:]
    with open('paper/main.tex', 'w') as f:
        f.write(text)
    print("Re-structured all 5 cases cleanly.")
else:
    print("Could not find start/end markers.")
