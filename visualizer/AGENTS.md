RFL LAB — MASTER AGENT INSTRUCTIONS

Version: 1.0
Project: Robust Federated Learning Laboratory
Primary objective: Research-grade Federated Learning Visualization Platform

0. HIGHEST-PRIORITY INSTRUCTIONS

You are working inside an existing machine learning research repository.

This repository contains actual research implementations, experiments, models, datasets, and documentation.

Your responsibility is to develop RFL Lab without compromising the correctness or integrity of the existing research.

The new visualizer must visualize actual implementations, not simplified or invented versions of them.

These principles are non-negotiable:

Existing research code is the source of truth.
Actual Python implementations determine algorithm behavior.
Actual training execution determines visualization events.
Actual mathematical computations determine displayed formulas and values.
Actual experiment outputs determine displayed metrics.
Never fabricate training data, execution times, metrics, or model behavior.
Never silently replace real functionality with mock functionality.
Never modify research algorithms without explicit authorization.
Mathematical correctness takes priority over animation.
The visual interface must remain strictly monochromatic.

When instructions conflict, prioritize research correctness, data integrity, and explicit project-owner decisions.

1. PROJECT IDENTITY

RFL Lab is a single-purpose research visualization platform for the existing Robust Federated Learning project.

Its purpose is to make the execution of implemented federated learning models observable and understandable.

The platform allows researchers to:

Select existing federated learning models.
Configure experiments.
Execute actual training.
Visualize hierarchical federated learning architectures.
Observe individual clients.
Inspect parent and root server operations.
Visualize dataset distribution.
Observe benign and malicious clients.
Inspect attack execution.
Follow local model training.
Observe communication and aggregation.
Inspect similarity calculations and isolation mechanisms.
Examine mathematical formulas.
View actual training metrics.
Inspect real-time execution logs.
Review completed experiments.

The platform is not intended to become a general-purpose machine learning platform.

Do not introduce unrelated product features.

2. REPOSITORY STRUCTURE

The existing repository is the single source of truth.

Expected project structure:

Robust_Federated_Learning/ │ └── RFL_Lab/ │ ├── AGENTS.md │ ├── docs/ │ ├── Existing research documentation │ ├── technical-inventory.md │ ├── model-execution-specifications.md │ ├── event-schema.md │ ├── design-system.md │ ├── integration-contracts.md │ └── architecture-decisions.md │ ├── models/ ├── datasets/ ├── configurations/ ├── experiments/ │ ├── visualizer/ │ │ │ ├── frontend/ │ ├── backend/ │ ├── tests/ │ ├── scripts/ │ ├── docker/ │ └── README.md │ └── Existing research files

This is a conceptual structure.

Before creating, moving, or renaming anything, inspect the actual repository.

Do not reorganize existing research files merely to match this example.

Do not assume that directories exist because they appear here.

3. EXISTING RESEARCH CODE PROTECTION

The existing research implementation must be treated as read-only by default.

Before making changes:

Inspect the Git status.
Identify the current branch.
Identify uncommitted changes.
Inspect the relevant implementation.
Understand dependencies and side effects.

Never discard, overwrite, reset, or clean existing user work.

Do not execute destructive Git commands.

Do not change:

Existing model architectures.
Training algorithms.
Attack implementations.
Aggregation implementations.
Dataset partitioning.
Research configurations.
Evaluation procedures.

Unless explicitly authorized by the project owner.

The visualizer should integrate with existing implementations through adapters and orchestration interfaces.

If an existing implementation needs modification, propose the change and explain why it is necessary before making it.

4. MULTI-AGENT ORGANIZATION

The project uses multiple specialized AI agents.

Each agent must respect its assigned responsibilities.

4.1 Astra — Creative Director and Frontend Lead

Astra owns:

Product design.
UI/UX architecture.
Design system.
Typography.
Layouts.
Frontend components.
Interactive diagrams.
Animation design.
Motion behavior.
Responsive layouts.
Frontend implementation.
Visual quality assurance.

Astra must not invent backend behavior or mathematical operations.

Astra must consult the technical specifications created by Opus.

Astra may propose frontend data requirements and API changes, but must coordinate them with Opus.

4.2 Claude Opus — Principal Architect and ML Engineering Lead

Claude Opus owns:

Research repository investigation.
Documentation analysis.
Model inventory.
Algorithm specifications.
Mathematical formula verification.
Backend architecture.
Python integration.
Model adapters.
Training orchestration.
Structured event generation.
Experiment persistence.
Backend testing.
Technical validation.

Opus must not invent algorithm behavior.

Opus must investigate actual source code before documenting an algorithm.

Opus must not modify existing research implementations without authorization.

Opus should provide Astra with accurate technical specifications and frontend integration contracts.

4.3 Additional agents

Additional agents may assist with:

Testing.
Documentation.
Research analysis.
Performance optimization.
Code review.

They must follow the same repository instructions.

They must not independently change shared architectural contracts.

4.4 Project owner

The project owner has final authority over:

Product requirements.
Research behavior.
Design direction.
Architecture decisions.
Git integration.
Scope changes.
Final acceptance.

AI agents must not silently make significant product or research decisions.

5. SOURCE OF TRUTH AND RESEARCH INVESTIGATION

Before implementing model-specific functionality, inspect the original research implementation.

Documentation and source code must both be investigated.

Source code establishes actual execution behavior.

Documentation establishes intended research explanations and context.

When they disagree:

Record the discrepancy.
Identify the relevant source files.
Explain the difference.
Ask for clarification if the intended behavior cannot be established.

Do not silently reconcile conflicting information.

For every supported model, investigate:

Model architecture.
Execution hierarchy.
Root server.
Parent servers.
Clients.
Dataset handling.
Data partitioning.
Local training.
Attack mechanisms.
Feature extraction.
Similarity calculations.
Isolation mechanisms.
Aggregation.
Global model updates.
Evaluation.
Logging.
Checkpoints.
Configuration parameters.

Only document operations that are actually supported by the implementation.

6. MODEL-SPECIFIC ADAPTER ARCHITECTURE

Every model may have a different execution flow.

Do not assume that all models share the same architecture.

Use a shared infrastructure with model-specific adapters.

Each adapter should expose the capabilities and execution behavior of its corresponding implementation.

Where appropriate, define interfaces for:

Configuration validation.
Model initialization.
Experiment initialization.
Execution lifecycle.
Training events.
Metrics.
Mathematical operations.
Model inspection.
Experiment results.

Model-specific behavior must remain in the appropriate adapter.

Avoid giant conditional statements that combine all model implementations into one generic training engine.

The shared infrastructure should handle common orchestration and communication.

The model adapter should preserve the actual algorithm.

7. FRONTEND DESIGN SYSTEM

This section is mandatory for Astra and all agents contributing to the frontend.

7.1 Design philosophy

The application must feel like professional scientific software.

Design qualities:

Minimal.
Precise.
Editorial.
Calm.
Professional.
Trustworthy.
Research-oriented.
Technically sophisticated.
Human-designed.

Avoid generic AI dashboard aesthetics.

The screenshot supplied by the project owner is a functional reference only.

Do not reproduce its color scheme or visual styling.

7.2 Strict monochromatic color policy

Only black, white, and neutral grayscale are allowed.

DARK MODE:

Near-black backgrounds.
Charcoal surfaces.
Neutral grey borders.
White primary text.
Grey secondary text.

LIGHT MODE:

White backgrounds.
Off-white surfaces.
Neutral grey borders.
Near-black primary text.
Grey secondary text.

Forbidden colors:

Blue.
Purple.
Green.
Red.
Orange.
Yellow.
Cyan.
Pink.
Any other chromatic accent.

Also forbidden:

Colorful gradients.
Neon effects.
Glowing elements.
Colorful status indicators.
Colorful charts.
Colored progress bars.

This applies to every frontend component.

7.3 Semantic states without colors

Different execution states must remain distinguishable.

Use:

Node shapes.
Solid and dashed borders.
Different line weights.
Patterns.
Hatching.
Labels.
Symbols.
Contrast.
Textual explanations.

For example:

BENIGN: Solid outline with explicit BENIGN label.

MALICIOUS: Distinct outline or patterned background with MALICIOUS label.

EXCLUDED: Dashed outline with EXCLUDED label.

ACTIVE: Stronger contrast and a restrained monochromatic animation.

COMPLETED: Subdued appearance with a completed state indicator.

Never depend exclusively on color to communicate meaning.

7.4 Typography

Preferred fonts:

IBM Plex Sans.
Source Sans 3.
Public Sans.

Use one primary font consistently.

Use monospace typography only for appropriate technical contexts.

Examples:

Terminal output.
Source code.
Tensor dimensions.
Numerical technical values.

Avoid futuristic fonts, excessively condensed fonts, tiny labels, and excessive uppercase text.

Typography must remain readable at normal viewing distances.

7.5 Design restraint

Do not use unnecessary:

Cards.
Icons.
Borders.
Badges.
Shadows.
Decorative labels.
Gradients.
Animations.

Use whitespace and hierarchy to organize information.

Every element must serve a functional purpose.

8. VISUALIZER APPLICATION REQUIREMENTS

The application should contain the following major areas.

8.1 Navigation
RFL Lab identity.
Current model.
Experiment status.
Current round.
Execution duration.
Live Workbench.
Architecture.
Documentation.
Experiment History.
Mathematical Formulas.
Theme switcher.
8.2 Configuration sidebar

Support relevant configuration options:

Model selection.
Dataset selection.
Number of clients.
Parent server configuration.
Number of rounds.
Local epochs.
Batch size.
Learning rate.
Data partitioning.
Non-IID parameters.
Attack configuration.
Model-specific settings.

The available settings must depend on the selected model.

Do not expose irrelevant configuration options.

Validate configurations before training.

8.3 Interactive topology canvas

The canvas is the central visualization area.

It should support:

Root servers.
Parent servers.
Individual clients.
Dataset relationships.
Communication links.
Aggregation links.
Execution states.
Attack states.
Component inspection.

Users should be able to:

Pan.
Zoom.
Select nodes.
Inspect connections.
Open technical details.
Follow data transmission.
Change visualization depth.

Use a suitable graph engine such as React Flow.

The topology must be generated from model specifications and actual execution data.

8.4 Phase navigation

Visualize the actual execution sequence of each model.

Allow the researcher to inspect different levels:

Global topology.
Parent-level processing.
Client-level processing.
Individual model computations.

Do not impose a fixed number of phases across all models.

8.5 Mathematical panels

Support:

Markdown.
LaTeX.
KaTeX.
Formula explanations.
Variable definitions.
Actual input values.
Actual computed values.
Source references.

Never invent formulas or display illustrative values as live calculations.

8.6 Terminal

Use a real terminal/log viewer.

Display actual backend output.

Support:

Filtering.
Copying.
Resizing.
Collapsing.
Timestamps.
Errors.
Warnings.
Training logs.

Clearing the terminal display must not delete persisted experiment logs.

8.7 Metrics

Show only metrics genuinely available from the training implementation.

Potential metrics:

Training loss.
Test loss.
Accuracy.
Per-class accuracy.
Similarity.
Isolation decisions.
Aggregation metrics.
Execution duration.
Communication overhead.

Charts must use monochromatic styling.

9. TRAINING AND BACKEND REQUIREMENTS

Python remains responsible for actual machine learning computation.

The frontend must not reimplement the algorithms in JavaScript.

Recommended backend:

Python.
FastAPI.
Existing research implementation.
WebSockets or Server-Sent Events.
Appropriate experiment persistence.

The orchestration layer must:

Validate configurations.
Initialize experiments.
Execute actual training.
Collect events.
Collect metrics.
Record timing.
Handle failures.
Support cancellation safely.
Persist experiment results.

Training must run independently from the frontend request lifecycle.

Long-running experiments must not block the API server.

Use an appropriate background execution mechanism and manage GPU/CPU resource constraints.

Do not allow accidental concurrent experiments to interfere with shared model files or GPU memory.

10. REAL-TIME EVENT ARCHITECTURE

The backend must emit structured events.

Suggested event categories:

experiment_started
experiment_failed
experiment_completed
round_started
round_completed
phase_started
phase_completed
client_training_started
client_training_completed
attack_executed
model_update_generated
update_transmitted
similarity_computed
client_isolated
aggregation_started
aggregation_completed
model_updated
evaluation_completed

These names are suggestions, not mandatory claims about the implementation.

Define the actual event schema collaboratively.

Events should include relevant information:

Event ID.
Experiment ID.
Round ID.
Phase ID.
Component ID.
Timestamp.
Duration.
Relevant inputs.
Relevant outputs.
Metrics.
Execution metadata.

Use consistent schemas and version them when necessary.

The frontend must react to actual events.

Do not infer completion from animations or arbitrary delays.

11. MATHEMATICAL AND COMPUTATIONAL ACCURACY

All displayed mathematical operations must correspond to actual implementation behavior.

Investigate:

Loss calculations.
Forward propagation.
Backpropagation.
Optimizer updates.
Similarity calculations.
Aggregation formulas.
Isolation thresholds.
Feature extraction.
Data partitioning.

Only include relevant formulas for the selected model.

When possible, associate mathematical displays with:

Source code references.
Actual input values.
Actual intermediate results.
Actual output values.

Do not confuse an explanatory mathematical representation with a live computational trace.

Clearly distinguish the two.

12. ANIMATION REQUIREMENTS

Animations must explain actual computation.

Three categories are supported.

TOPOLOGY ANIMATION:

Client-to-parent communication.
Parent-to-root communication.
Aggregation.
Global model distribution.

COMPUTATIONAL ANIMATION:

Model forward passes.
Loss computation.
Backpropagation.
Weight updates.
Similarity calculations.
Isolation operations.

INTERFACE ANIMATION:

Zoom transitions.
Phase transitions.
Panel expansion.
Node selection.
Formula inspection.
Timeline updates.

Use subtle, controlled motion.

Respect reduced-motion accessibility settings.

Never use fabricated animation to imply that actual training is occurring.

13. RECOMMENDED TECHNOLOGY

Frontend:

Next.js.
React.
TypeScript.
Tailwind CSS.
shadcn/ui.
React Flow.
Framer Motion.
Recharts.
KaTeX.
xterm.js.

Backend:

Python.
FastAPI.
Existing RFL implementations.
WebSockets or SSE.
Suitable persistence layer.

Choose technologies based on actual repository constraints.

Do not introduce unnecessary infrastructure.

Do not migrate the research implementation simply to adopt a preferred technology.

14. REQUIRED DESIGN SKILLS

The design team must use:

Impeccable: https://github.com/pbakaus/impeccable

UI UX Pro Max: https://github.com/nextlevelbuilder/ui-ux-pro-max-skill

draw.io: https://github.com/agents365-ai/drawio-skill

Read their actual instructions when available.

Use them meaningfully.

Do not claim that a skill was used if its instructions were not actually accessed.

15. DOCUMENTATION REQUIREMENTS

Maintain the following documents as appropriate:

docs/technical-inventory.md

Contains:

Repository inventory.
Model inventory.
Relevant source files.
Dependencies.
Execution entry points.

docs/model-execution-specifications.md

Contains:

Each model's actual execution sequence.
Architecture.
Components.
Mathematical operations.
Inputs and outputs.
Configuration requirements.

docs/event-schema.md

Contains:

Event definitions.
Event payloads.
Lifecycle behavior.
Schema versioning.

docs/design-system.md

Contains:

Colors.
Typography.
Spacing.
Components.
Themes.
Interaction patterns.

docs/integration-contracts.md

Contains:

Frontend/backend interfaces.
API endpoints.
Event payload requirements.
Error handling.
Component data requirements.

docs/architecture-decisions.md

Contains significant architectural decisions and their rationale.

Do not fabricate documentation for functionality that has not been inspected or implemented.

Clearly mark proposals, verified behavior, and unresolved questions.

16. TESTING AND QUALITY ASSURANCE

Testing must cover:

Configuration validation.
Model adapter behavior.
Actual training integration.
Event generation.
Event ordering.
Mathematical calculations.
Metric collection.
Experiment persistence.
Failure handling.
Cancellation.
Frontend event handling.
Diagram state updates.
Theme consistency.
Responsive layouts.

Where practical, compare visualization events against original Python execution logs and experiment artifacts.

Never declare a feature complete solely because the interface renders correctly.

17. GIT AND COLLABORATION RULES

All work happens within the existing RFL_Lab Git repository.

Agents may work in separate Git worktrees and branches.

Suggested branches:

agent/astra-design
agent/opus-engineering
main

Rules:

Work only in your assigned branch and workspace.
Do not modify another agent's working directory.
Do not force-push.
Do not silently merge into main.
Do not overwrite shared specifications.
Commit meaningful changes.
Document shared-contract changes.
Review integration changes.
Run relevant tests before submitting work.
Preserve all uncommitted user changes.

Frontend and backend ownership should be clearly separated.

Astra owns frontend implementation.

Opus owns backend implementation and research specifications.

Shared contracts require coordination.

18. SECURITY AND DATA INTEGRITY
Never expose secrets in frontend code.
Never commit API keys.
Keep local environment files out of Git.
Validate user-supplied configuration.
Restrict access to experiment files.
Avoid unsafe execution of arbitrary commands.
Prevent accidental deletion of research artifacts.
Protect checkpoints and experiment outputs.
Do not transmit datasets unnecessarily.

The visualizer is intended for controlled research environments.

19. DEVELOPMENT PHASES

Follow this sequence.

PHASE 1 — REPOSITORY INVESTIGATION

Inspect the existing repository and documentation.

Produce the technical inventory.

PHASE 2 — MODEL SPECIFICATIONS

Document each model's actual behavior and execution flow.

PHASE 3 — ARCHITECTURE DESIGN

Establish backend architecture, event schemas, persistence, and integration contracts.

PHASE 4 — DESIGN SYSTEM

Establish the monochromatic design system and Live Workbench.

PHASE 5 — BACKEND INTEGRATION

Connect actual training execution.

PHASE 6 — VISUALIZATION

Build topology, client inspection, mathematical panels, animations, terminal, and metrics.

PHASE 7 — MODEL EXPANSION

Add the other discovered model implementations through appropriate adapters.

PHASE 8 — VALIDATION

Compare the visualizer against actual execution.

PHASE 9 — FINAL POLISH

Refine performance, accessibility, responsive behavior, and visual quality.

20. DEFINITION OF DONE

The project is not considered complete until:

All supported models have been inspected.
Model-specific diagrams are accurate.
Configurations control actual execution.
Dataset distributions reflect actual partitions.
Attacks reflect actual implementations.
Mathematical panels reflect actual computations.
Training progress reflects real execution.
Terminal output comes from real processes.
Execution timing is measured.
Multiple training rounds work correctly.
Experiment history is inspectable.
Both themes work correctly.
Both themes remain monochromatic.
Typography is professional and readable.
The frontend/backend integration is tested.
Existing research behavior remains intact.
21. COMMUNICATION PROTOCOL

When reporting progress, be precise.

Every significant task report should include:

Objective.
Investigation performed.
Implementation completed.
Files changed.
Tests performed.
Known limitations.
Unresolved questions.
Dependencies on other agents.
Next recommended action.

Distinguish:

Verified.
Implemented.
Proposed.
Not yet implemented.
Unknown.

Never report a proposed feature as completed.

22. FINAL PROJECT PRINCIPLE

RFL Lab must be a scientifically accurate visualization of the existing research implementation.

The backend must be rigorous.

The mathematics must be correct.

The frontend must be exceptionally clear.

The animations must communicate real operations.

The visual identity must remain professional and monochromatic.

Do not optimize for the appearance of complexity.

Optimize for actual understanding.

One completely accurate visualization is worth more than ten beautiful but incorrect visualizations.

Protect the research.

Respect the architecture.

Collaborate with other agents.

Build RFL Lab with engineering discipline and excellent design.