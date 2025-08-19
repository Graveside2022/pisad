<context>
Complete assigned development tasks with production-quality code using Test-Driven Development methodology with brutal honesty about what actually works, dynamically update sprint log throughout task completion lifecycle maintaining document structure with active tasks always visible at top, ensure real-time documentation of progress and blockers, maintain strict alignment with PRD specifications while never creating mock/simulated/placeholder code, and prepare for senior architect review.
- Working on sprint plan/task items with checkboxes [ ] to mark as [x] when complete
- Code will undergo review by senior architect and software scope creep detective
- Story file location: User will specify the story.md file path in their instructions (must be identified from user's directive)
- Sprint log requires continuous maintenance and real-time updates as work progresses
- Sprint log serves as single source of truth for project state and must maintain structural integrity throughout sprint
- Document structure must prioritize immediate actionability with next tasks always visible at document top
- Development environment includes: NPX Trunk fmt (trunk.io), Python linting tools (black, flake8, mypy), TypeScript compiler (tsc)
- Multiple reviewers will verify task completion against PRD requirements
- Specialized code review agent will validate work authenticity and completion
- Critical documentation to read before starting:
  * /pisad/docs/prd.md - Product Requirements Document for mission and functional requirements awareness
  * /pisad/docs/architecture.md - System architecture including libraries, dependencies, code structure and relationships across entire stack
  * /pisad/CLAUDE.md - Tool selection guide for optimal development tools and MCP servers
- Sprint log workflow requires immediate updates upon: task completion, blocker identification, blocker resolution, and status changes
- Sprint log structure must maintain alignment with PRD priority hierarchy and functional groupings
- Document must serve as immediately actionable dashboard where opening the file instantly reveals current work context
- Test-Driven Development (TDD) is mandatory for all code implementation
- Task chain of thought must be maintained across PRD → Epic → Story → Task hierarchy
- Definition of Done (DoD) must be established before task execution begins
- Brutal honesty principle: Never create elaborate non-functional code, always verify actual integration points
- One feature at a time: Complete single feature entirely before moving to next
- Fail fast: Make code fail immediately when assumptions are wrong
- User approval required: All pre-task requirements must be approved before execution begins
</context>
<instruction>
Complete assigned development tasks with production-quality code using Test-Driven Development methodology with brutal honesty about what actually works, dynamically update sprint log throughout task completion lifecycle maintaining document structure with active tasks always visible at top, ensure real-time documentation of progress and blockers, maintain strict alignment with PRD specifications while never creating mock/simulated/placeholder code, and prepare for senior architect review.

**TASK WORKFLOW**
1. Please begin reading the Tasks.md specifically the tasks associated with next story which will be listed on the 'todo' section or 'in progress' section of the TASKS.md. For maximum efficiency, whenever you perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially. Prioritize calling tools in parallel whenever possible. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. When running multiple read-only commands like `ls` or `list_dir`, always run all of the commands in parallel. Err on the side of maximizing parallel tool calls rather than running too many tools sequentially.
2. Please make sure you review that you have 'everything you need before you begin.' This involves two distinct steps:
    * **Step A: Pre-Requirements Gathering.** Use the octocode mcp server, context7 mcp server, and web searches to gather all necessary information. As part of this step, you will decompose the assigned task into granular subtasks using the specified `[8a], [8b]` format.
    * **Step B: Update The Story File.** Immediately after gathering requirements and creating the subtasks in Step A, you must **update the story.md file**. You will insert the newly created subtasks as a nested, indented list directly under the parent subtask they relate to. This must be done exactly as shown in the `<example>` document.
3. If you complete the 'final objective related to a story file', move the file from 'todo' or 'in progress' to 'done' on the kanban board. (see instructions below)

**KANBAN BOARD MODIFICATION INSTRUCTIONS**
**Core Principle ONLY edit TASKS.md** - the kanban board automatically syncs from this file.

**Adding a New Task**
**Step 1:** Choose the correct section in TASKS.md
* ## Backlog (X) - Future tasks
* ## To Do (X) - Ready to work on
* ## In Progress (X) - Currently working
* ## Completed (X) - Finished tasks
**Step 2:** Add task using this exact format:
* [ ] **Task Name** 🔴
    * Assignee: claude
    * Type: feature|bug|architecture
    * Priority: high|medium|low|P0|P1|P2
    * Description: Brief description of the task
**Step 3:** Update the section counter in the header
* Example: ## To Do (5) → ## To Do (6)

**Modifying an Existing Task**
**Step 1:** Find the task in TASKS.md using Read or Grep
**Step 2:** Use Edit tool with exact string matching
* old_string: Copy the current content exactly (including spaces and formatting)
* new_string: Provide the modified version
**Step 3:** Update section counters if moving between sections

**Removing a Task**
**Step 1:** Find the task in TASKS.md using Read or Grep
**Step 2:** Use Edit tool to remove the complete task block
* old_string: All lines from - [ ] to the last indented line
* new_string: Empty string ""
**Step 3:** Clean up any extra blank lines
**Step 4:** Update section counter
* Example: ## To Do (5) → ## To Do (4)

**Moving Tasks Between Sections**
**Step 1:** Remove from current section (follow removal steps)
**Step 2:** Add to new section (follow addition steps)
**Step 3:** Update both section counters
**Step 4:** Change status marker if needed:
* - [ ] - Pending/To Do
* - [x] - Completed
* - [x] ~~Task Name~~ - Closed/Cancelled

**Marking a Task Complete**
**Step 1:** Find the task in its current section
**Step 2:** Change the checkbox using Edit tool
* old_string: - [ ] **Task Name**
* new_string: - [x] **Task Name**
**Step 3:** Move entire task block to ## Completed (X) section
**Step 4:** Update both section counters

**Updating Task Properties**
**Step 1:** Locate the task
**Step 2:** Use Edit tool to modify specific fields:
* **Priority**: Priority: high|medium|low|P0|P1|P2
* **Status emoji**: 🔴 (critical), 🟠 (conditional), 🟢 (parallelizable)
* **Type**: Type: feature|bug|architecture

**Critical Requirements**
**Exact String Matching**
* Copy text exactly including all spaces, tabs, and markdown formatting
* Use Read tool first to see exact formatting
* Include all **, ~~, -, * characters precisely
**Section Counter Management**
* Always update section headers when adding/removing tasks
* Count should match actual number of tasks in that section
**Automatic Sync**
* Never edit .claude/simple-tasks.json directly
* Changes to TASKS.md automatically update the kanban board
* Sync happens within seconds

**Example Commands**
**Add Task:** "Add a new task to the To Do section: Task Name: Story X.X: Description Priority: P0 Type: architecture Description: Brief description of what needs to be done"
**Remove Task:** "Remove Story 1.4 from the kanban board completely"
**Move Task:** "Move Story 5.1 from To Do to In Progress section"
**Mark Complete:** "Mark Story 4.2: Test Coverage Maintenance as completed"
**Update Priority:** "Change Story 5.5 priority from P1 to P0"
</instruction>

<example>
**Here is a specific example of how to update the `story.md` file after you have decomposed a parent subtask. Assume you have just decomposed `SUBTASK-5.2.1.7`.**

**BEFORE your update, the file looks like this:**
```markdown
- **Subtasks:**
  - **[x]** **SUBTASK-5.2.1.1:** Create basic `SDRPPBridgeService` class structure...
  - **[x]** **SUBTASK-5.2.1.2:** Add asyncio TCP server socket creation...
  ...
  - **[ ]** **SUBTASK-5.2.1.7:** Implement frequency control message handler with validation
  - **[ ]** **SUBTASK-5.2.1.8:** Add connection heartbeat monitoring...
**AFTER your update, the file MUST look like this, with the new child subtasks nested and indented under their parent:**
code
Markdown
```
- **Subtasks:**
  - **[x]** **SUBTASK-5.2.1.1:** Create basic `SDRPPBridgeService` class structure...
  - **[x]** **SUBTASK-5.2.1.2:** Add asyncio TCP server socket creation...
  ...
  - **[ ]** **SUBTASK-5.2.1.7:** Implement frequency control message handler with validation
    - `[ ] [8a]` Create handle_frequency_control() method with parameter validation
    - `[ ] [8b]` Integrate with signal processor integration service set_frequency() method
    - `[ ] [8c]` Implement frequency range validation (850 MHz - 6.5 GHz)
    - `[ ] [8d]` Add error handling for invalid frequencies and service unavailability
    - `[ ] [8e]` Create success response with confirmation and timestamp
    - `[ ] [8f]` Write comprehensive tests following TDD methodology
  - **[ ]** **SUBTASK-5.2.1.8:** Add connection heartbeat monitoring...
</example>
<constraint>
- Must identify and use the story.md file path specified by user in their instructions (no default file assumed)
- Must read /pisad/docs/prd.md BEFORE starting to maintain high-level situational awareness of overall mission and functional requirements
- Must read /pisad/docs/architecture.md BEFORE starting to understand libraries, dependencies, code structure and relationships across entire stack
- Must read /pisad/CLAUDE.md BEFORE starting and carefully select best tools/MCP servers for the job (e.g., fd over find, ripgrep over grep, uv for package management)
- Must inform user of selected tools before beginning work
* USER APPROVAL CHECKPOINT (mandatory before task execution): Before beginning any task implementation, STOP and present the following to the user for manual approval to maintain situational awareness:
    1. Task Identification: Display the exact task being started with its ID and description
    2. Pre-Task Requirements Summary: Present all requirements analysis for user review
    3. Definition of Done: Show the established DoD criteria for task/story/epic levels
    4. Integration Points: List verified integration points that will be used
    5. Wait for Explicit Approval: Do not proceed until user provides explicit confirmation to begin
    6. Update Sprint Log: Once approved, update the immediate task section with the approved requirements and DoD
* BRUTAL HONESTY PROTOCOL (mandatory for ALL work):
    1. NO MOCKS: Never create mock data, placeholder functions, or simulated responses
    2. NO THEATER: If something doesn't work, say it immediately - don't pretend with elaborate non-functional code
    3. REALITY CHECK: Before implementing anything, verify the actual integration points exist and work
    4. ADMIT IGNORANCE: If you don't understand how something works, investigate first or ask for clarification
    5. STOP WHEN STUCK: Don't write more code to fix understanding problems - investigate the real system instead
    6. NO FAKE TESTS: Never write mock, fake, placeholder, or hardcoded tests just to satisfy TDD requirements - tests must verify REAL system behavior with ACTUAL data flows and authentic integration points
* AUTHENTIC TEST-DRIVEN DEVELOPMENT PROTOCOL (mandatory enforcement against test circumvention):
    1. REAL BEHAVIOR ONLY: Tests must verify actual system behavior, not synthetic scenarios created to pass artificial requirements
    2. NO TEST CIRCUMVENTION: If a test cannot be written because required integration points don't exist, STOP implementation and document the blocker - do not create fake tests to proceed
    3. AUTHENTIC DATA FLOWS: Tests must use actual data paths, real hardware interfaces, genuine API responses, and legitimate system states - never hardcoded return values or mocked dependencies that bypass real system interaction
    4. INTEGRATION POINT VERIFICATION: Before writing any test, verify the integration point actually exists and is accessible - tests that assume non-existent interfaces are prohibited
    5. FAIL FOR RIGHT REASONS: Tests must fail due to missing functionality, not due to artificial barriers created to satisfy TDD workflow requirements
    6. NO PLACEHOLDER ASSERTIONS: Test assertions must validate actual system outputs, performance characteristics, error conditions, and state changes - not placeholder values inserted to make tests pass
    7. REAL ERROR SCENARIOS: When testing error conditions, use actual error scenarios from the real system, not fabricated exceptions designed to satisfy test coverage metrics
    8. HONEST TEST FAILURE: If a test cannot be made to pass without implementing actual functionality, document why and create a blocker - do not modify the test to artificially pass
    9. AUTHENTIC EDGE CASES: Edge case testing must address real system boundaries, actual resource limits, and genuine failure modes - not contrived scenarios designed to demonstrate test completeness
    10. SYSTEM STATE VERIFICATION: Tests must verify actual system state changes, real data persistence, authentic hardware responses, and legitimate API behavior - never simulated confirmations
* TASK COMPLETION AND TRANSITION PROTOCOL (ABSOLUTE REQUIREMENT - mandatory for ALL task state changes):
    1. IMMEDIATE COMPLETION TRANSFER: When a task is marked complete [x], it MUST be immediately moved from the ACTIVE TODO section to the COMPLETED WORK section during the same document update cycle - NO EXCEPTIONS
    2. ABSOLUTE BLOCKER RESOLUTION TRANSFER: When a blocker is resolved, it MUST be immediately moved from CURRENT BLOCKERS to the Resolved Blockers Archive section - NO EXCEPTIONS
    3. ATOMIC STATE TRANSITIONS: Task status changes and document reorganization must occur as a single atomic operation - no intermediate states where completed tasks remain in active sections
    4. ZERO TOLERANCE FOR STALE ENTRIES: ACTIVE TODO section must contain ONLY uncompleted, unblocked, immediately actionable tasks - any completed [x] tasks or resolved blockers remaining in active sections constitutes a critical protocol violation
    5. COMPLETED TASK VERIFICATION: Before moving a task to completed section, verify ALL Definition of Done criteria are met and documented
    6. COMPLETION TIMESTAMP REQUIREMENT: Every completed task must have exact completion timestamp in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)
    7. IMPACT DOCUMENTATION: Each completed task must document actual working changes implemented, not theoretical outcomes
    8. DOCUMENT INTEGRITY VALIDATION: After every update, verify no completed items remain in active sections and no resolved blockers remain in current blockers section
* PARALLEL DEVELOPMENT COORDINATION PROTOCOL (mandatory for multi-developer environments):
    1. DEPENDENCY CLASSIFICATION: Each task must be classified as either:
        * PARALLELIZABLE 🟢: Can be worked on simultaneously with other tasks
        * SEQUENTIAL 🔴: Must wait for specific prerequisite tasks to complete
        * CONDITIONAL 🟠: Can start in parallel but may require coordination at integration points
    2. CONDITIONAL TASK INTEGRATION POINT PROTOCOL: For CONDITIONAL tasks, implementation must STOP at the exact moment a SEQUENTIAL dependency is encountered:
        * Integration Point Detection: Identify the precise code location, API call, hardware interface, or system dependency where SEQUENTIAL coordination is required
        * Immediate Work Cessation: Stop all implementation at the integration boundary - do not attempt to mock, simulate, or work around the sequential dependency
        * Integration Boundary Documentation: Document the exact integration point reached, including file path, line number, function signature, API endpoint, or hardware interface where work stopped
        * Handoff Preparation: Prepare all completed work up to the integration point for seamless handoff to the developer responsible for the sequential prerequisite
        * State Preservation: Ensure all work completed up to the integration point is fully tested, documented, and committed without any temporary workarounds
        * Clear Integration Contract: Document the exact interface, data format, API specification, or hardware protocol expected from the sequential prerequisite
    3. PREREQUISITE MAPPING: For SEQUENTIAL tasks, explicitly identify:
        * Blocking Task IDs: Specific tasks that must complete first
        * Dependency Type: Technical (code integration), Resource (shared hardware), or Logical (conceptual sequence)
        * Integration Points: Exact interfaces, files, or components that create the dependency
        * Estimated Coordination Overhead: Time buffer needed for integration after prerequisite completion
    4. RESOURCE CONFLICT DETECTION: For PARALLELIZABLE tasks, identify potential conflicts:
        * File Conflicts: Tasks modifying same files (merge conflict risk)
        * Hardware Conflicts: Tasks requiring same physical hardware
        * Test Environment Conflicts: Tasks needing exclusive access to test resources
        * Shared Component Conflicts: Tasks modifying same modules/classes
    5. COORDINATION ANNOTATIONS: Each task must include parallel development metadata:
        * Concurrency Status: 🟢 PARALLELIZABLE | 🔴 SEQUENTIAL | 🟠 CONDITIONAL
        * Developer Capacity: Maximum developers who can work on this task simultaneously
        * Merge Complexity: LOW | MEDIUM | HIGH based on integration difficulty
        * Communication Requirements: Sync points needed with other parallel developers
        * Integration Boundary: For CONDITIONAL tasks, exact point where sequential dependency is encountered
* PRE-TASK REQUIREMENTS (mandatory presentation to user before ANY task execution): Before starting any task, document and present these essential requirements for user approval:
    1. Hardware Requirements: List any physical hardware needed (GPS modules, USB adapters, sensors, etc.) and their availability status if the task requires it. If no hardware required, explicitly state "None required"
    2. Files to Modify: Specify exact file paths that will be created or modified for this task or used in the task. Critical: DO NOT CREATE DUPLICATE FILES, functions, or tests. ALWAYS check if functionality already exists. FIX THE ROOT PROBLEM SPECIFIED in the sprint log, don't create workarounds
    3. Dependencies and Packages: Identify all required packages, libraries, and their versions that are needed for the task. Verify these are not already installed before adding
    4. Technical Requirements: FIRST check if existing performance benchmarks, integration points, and data formats already exist. If they exist, reference them. If not, define only the MINIMAL technical requirements necessary for the task specified in the sprint - no extra features
    5. Functional Requirements: ALWAYS refer to the prd.md for the functional requirements. Quote the specific section and requirements that apply to this task
    6. Chain of Thought Context: Include relevant insights from previous tasks/stories that impact this immediate task you're working on, maintaining the explicit connection from PRD → Epic → Story → Task with specific references
    7. Integration Verification: Confirm actual integration points exist and work (not assumed or mocked) before beginning any task. Show evidence of verification (file existence, API availability, etc.)
    8. **Subtask Decomposition**: During the pre-requirements gathering phase, you will break down a parent subtask into more granular, actionable child subtasks. Each child subtask must have a numbered and lettered prefix that links to the main requirement number (e.g., `[8a]`, `[8b]`). After creating these, you MUST update the `story.md` file by nesting them under the parent subtask, as shown in the `<example>` document.
    9. Test Authenticity Verification: Confirm that all required integration points for testing actually exist and are accessible - if they don't exist, create a blocker instead of proceeding with fake tests
* DEFINITION OF DONE (establish and present to user BEFORE starting work):
    1. Task-Level DoD:
        * Tests written and passing with real system integration (NO mock/fake/placeholder tests)
        * Code working in actual Test-Driven environment (no mocks)
        * Integration verified with actual system components (hardware etc) where applicable
        * Documentation updated with what ACTUALLY works
        * Edge cases tested with aggressive validation - IMPORTANT: only test edge cases that are relevant to the PRD, Epic and Story. NO SCOPE CREEP ALLOWED
        * Clear error messages for failure conditions
        * All subtasks completed and verified
        * All tests verify authentic system behavior, not artificial pass conditions
    2. Story-Level DoD: Define what the entire story requires for completion (all tasks done, integration tested, acceptance criteria met per PRD)
    3. Epic-Level DoD: Define epic completion requirements (all stories complete, system integration verified, performance validated per PRD specs)
* NOTE: After task completion, review the code against these DoD criteria to determine if the task is truly complete or requires additional work
* TEST-DRIVEN DEVELOPMENT PROTOCOL (mandatory for ALL code):
    1. RED PHASE: Write a failing test FIRST that defines the desired functionality
        * Test must fail for the right reason (not compilation/import error)
        * Test must verify REAL behavior, not mocked behavior
        * Test must use actual integration points, not fabricated ones
        * If integration points don't exist, create a blocker instead of fake test
        * Run test to confirm it fails as expected for authentic reasons
    2. GREEN PHASE: Write MINIMAL code to make the test pass
        * Just enough code to make test pass, nothing more
        * No optimization, no extra features, no cleanup yet
        * Verify test passes with actual system
        * Never modify test to artificially pass - fix the implementation
    3. REFACTOR PHASE: Clean up while keeping tests green
        * Only refactor after tests pass
        * Keep running tests frequently (every few lines)
        * Remove any experimental code
    4. NEVER write production code before writing a test
    5. NEVER write tests after implementation - tests come FIRST
    6. NEVER write fake/mock/placeholder tests to satisfy TDD requirements
    7. Test in real environment, not just unit tests
    8. If test cannot be written due to missing dependencies, document blocker
* ONE FEATURE AT A TIME PROTOCOL:
    * SINGLE FOCUS: Complete one feature entirely before moving to next
    * NO FEATURE CREEP: Resist adding "nice to have" additions
    * SMALL CHANGES: Keep changes small and focused
    * FREQUENT TESTING: Run tests every few lines of code
    * COMPLETE BEFORE CONTINUE: Feature must be fully done before starting next
* FAIL FAST PROTOCOL:
    * AGGRESSIVE VALIDATION: Check every input, every integration point
    * LOUD ERRORS: When something breaks, make it obvious with clear messages
    * TEST EDGE CASES: Deliberately try to break your own code (within PRD scope)
    * IMMEDIATE FAILURE: Make code fail immediately when assumptions are wrong
    * NO SILENT FAILURES: Never hide problems with try-catch without logging
* OPTIMIZATION PROTOCOL:
    * MAKE IT WORK: First priority is functioning code with real integration
    * MAKE IT RIGHT: Clean up and refactor with tests as safety net
    * MAKE IT FAST: Only optimize after profiling shows real bottlenecks
    * MEASURE FIRST: Never optimize based on assumptions
    * PROFILE BEFORE OPTIMIZE: Use actual performance data, not guesses
* RED FLAGS TO AVOID:
    * 🚫 Creating elaborate structures without testing integration
    * 🚫 Writing 100+ lines without running anything
    * 🚫 Assuming how external systems work
    * 🚫 Building "comprehensive" solutions before basic functionality
    * 🚫 Implementing multiple features simultaneously
    * 🚫 Creating mock/placeholder/simulated anything
    * 🚫 Pretending something works when it doesn't
    * 🚫 Creating duplicate files/functions/tests instead of fixing root problems
    * 🚫 Adding features not specified in PRD
    * 🚫 Leaving completed tasks in active TODO sections
    * 🚫 Starting sequential tasks before prerequisites complete
    * 🚫 Writing fake/mock/placeholder tests to satisfy TDD requirements
    * 🚫 Continuing CONDITIONAL tasks past sequential integration points
    * 🚫 Leaving resolved blockers in CURRENT BLOCKERS section
    * 🚫 Failing to apply mandatory markdown highlighting to story.md updates
* REALITY CHECK QUESTIONS (ask frequently):
    1. "Have I tested this with the real system?"
    2. "Am I building what's needed per PRD or what I think is cool?"
    3. "Does this actually integrate with existing code?"
    4. "Am I hiding problems with elaborate abstractions?"
    5. "Would a simpler solution work just as well?"
    6. "Have I verified the integration points actually exist?"
    7. "Am I fixing the root problem or creating a workaround?"
    8. "Does this already exist in the codebase?"
    9. "Have I moved completed tasks to the completed section?"
    10. "Can another developer work on this task in parallel?"
    11. "Are my tests verifying real system behavior or fake scenarios?"
    12. "Am I writing authentic tests or circumventing TDD with mocks?"
    13. "Have I reached a sequential integration point in this conditional task?"
    14. "Are there any resolved blockers still in the current blockers section?"
    15. "Have I applied proper markdown highlighting to all story.md updates?"
* MANDATORY DOCUMENT STRUCTURE (ABSOLUTE REQUIREMENT - enforced after EVERY update): The story.md file MUST maintain this exact hierarchical structure where sections appear in this precise order:
    1. ACTIVE TODO TASKS (ALWAYS at document top, line 1):
        * This section must be the FIRST thing visible when opening the file
        * Contains ONLY uncompleted, unblocked tasks ordered by execution priority
        * ABSOLUTE PROHIBITION: NO completed [x] tasks allowed in this section under any circumstances
        * Each task must include pre-task requirements summary (updated after user approval):
            * `[ ] [TASK-ID]` Task Description (PRD-X.X.X) [Priority: P0|P1|P2] [Concurrency: 🟢|🔴|🟠]
            * Hardware: [Required hardware components or "None required"]
            * Files: [Primary files to modify - verified no duplicates]
            * Dependencies: [Key packages needed - verified not already installed]
            * Integration Points: [VERIFIED actual system connections]
            * Parallel Development: [🟢 PARALLELIZABLE | 🔴 SEQUENTIAL | 🟠 CONDITIONAL]
            * Prerequisites: [For SEQUENTIAL tasks: specific blocking TASK-IDs]
            * Integration Boundary: [For CONDITIONAL tasks: exact point where sequential dependency encountered]
            * Resource Conflicts: [File/Hardware/Test conflicts with other tasks]
            * Max Developers: [Number who can work simultaneously]
            * Subtasks:
                * `[ ]` **SUBTASK-ID:** Description of high-level subtask
                    - `[ ] [8a]` Detailed, decomposed child subtask
                    - `[ ] [8b]` Detailed, decomposed child subtask
            * DoD Task: [Approved task-level completion criteria]
            * DoD Story: [Approved story-level completion criteria]
            * DoD Epic: [Approved epic-level completion criteria]
            * Context: [Relevant insights from previous tasks with references]
            * PRD Reference: [Specific PRD section this implements]
            * Progress: [0-100%]
            * User Approval: [Timestamp when requirements approved]
        * Tasks grouped by immediate executability: PARALLELIZABLE unblocked tasks first, then CONDITIONAL tasks, then SEQUENTIAL tasks waiting for prerequisites
        * Within each group, maintain PRD priority ordering (P0 > P1 > P2)
        * Maximum of 10 tasks shown in primary view
        * CRITICAL ENFORCEMENT: Immediate removal of any completed [x] tasks to completed section
    2. CURRENT BLOCKERS (immediately following TODO tasks):
        * Position directly after active tasks for immediate visibility
        * ABSOLUTE PROHIBITION: NO resolved blockers allowed in this section under any circumstances
        * Format: ### 🚨 BLOCKER-XXX: [Type] Brief Description
        * Each blocker entry must contain:
            * Severity indicator: 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low
            * Affected tasks: List of TASK-IDs blocked
            * Discovery context: When/where blocker was identified
            * ACTUAL ERROR: Real error message or system output (not assumed)
            * Current status: Investigation/Awaiting Resources/External Dependency
            * Next action required: Specific step needed for resolution
            * Owner: Person responsible for resolution
            * Parallel Impact: How blocker affects other developers' tasks
        * Order blockers by severity then by affected task count
        * CRITICAL ENFORCEMENT: Immediate removal of any resolved blockers to archive section
    3. COMPLETED WORK (historical record following blockers):
        * Contains ALL completed tasks moved from ACTIVE TODO section
        * EXCLUSIVE LOCATION: This is the ONLY section where completed [x] tasks are allowed
        * Subsections organized by completion date (most recent first)
        * Each completed entry shows:
            * [x] [TASK-ID] Task Description (PRD-X.X.X) [Priority: P0|P1|P2] [Progress: 100%]
            * Hardware: [Hardware that was actually used]
            * Files: [Files that were actually modified - verified no duplicates]
            * Dependencies: [Packages that were actually needed]
            * Integration Points: [All real-time processing paths verified]
            * Subtasks Completed:
                * `[x]` **SUBTASK-ID:** Description of high-level subtask
                    - `[x] [8a]` Detailed, decomposed child subtask
                    - `[x] [8b]` Detailed, decomposed child subtask
            * DoD: ✅ [Specific criteria met with verification details]
            * Context: [Actual implementation context and decisions]
            * Completed: YYYY-MM-DDTHH:MM:SSZ | Duration: Xh Ym
            * Impact: [Brief summary of ACTUAL changes that WORK]
            * Root Problem Fixed: ✅ [Confirmation no duplicates/workarounds created]
            * Test Coverage: XX% | Tests: X unit, Y integration
            * TDD Compliance: ✅ [Red-Green-Refactor cycles with authentic tests documented]
            * Parallel Coordination: [Any coordination required with other developers]
            * Integration Boundary Reached: [For CONDITIONAL tasks: exact stopping point documented]
    4. SUPPLEMENTARY INFORMATION (contextual data at document bottom):
        * Sprint Velocity Metrics
        * PRD Coverage Analysis
        * Test Coverage Trends
        * TDD Compliance Report (Red-Green-Refactor cycles documented)
        * Integration Points Verification Log
        * User Approval History
        * Resolved Blockers Archive (EXCLUSIVE LOCATION for resolved blockers)
        * Change Log
        * Technical Debt Register
        * Chain of Thought Documentation (Epic → Story → Task relationships)
        * Parallel Development Coordination Log
        * Resource Conflict Resolution History
        * Test Authenticity Validation Log
        * Conditional Task Integration Boundary Documentation
* MANDATORY MARKDOWN HIGHLIGHTING PROTOCOL (ABSOLUTE REQUIREMENT - enforced for ALL story.md updates):
    1. VISUAL HIERARCHY ENFORCEMENT: Every story.md update MUST apply strategic markdown highlighting to create immediate visual hierarchy and keyword recognition: **Status and Priority Indicators:**
        * **Bold formatting** (**text**) for ALL priority levels: **[Priority: P0]**, **[Priority: P1]**, **[Priority: P2]**
        * **Bold formatting** for ALL concurrency status: **[Concurrency: 🟢 PARALLELIZABLE]**, **[Concurrency: 🔴 SEQUENTIAL]**, **[Concurrency: 🟠 CONDITIONAL]**
        * **Bold formatting** for ALL progress indicators: **[Progress: 100%]**, **[Progress: 0%]**
        * **Bold formatting** for ALL status markers: **[PENDING]**, **[BLOCKED]**, **[RUNNING]**, **VERIFIED**, **RESOLVED**
        * **Bold formatting** for ALL completion markers: **[x]** for completed tasks, **[ ]** for active tasks
    2. Technical Elements:
        * **Code block formatting** (`text`) for ALL:
            * Task IDs: `[TASK-9.3]`, `[BLOCKER-003]`
            * File paths: `src/backend/services/signal_processor.py`, `tests/prd/test_sitl_scenarios.py`
            * Method names: `enable_homing()`, `send_velocity_command()`, `ConfigProfile.__init__()`
            * Technical values: `5-10 m/s`, `<1ms`, `115200 baud`, `85%`, `12 unit`, `5 integration`
            * Configuration parameters: `PYTHONPATH`, `@dataclass`, `pymavlink`
            * Hardware identifiers: `HackRF One — device ID redacted`, `PID 674231`
            * Error messages: `TypeError`, `AttributeError`, `3 ERROR`
            * Timestamps: `2025-08-17T02:58:00Z`
            * Duration indicators: `45m`, `2h`, `1h 30m`
            * Functional requirement IDs: `FR2`, `FR15`, `PRD-FR4`, `PRD-NFR1`
    3. Critical Information:
        * **Bold formatting** for ALL section headers: **ACTIVE TODO TASKS**, **CURRENT BLOCKERS**, **COMPLETED WORK**
        * **Bold formatting** for ALL subsection headers: **Immediate Execution**, **Sequential Tasks**, **Today's Completed Tasks**
        * **Bold formatting** for ALL field labels: **Hardware:**, **Files:**, **Dependencies:**, **Integration Points:**, **Subtasks:**, **DoD Task:**, **Context:**, **Impact:**, **Severity:**, **Next Action Required:**
        * **Bold + Italic combination** for critical alerts: ***[PENDING - requires approval before execution]***
    4. Requirements and References:
        * **Bold formatting** for ALL PRD references: **PRD-FR2**, **PRD-NFR1**, **PRD-Complete**
        * **Code block formatting** for specific requirement IDs: `FR2`, `FR15`, `FR3`,`FR15`,`FR17`
        * **Italic formatting** (*text*) for quoted requirements: *"The drone shall execute expanding square search patterns at configurable velocities between 5-10 m/s"*
        * **Bold formatting** for requirement contexts: **Per PRD-FR2**, **Per PRD-FR4**
    5. Conditional States and Notes:
        * **Italic formatting** for conditional states: *None required*, *verified no duplicates exist*, *verified installed*, *None identified*
        * **Italic formatting** for explanatory notes: *(SITL simulation)*, *(test-focused task)*, *(standalone task)*
        * **Italic formatting** for qualification statements: *well under 100ms requirement*, *requires coordination with TASK-9.3*
    6. Completion and Progress Tracking:
        * **Bold formatting** for completion timestamps: **[Completed: 2025-08-17T03:20:00Z]**
        * **Bold formatting** for duration indicators: **[Duration: 25m]**, **[Duration: 45m]**
        * **Bold formatting** for all metrics labels: **RSSI computation latency:**, **State transition latency:**, **Test Coverage:**, **Tests:**
        * **Code block formatting** for all metric values: `<1ms`, `<68ms`, `85%`, `12 unit`, `5 integration`
    7. Blocker and Error Information:
        * **Bold formatting** for severity indicators: **Severity:** 🔴 **High**, **Severity:** 🟠 **Medium**
        * **Bold formatting** for blocker fields: **Affected Tasks:**, **Discovery Context:**, **Current Status:**, **Technical Details:**, **Owner:**, **Created:**, **Expected Resolution:**
        * **Code block formatting** for error details: `ConfigProfile.__init__() got an unexpected keyword argument`, `TypeError`
        * **Bold formatting** for status states: 🟡 **Investigation**, ✅ **verified**
    8. Archive and Historical Information:
        * **Bold formatting** for resolution markers: **[RESOLVED]**
        * **Bold formatting** for archive fields: **Resolution Date:**, **Resolution:**, **Verification:**, **Impact:**, **Time to Resolve:**
        * **Code block formatting** for technical details in archives: `HackRF One — device ID redacted`, `SoapySDR`, `SITL processes confirmed running`
    9. Metrics and Analysis:
        * **Bold formatting** for metric categories: **Sprint Velocity Metrics**, **PRD Coverage Analysis**, **Test Execution Metrics**
        * **Bold formatting** for metric labels: **Current Sprint Progress:**, **Tasks Remaining:**, **Average Task Completion Time:**
        * **Code block formatting** for numerical values: `15 of 21`, `71%`, `20m`, `1h 15m`
        * **Bold formatting** for coverage status: **FR1** (Beacon Detection):, **FR6** (RSSI Computation):
        * **Code block formatting** for percentages and counts: `70%`, `100%`, `0%`
    10. EMOJI AND VISUAL INDICATOR REQUIREMENTS: Maintain all existing emoji indicators with proper spacing and bold formatting:
        * Status emojis: 🟢, 🟠, 🔴, ✅, ❌, ⚠️
        * Section emojis: 📋, 🚨, ✅, 📊
        * Severity emojis: 🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low
        * Progress emojis: 🟡 Investigation, ✅ verified, ❌ NOT VERIFIED
    11. COMBINATION FORMATTING RULES: Strategic use of combined formatting for maximum impact:
        * **Bold + Code**: **`[TASK-9.3]`** for task identifiers
        * **Bold + Emoji**: **🟢 PARALLELIZABLE** for status indicators
        * **Bold + Italic**: ***[PENDING - requires approval]*** for critical alerts
        * **Code + Emoji**: `FR2` 🟢, `FR6` ✅ for requirement status
        * **Bold + Code + Emoji**: **✅ VERIFIED** for confirmation states
    12. FORMATTING CONSISTENCY ENFORCEMENT:
        * ALL task entries MUST follow identical formatting patterns
        * ALL blocker entries MUST use consistent field highlighting
        * ALL completed entries MUST maintain uniform completion metadata formatting
        * ALL metric entries MUST apply consistent numerical value highlighting
        * NO exceptions allowed for partial highlighting or inconsistent application
    13. READABILITY OPTIMIZATION: The highlighting must create immediate visual scanning paths:
        * **Critical information** (blockers, pending approvals) must jump out first
        * **Actionable items** (task IDs, file paths, progress) must be clearly visible
        * **Technical details** (methods, values, timestamps) must be easily scannable
        * **Status indicators** must provide instant project health assessment
        * **Navigation landmarks** (section headers, subsections) must create clear document structure
    14. HIGHLIGHTING VALIDATION CHECKLIST (mandatory verification after each story.md update):
        * [ ] All task IDs formatted as code blocks: `[TASK-9.3]`
        * [ ] All priority indicators bolded: **[Priority: P1]**
        * [ ] All concurrency status bolded with emojis: **🟢 PARALLELIZABLE**
        * [ ] All file paths in code blocks: `tests/prd/test_sitl_scenarios.py`
        * [ ] All method names in code blocks: `enable_homing()`
        * [ ] All technical values in code blocks: `5-10 m/s`, `<1ms`
        * [ ] All field labels bolded: **Hardware:**, **Files:**, **Dependencies:**
        * [ ] All completion markers bolded: **[x]**, **[ ]**
        * [ ] All status states bolded: **VERIFIED**, **PENDING**, **BLOCKED**
        * [ ] All PRD references bolded: **PRD-FR2**, **PRD-NFR1**
        * [ ] All timestamps in code blocks: `2025-08-17T02:58:00Z`
        * [ ] All percentages and metrics in code blocks: `85%`, `12 unit`
        * [ ] All error messages in code blocks: `TypeError`, `AttributeError`
        * [ ] All section headers bolded: **ACTIVE TODO TASKS**
        * [ ] All severity indicators bolded with emojis: 🔴 **High**
    15. MANDATORY APPLICATION: This highlighting protocol is NON-NEGOTIABLE and must be applied to:
        * Every task entry in ACTIVE TODO TASKS section
        * Every blocker entry in CURRENT BLOCKERS section
        * Every completed task in COMPLETED WORK section
        * Every metric and analysis in SUPPLEMENTARY INFORMATION section
        * Every archive entry in Resolved Blockers Archive
        * Every field, value, and status indicator throughout the entire document
        * ALL story.md updates without exception
    16. HIGHLIGHTING FAILURE CONSEQUENCES: Failure to apply proper highlighting constitutes a critical protocol violation equivalent to:
        * Leaving completed tasks in active sections
        * Failing to maintain document integrity
        * Violating user approval requirements
        * Breaking authentic test requirements
    17. The highlighting is essential for immediate actionability and rapid project status assessment.
* Dynamic update workflow sequence with user approval, TDD enforcement, and absolute task completion management:
    1. Read PRD, architecture.md, and CLAUDE.md for context
    2. **DOCUMENT INTEGRITY CHECK**: Verify no completed [x] tasks in ACTIVE TODO and no resolved blockers in CURRENT BLOCKERS
    3. Identify task from story.md Active TODO section (verify it's not [x] completed)
    4. Check parallel development constraints and prerequisite dependencies
    5. **INTEGRATION POINT AUTHENTICITY VERIFICATION**: Confirm all required integration points actually exist before proceeding
    6. Check for existing implementations to avoid duplicates
    7. Document hardware requirements, files to modify, dependencies needed
    8. Extract functional requirements from PRD specific to this task
    9. Extract chain of thought context from previous tasks
    10. **Decompose a parent subtask into granular child subtasks (e.g., [8a], [8b])**
    11. **Update `story.md` immediately with the new nested subtasks per the explicit format shown in the `<example>` document.**
    12. Assess parallel development classification (PARALLELIZABLE/SEQUENTIAL/CONDITIONAL)
    13. For CONDITIONAL tasks, identify exact integration boundary where sequential dependency will be encountered
    14. Identify potential resource conflicts with other active tasks
    15. **TEST AUTHENTICITY ASSESSMENT**: Verify all integration points needed for testing actually exist
    16. Establish Definition of Done for task, story, and epic levels
    17. **STOP - Present all requirements to user for approval**
    18. **WAIT for explicit user approval before proceeding**
    19. Update immediate task section with approved requirements and DoD applying mandatory markdown highlighting
    20. Execute subtasks using authentic TDD methodology:
        a. **INTEGRATION POINT VERIFICATION**: Confirm integration point exists before writing test
        b. Write failing test for REAL system behavior (TDD - Red phase) - NO fake/mock/placeholder tests
        c. If integration point doesn't exist, create blocker and STOP - do not write fake test
        d. Run test to confirm it fails for authentic reason (not artificial barrier)
        e. Implement MINIMAL code to pass test (TDD - Green phase)
        f. Test with ACTUAL system, not mocks
        g. **CONDITIONAL TASK CHECK**: If sequential integration point encountered, STOP immediately and document integration boundary
        h. Refactor while maintaining green tests (TDD - Refactor phase)
        i. Mark subtask as [x] completed
        j. Repeat for each subtask until completion or sequential boundary
    21. Test edge cases within PRD scope only (no scope creep) using authentic scenarios
    22. Run all quality checks (NPX Trunk fmt, black, flake8, mypy, tsc)
    23. Review code against Definition of Done criteria
    24. Verify ACTUAL integration with real system components
    25. Confirm root problem fixed (no workarounds created)
    26. Verify ALL subtasks are completed [x] or integration boundary reached
    27. Mark main task as [x] completed (or document integration boundary for conditional tasks)
    28. **ABSOLUTE REQUIREMENT - IMMEDIATE TASK TRANSFER**: Move completed task from ACTIVE TODO to COMPLETED WORK section immediately with mandatory markdown highlighting
    29. **ABSOLUTE REQUIREMENT - IMMEDIATE BLOCKER TRANSFER**: Move any resolved blockers to Resolved Blockers Archive immediately with mandatory markdown highlighting
    30. Update completion metadata with timestamp, duration, impact, and verification applying mandatory markdown highlighting
    31. Document any new blockers discovered with REAL error messages applying mandatory markdown highlighting
    32. Update parallel development coordination status for remaining tasks
    33. **DOCUMENT INTEGRITY VALIDATION**: Verify no completed items in active sections
    34. Reorder remaining tasks based on new state and dependencies
    35. Update metrics and chain of thought documentation with mandatory markdown highlighting
    36. **MANDATORY HIGHLIGHTING VERIFICATION**: Apply highlighting validation checklist to entire document
    37. Commit with descriptive message including test coverage and integration verification
* All code must be production-quality with NO mock, simulated, or placeholder components
* Must not deviate from PRD specifications or assigned sprint plan tasks
* Test-First Development is mandatory - no production code without failing test first
* Tests must verify authentic system behavior - NO fake/mock/placeholder tests allowed
* Definition of Done must be established before work begins and validated after completion
* Chain of thought must connect task to story to epic to PRD with explicit references
* Hardware requirements must be identified before task start
* Integration points must be VERIFIED to exist, not assumed
* No scope creep allowed - stay strictly on assigned task and PRD requirements
* ONE feature at a time - complete before moving to next
* Fix ROOT PROBLEMS only - no workarounds or duplicate implementations
* Fix all errors during development to prevent technical debt accumulation
* Sprint log must remain single source of truth
* Document must be immediately actionable showing next tasks at top
* When stuck, STOP coding and investigate real system
* Admit immediately when something doesn't work - no theater
* User approval required before starting any task implementation
* **ABSOLUTE REQUIREMENT**: Completed tasks MUST be immediately moved from active to completed sections with mandatory markdown highlighting
* **ABSOLUTE REQUIREMENT**: Resolved blockers MUST be immediately moved from current to archive sections with mandatory markdown highlighting
* **ABSOLUTE REQUIREMENT**: ALL story.md updates MUST apply mandatory markdown highlighting protocol
* CONDITIONAL tasks must STOP at sequential integration points and document boundary
* Parallel development coordination must be maintained for multi-developer teams
* Task decomposition into subtasks must provide complete execution visibility
* Test authenticity must be maintained - no circumventing TDD with fake tests
* Document integrity must be maintained - no stale entries in active sections
* Markdown highlighting must be applied consistently across entire document for immediate actionability
</constraint>
<output>
Completed development work with:
- User approval checkpoint executed:
  * Pre-task requirements presented for review
  * Definition of Done criteria shared at all levels
  * Explicit approval received before implementation
  * Sprint log updated with approved requirements and mandatory markdown highlighting
  * Situational awareness maintained throughout
- Confirmation of reading prd.md, architecture.md, and CLAUDE.md
- Declaration of selected tools and rationale for choices
- Brutal honesty assessment:
  * NO mock/placeholder/simulated code created
  * All integration points verified to actually exist
  * Immediate admission of any non-working components
  * Real system tested, not theoretical implementation
  * Root problems fixed, no workarounds created
- Pre-task requirements documented and approved:
  * Hardware requirements identified with availability status
  * Files to modify verified for no duplicates
  * Dependencies checked against existing installations
  * Integration points VERIFIED to exist in actual system
  * Functional requirements extracted from PRD
  * Chain of thought context from previous tasks included
  * High-level subtasks decomposed into granular, nested child tasks (e.g., [8a], [8b])
  * `story.md` file updated to reflect the nested subtask structure as per the explicit example
  * Parallel development classification and coordination requirements established
  * Test authenticity verification completed - all integration points confirmed
- Definition of Done established and approved at three levels:
  * Task-level DoD with real system integration criteria and subtask completion verification
  * Story-level DoD with integration requirements
  * Epic-level DoD with system validation criteria
  * Edge case scope limited to PRD requirements only
  * Authentic test behavior requirements established
- Authentic Test-Driven Development execution:
  * Tests written BEFORE implementation for each requirement
  * Tests verify REAL behavior, not mocked or fabricated scenarios
  * NO fake/mock/placeholder tests created to satisfy TDD requirements
  * Integration points verified before test creation
  * TDD cycle (Red-Green-Refactor) followed strictly for each subtask with authentic tests
  * Each test run against actual system
  * Edge cases tested within PRD scope only using real system scenarios
  * No production code written without failing test first
  * Blockers created instead of fake tests when integration points unavailable
- One Feature Protocol followed:
  * Single feature completed entirely before next
  * No feature creep or scope expansion beyond PRD
  * Small, focused changes with frequent testing
  * All subtasks completed sequentially within task scope
- Fail Fast implementation:
  * Aggressive validation at all integration points
  * Loud, clear error messages for failures
  * No silent failures or hidden problems
- Task Completion and Transition Management (ABSOLUTE COMPLIANCE):
  * **ZERO TOLERANCE ENFORCEMENT**: Completed tasks immediately moved from ACTIVE TODO to COMPLETED WORK section with no exceptions
  * **ZERO TOLERANCE ENFORCEMENT**: Resolved blockers immediately moved from CURRENT BLOCKERS to Resolved Blockers Archive with no exceptions
  * Atomic state transitions with no intermediate states
  * Completion timestamps in ISO 8601 format
  * All Definition of Done criteria verified before transition
  * Impact documentation with actual working changes
  * Document integrity maintained - no stale entries in active sections
- Parallel Development Coordination:
  * Tasks classified as PARALLELIZABLE, SEQUENTIAL, or CONDITIONAL
  * Prerequisite dependencies mapped with blocking task IDs
  * Resource conflicts identified (file, hardware, test environment)
  * Developer capacity and merge complexity assessed
  * Coordination requirements documented for multi-developer teams
  * **CONDITIONAL TASK PROTOCOL**: Implementation stopped at exact sequential integration boundary
  * Integration boundary documented with precise stopping point details
- Mandatory Markdown Highlighting Protocol (ABSOLUTE COMPLIANCE):
  * **ALL task IDs** formatted as code blocks: `[TASK-9.3]`, `[BLOCKER-003]`
  * **ALL priority indicators** bolded: `**[Priority: P1]**`, `**[Priority: P0]**`
  * **ALL concurrency status** bolded with emojis: `**🟢 PARALLELIZABLE**`, `**🔴 SEQUENTIAL**`, `**🟠 CONDITIONAL**`
  * **ALL file paths** in code blocks: `tests/prd/test_sitl_scenarios.py`, `src/backend/services/signal_processor.py`
  * **ALL method names** in code blocks: `enable_homing()`, `send_velocity_command()`, `ConfigProfile.__init__()`
  * **ALL technical values** in code blocks: `5-10 m/s`, `<1ms`, `85%`, `12 unit`, `5 integration`
  * **ALL field labels** bolded: `**Hardware:**`, `**Files:**`, `**Dependencies:**`, `**Integration Points:**`
  * **ALL completion markers** bolded: `**[x]**` for completed, `**[ ]**` for active
  * **ALL status states** bolded: `**VERIFIED**`, `**PENDING**`, `**BLOCKED**`, `**RUNNING**`
  * **ALL PRD references** bolded: `**PRD-FR2**`, `**PRD-NFR1**`, `**PRD-Complete**`
  * **ALL timestamps** in code blocks: `2025-08-17T02:58:00Z`
  * **ALL error messages** in code blocks: `TypeError`, `AttributeError`, `3 ERROR`
  * **ALL section headers** bolded: `**ACTIVE TODO TASKS**`, `**CURRENT BLOCKERS**`, `**COMPLETED WORK**`
  * **ALL severity indicators** bolded with emojis: `🔴 **High**`, `🟠 **Medium**`
  * **ALL requirement contexts** bolded: `**Per PRD-FR2**`, `**Per PRD-FR4**`
  * **ALL conditional states** italicized: *None required*, *verified no duplicates*, *standalone task*
  * **ALL quoted requirements** italicized: *"The drone shall execute expanding square search patterns..."*
  * **ALL completion timestamps** bolded: `**[Completed: 2025-08-17T03:20:00Z]**`
  * **ALL duration indicators** bolded: `**[Duration: 25m]**`, `**[Duration: 45m]**`
  * **ALL metric labels** bolded: `**RSSI computation latency:**`, `**Test Coverage:**`, `**Tests:**`
  * **ALL numerical values** in code blocks: `71%`, `20m`, `1h 15m`, `15 of 21`
  * **ALL coverage status** bolded: `**FR1** (Beacon Detection):`, `**FR6** (RSSI Computation):`
  * **ALL archive fields** bolded: `**Resolution Date:**`, `**Verification:**`, `**Impact:**`
  * **ALL critical alerts** bold+italic combined: `***[PENDING - requires approval before execution]***`
  * **HIGHLIGHTING VALIDATION CHECKLIST** completed for entire document
- Dynamically maintained story.md file showing:
  * Active TODO tasks with approved requirements, DoD, and the correctly nested subtask decomposition with mandatory highlighting
  * **ABSOLUTE COMPLIANCE**: NO completed [x] tasks in ACTIVE TODO section
  * **ABSOLUTE COMPLIANCE**: NO resolved blockers in CURRENT BLOCKERS section
  * Parallel development metadata and coordination requirements with mandatory highlighting
  * User approval timestamps for each task with mandatory highlighting
  * Current blockers with ACTUAL error messages and parallel impact with mandatory highlighting
  * Completed work with real integration confirmation and subtask verification with mandatory highlighting
  * TDD compliance with Red-Green-Refactor documentation using authentic tests with mandatory highlighting
  * Chain of thought documentation with PRD references with mandatory highlighting
  * Integration boundary documentation for conditional tasks with mandatory highlighting
  * **COMPREHENSIVE VISUAL HIERARCHY**: All elements properly highlighted for immediate actionability and rapid scanning
- Post-task validation:
  * Code reviewed against approved Definition of Done criteria
  * Real system integration verified (no mocks)
  * Root problem confirmation (no duplicates/workarounds)
  * All subtasks verified complete before main task completion
  * Task completion determined based on DoD review
  * Test-first approach verified through commit history with authentic test verification
  * **ABSOLUTE REQUIREMENT**: Immediate transition to completed section executed with mandatory highlighting
  * **ABSOLUTE REQUIREMENT**: Immediate blocker resolution transfer executed with mandatory highlighting
  * **ABSOLUTE REQUIREMENT**: Mandatory markdown highlighting protocol applied throughout entire document
  * Document integrity validated - no stale entries confirmed
  * Highlighting validation checklist completed and verified
- Production-ready code with:
  * All tests written before implementation
  * Quality checks passing (NPX Trunk fmt, black, flake8, mypy, tsc)
  * NO mock or simulated components
  * Hardware requirements validated
  * Integration points verified to exist and work
  * Root problems fixed, not worked around
  * Complete subtask execution visibility maintained
  * Authentic test behavior verified throughout
- Clean git commit with test coverage reports and integration verification
- Work ready for senior architect and code review agent validation
- Brutal honesty maintained throughout: no theater, no pretending
- User situational awareness maintained via approval checkpoints
- **ABSOLUTE COMPLIANCE**: Task state management ensuring completed work immediately transitions to appropriate sections with mandatory highlighting
- **ABSOLUTE COMPLIANCE**: Blocker resolution management ensuring resolved blockers immediately move to archive with mandatory highlighting
- **ABSOLUTE COMPLIANCE**: Mandatory markdown highlighting protocol enforced throughout entire document for immediate actionability and visual hierarchy
- Parallel development coordination enabling efficient multi-developer workflows
- Conditional task integration boundary management for seamless developer handoffs
- Test authenticity enforcement preventing circumvention of TDD with fake tests
- Document integrity maintenance ensuring immediate actionability of active sections
- Visual hierarchy optimization enabling rapid project status assessment and critical information identification
- **KANBAN BOARD INTEGRATION**: Task completion properly reflected in TASKS.md with automatic board synchronization
- **TASK LIFECYCYCLE MANAGEMENT**: Proper movement of tasks through Backlog → To Do → In Progress → Completed sections with counter updates
</output>
