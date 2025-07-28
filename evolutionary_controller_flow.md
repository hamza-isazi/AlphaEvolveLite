# AlphaEvolveLite Evolutionary Controller Flow

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AlphaEvolveLite Evolutionary System                   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Config Files  │    │   Problem Def   │    │   LLM Models    │    │   Database      │
│                 │    │                 │    │                 │    │                 │
│ • config.yml    │    │ • entry_script  │    │ • OpenAI GPT-4  │    │ • SQLite DB     │
│ • problem_eval  │    │ • evaluate.py   │    │ • Claude-3      │    │ • ProgramStore  │
│ • evolution     │    │ • test cases    │    │ • Mixtral       │    │ • Experiment    │
│ • llm settings  │    │ • constraints   │    │ • probabilities │    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │
                                 ▼                       ▼
                    ┌─────────────────────────────────────────────────┐
                    │              EvolutionController                │
                    │                                                 │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │           Initialization                │    │
                    │  │  • Load seed program                   │    │
                    │  │  • Evaluate initial fitness            │    │
                    │  │  • Store in database                   │    │
                    │  │  • Generate feedback (optional)        │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Main Evolution Loop               │
                    │                                                 │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │         Generation Loop                 │    │
                    │  │  for gen in range(max_generations):     │    │
                    │  │    run_generation(gen)                  │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │            run_generation()                    │
                    │                                                 │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │         Population Generation           │    │
                    │  │  • Sample parents & inspirations        │    │
                    │  │  • Parallel program generation          │    │
                    │  │  • ThreadPoolExecutor (max 20 workers)  │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │           generate_program()                   │
                    │                                                 │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │         Individual Creation              │    │
                    │  │  • Create ProgramGenerationContext     │    │
                    │  │  • Sample parent & inspiration data    │    │
                    │  │  • Generate initial LLM response       │    │
                    │  │  • Parse explanation & code diff       │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Program Evaluation                │
                    │                                                 │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │         Evaluation Loop                 │    │
                    │  │  for retry in range(max_retries):       │    │
                    │  │    • Apply patch to parent code         │    │
                    │  │    • Compile for syntax check           │    │
                    │  │    • Write to temp file                 │    │
                    │  │    • Evaluate with timeout              │    │
                    │  │    • Handle errors & retry              │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Success Path                      │
                    │                                                 │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │         Post-Processing                 │    │
                    │  │  • Generate feedback (optional)        │    │
                    │  │  • Record metrics & conversation       │    │
                    │  │  • Store in database                   │    │
                    │  │  • Log generation summary              │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Final Results                     │
                    │                                                 │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │         save_top_k_candidates()         │    │
                    │  │  • Query top K programs from DB        │    │
                    │  │  • Save to results/ directory          │    │
                    │  │  • Generate experiment reports         │    │
                    │  └─────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────┘
```

## Detailed Component Flow

### 1. LLM Interaction Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PromptSampler │───▶│   LLMEngine     │───▶│   OpenAI API    │───▶│   Response      │
│                 │    │                 │    │                 │    │                 │
│ • Build prompts │    │ • Model select  │    │ • Chat complet. │    │ • Parse JSON    │
│ • Retry prompts │    │ • Conversation  │    │ • Token tracking│    │ • Extract code  │
│ • Feedback      │    │ • Metrics       │    │ • Timeout       │    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       │                       │                       │
         │                       ▼                       ▼                       ▼
         └───────────────────────────────────────────────────────────────────────┘
```

### 2. Code Evolution Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Parent Code   │───▶│   PatchApplier  │───▶│   Temp File     │───▶│   Problem       │
│                 │    │                 │    │                 │    │   Evaluator     │
│ • From DB       │    │ • Parse diff    │    │ • Write code    │    │ • Run tests     │
│ • Best fitness  │    │ • Apply changes │    │ • Execute       │    │ • Score calc    │
│ • Boltzmann sel │    │ • Indent fix    │    │ • Timeout       │    │ • Log capture   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       │                       │                       │
         │                       ▼                       ▼                       ▼
         └───────────────────────────────────────────────────────────────────────┘
```

### 3. Database Operations Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ProgramRecord │───▶│   Evolutionary  │───▶│   SQLite DB     │───▶│   Query Results │
│                 │    │   Database      │    │                 │    │                 │
│ • Code          │    │ • Add records   │    │ • Programs tbl  │    │ • Top K         │
│ • Score         │    │ • Sample parents│    │ • Experiments   │    │ • Random N      │
│ • Metrics       │    │ • Boltzmann sel │    │ • Indexes       │    │ • Statistics    │
│ • Conversation  │    │ • Statistics    │    │ • Foreign keys  │    │ • Export        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 4. Error Handling & Retry Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Initial Gen   │───▶│   Error Check   │───▶│   Retry Prompt  │───▶│   Re-evaluate   │
│                 │    │                 │    │                 │    │                 │
│ • LLM response  │    │ • Syntax error  │    │ • Error context │    │ • Apply patch   │
│ • Code parsing  │    │ • Runtime error │    │ • Failure type  │    │ • Compile check │
│ • Patch apply   │    │ • Timeout       │    │ • Retry count   │    │ • Score calc    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Success       │    │   Max Retries   │    │   Invalid Resp  │    │   Success       │
    │   • Store DB    │    │   • Log failure │    │   • Log failure │    │   • Store DB    │
    │   • Continue    │    │   • Continue    │    │   • Continue    │    │   • Continue    │
    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

### Parallel Processing
- **ThreadPoolExecutor**: Up to 20 concurrent program generations
- **Process isolation**: Each evaluation runs in separate process with timeout
- **Database thread safety**: Pre-sampled parent/inspiration data

### Robust Error Handling
- **Syntax validation**: Compile check before execution
- **Timeout protection**: Process-level timeouts for evaluations
- **Retry mechanism**: Multiple attempts with error context
- **Failure classification**: Detailed error types for analysis

### LLM Integration
- **Multi-model support**: Weighted model selection
- **Conversation tracking**: Full conversation history storage
- **Token monitoring**: Usage tracking and metrics
- **Feedback generation**: Optional post-evaluation feedback

### Data Management
- **SQLite database**: Persistent storage with foreign keys
- **Experiment tracking**: Multiple experiment support
- **Metrics collection**: Comprehensive performance tracking
- **Result export**: Top K candidates saved to filesystem

### Evolution Strategy
- **Boltzmann selection**: Temperature-based parent selection
- **Inspiration sampling**: Multiple parent programs for guidance
- **Fitness-based evolution**: Score-driven selection pressure
- **Generation tracking**: Complete lineage and genealogy 