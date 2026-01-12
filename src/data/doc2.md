# LangGraph Architecture

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with:

## Core Concepts

**State Graph**: A graph where each node represents a step in your workflow, and edges define how to transition between steps.

**Nodes**: Functions that receive the current state, perform some operation, and return an updated state.

**Edges**: Define the flow of control. Can be conditional (based on state) or unconditional.

**State**: A shared data structure that flows through the graph. Typically defined using TypedDict.

## Key Features

- **Conditional branching**: Routes to different nodes based on state values
- **Cycles and loops**: Allows for iterative workflows
- **Human-in-the-loop**: Can pause execution to wait for user input
- **Memory**: State persists across graph execution

## Example Pattern

```
[Start] → [Retrieve] → [Decision] → [Branch A or B] → [End]
```

This pattern is useful for RAG applications where you retrieve documents, evaluate relevance, and then either answer or ask for clarification.


