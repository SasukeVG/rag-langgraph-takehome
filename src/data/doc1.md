# LangChain Overview

LangChain is a framework for developing applications powered by language models. It enables applications to:

- **Be context-aware**: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
- **Reason**: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

The main value props of LangChain are:

1. **Components**: abstractions for working with language models, along with a collection of implementations for each abstraction. Components are modular and easy-to-use, whether you use the rest of the LangChain framework or not.
2. **Off-the-shelf chains**: a structured assembly of components for accomplishing specific higher-level tasks. Chains make it easy to combine multiple components together to accomplish more complex tasks.

LangChain was designed to be extensible, so while the framework comes with many built-in components and chains, you can easily add your own.


