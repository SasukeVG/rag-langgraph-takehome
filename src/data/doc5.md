# Streaming LLM Responses

Streaming allows applications to display model responses incrementally as tokens are generated, rather than waiting for the complete response.

## Benefits

- **Improved UX**: Users see progress immediately, reducing perceived latency
- **Real-time feedback**: Shows that the system is working, not frozen
- **Early insights**: Can display partial information while generation continues

## Implementation Patterns

### Token-by-token streaming
Display each token as it arrives from the API. Simple but can appear choppy.

### Chunk-based streaming
Accumulate tokens into words or phrases before displaying. Smoother visual experience.

### Incremental updates
Stream structured updates (e.g., "Retrieving...", "Found 3 docs...", "Generating answer...") along with tokens.

## Technical Considerations

- **Callback handlers**: Use LangChain callbacks to intercept streaming events
- **Error handling**: Handle mid-stream failures gracefully
- **Buffering**: Balance between responsiveness and output quality
- **Memory**: Streaming reduces memory footprint compared to buffering full responses

For OpenRouter, streaming is enabled via the `streaming=True` parameter and handled through standard LangChain streaming callbacks.


