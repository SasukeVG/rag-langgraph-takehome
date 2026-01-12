import sys
from pathlib import Path

from loguru import logger

from config import settings
from app.graph import RAGGraph
from logging_config import setup_logging
from app.memory import SessionMemory
from app.retrieval import DocumentRetriever

setup_logging(level=settings.log_level)


def display_retrieved_chunks(documents, scores) -> None:
    if not documents:
        print("[INFO] No documents retrieved.")
        return

    print(f"\n[INFO] Retrieved {len(documents)} documents:\n")
    for i, (doc, score) in enumerate(zip(documents, scores), 1):
        source = doc.metadata.get("source", "unknown")
        filename = Path(source).name if source != "unknown" else "unknown"

        content_preview = doc.page_content[:200]
        if len(doc.page_content) > 200:
            content_preview += "..."

        print(f"  {i}. {filename} (distance: {score:.3f})")
        print(f"     {content_preview}\n")


def main() -> None:
    print("=" * 70)
    print("AI Engineer Take-Home: RAG Application")
    print("=" * 70)
    print("\nInitializing...")

    logger.debug(f"Settings: {settings.model_dump_json(indent=2)}")
    try:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / settings.retrieval.data_dir
        retriever = DocumentRetriever(data_dir=str(data_dir))
        graph = RAGGraph(retriever=retriever)
        memory = SessionMemory()

        logger.info("Application initialized successfully")
        print("Ready! Type 'help' for commands or 'quit' to exit.\n")
    except Exception as e:
        logger.exception(f"Failed to initialize application: {e}")
        print(f"ERROR: Failed to initialize application: {e}")
        sys.exit(1)

    print("Commands:")
    print("  ask <question>  - Ask a question")
    print("  help           - Show this help message")
    print("  quit / exit    - Exit the application")
    print()

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower() == "help":
                print("\nCommands:")
                print("  ask <question>  - Ask a question")
                print("  help           - Show this help message")
                print("  quit / exit    - Exit the application\n")
                continue

            if user_input.lower().startswith("ask "):
                query = user_input[4:].strip()
            else:
                query = user_input

            if not query:
                print("Please provide a question after 'ask' or just type your question.")
                continue

            history = memory.get_conversation_history()

            try:
                logger.info(f"Processing query: {query}")
                result = graph.invoke(query=query, messages=history)

                retrieved_docs = result.get("retrieved_docs", [])
                distances = result.get("distances", [])
                display_retrieved_chunks(retrieved_docs, distances)

                memory.add_user_message(query)

                answer = result.get("answer", "")
                if answer:
                    memory.add_ai_message(answer)
                    print()
                else:
                    print("[WARN] No answer generated.")

            except KeyboardInterrupt:
                print("\n\n[INFO] Interrupted by user.")
                continue
            except Exception as e:
                logger.exception(f"Error processing query: {e}")
                print(f"\n[ERROR] An error occurred: {e}")
                print("Please try again or type 'quit' to exit.\n")

        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            print(f"\n[ERROR] Unexpected error: {e}\n")


if __name__ == "__main__":
    main()
