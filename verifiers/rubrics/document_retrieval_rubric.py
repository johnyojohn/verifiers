import json
from typing import Callable

from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State


class DocumentRetrievalRubric(Rubric):
    """Rubric that checks whether target documents were retrieved by the model.

    This rubric examines tool calls in the completion messages and verifies
    that specific documents (identified by their IDs) were accessed. Useful
    for document search Q&A environments where you want to assess retrieval quality.

    Args:
        tool: Tool that retrieves documents.
        arg_name: Name of the argument containing the document ID (e.g., "section_id").
        target_key: Key name for target document IDs (default: "target_documents").
            Recommended: Add an "info" column to your dataset with target docs nested inside.
            The rubric checks: state["info"][target_key] (recommended), then state["input"][target_key], then state[target_key].
        document_id_parser: Function to parse the document ID from the argument.(default: lambda x: x.split(":")[0])
    """

    def __init__(
        self,
        tool: Callable,
        arg_name: str = "section_id",
        target_key: str = "target_documents",
        document_id_parser: Callable[[str], str] = lambda x: x.split(":")[0],
    ):
        self.tool = tool
        self.arg_name = arg_name
        self.target_key = target_key
        self.document_id_parser = document_id_parser
        # Build reward functions
        reward_funcs = [
            self.retrieved_count,
            self.target_count,
            self.recall,
            self.precision,
        ]
        reward_weights = [0.0, 0.0, 0.0, 0.0]

        super().__init__(funcs=reward_funcs, weights=reward_weights)

    def _extract_retrieved_docs(self, completion: Messages) -> list[str]:
        """Extract document IDs from tool calls in completion messages."""
        retrieved = []
        assert isinstance(completion, list)
        for msg in completion:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("function", {}).get("name", "")
                        if tool_name == self.tool.__name__:
                            try:
                                args_str = tool_call.get("function", {}).get(
                                    "arguments", "{}"
                                )
                                args = json.loads(args_str)
                                if self.arg_name in args:
                                    doc_id = self.document_id_parser(
                                        args[self.arg_name]
                                    )
                                    retrieved.append(doc_id)
                            except (json.JSONDecodeError, KeyError):
                                continue
        return retrieved

    def _get_target_docs(self, state: State) -> list[str]:
        """Extract target document IDs from state.
        
        Access state["info"][target_key] where your dataset's "info" column data is stored.
        State's forwarding automatically handles state["info"] → state["input"]["info"].
        """
        target_docs = []
        
        # Use .get() to leverage State's forwarding behavior
        # state["info"] automatically forwards to state["input"]["info"]
        info = state.get("info")
        if isinstance(info, dict):
            target_docs = info.get(self.target_key, [])
        
        # Convert to list if needed
        if not isinstance(target_docs, list):
            if target_docs:  # Only warn if non-empty
                self.logger.warning(
                    f"Target documents must be a list, got {type(target_docs)}. Converting to list."
                )
                target_docs = [target_docs]
            else:
                target_docs = []

        # Apply document ID parser to each target
        result = [self.document_id_parser(doc) for doc in target_docs]
        return result

    async def retrieved_count(self, completion: Messages) -> float:
        """Count how many documents were retrieved by the model."""
        retrieved = self._extract_retrieved_docs(completion)
        return float(len(set(retrieved)))

    async def target_count(self, state: State) -> float:
        """Count how many target documents should have been retrieved."""
        target = self._get_target_docs(state)
        return float(len(set(target)))

    async def recall(self, completion: Messages, state: State) -> float:
        """Calculate recall: fraction of target documents that were retrieved."""
        retrieved = set(self._extract_retrieved_docs(completion))
        target = set(self._get_target_docs(state))

        if not target:
            return 1.0  # No targets, perfect recall

        overlap = len(retrieved & target)
        return float(overlap) / len(target)

    async def precision(self, completion: Messages, state: State) -> float:
        """Calculate precision: fraction of retrieved documents that were targets."""
        retrieved = set(self._extract_retrieved_docs(completion))
        target = set(self._get_target_docs(state))

        if not retrieved:
            return 0.0  # No retrievals, zero precision

        overlap = len(retrieved & target)
        return float(overlap) / len(retrieved)
